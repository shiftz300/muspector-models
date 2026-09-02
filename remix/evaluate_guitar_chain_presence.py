"""Evaluate a frozen Blind model on the independent random-position chain corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile
import torch
from scipy.signal import resample_poly

from .blind2 import (
    LABELS,
    RATE,
    WINDOW,
    BlindFamilyPresence,
    BlindPresence,
    aggregate_windows,
    log_mel,
)
from .guitar_chain_data import SOURCE_ID, discover, inventory
from .train_blind2 import _metrics


def _device(name: str) -> torch.device:
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    return torch.device("cpu")


def _windows(path: Path) -> torch.Tensor:
    audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
    value = audio.mean(axis=1)
    if rate != RATE:
        divisor = math.gcd(rate, RATE)
        value = resample_poly(value, RATE // divisor, rate // divisor)
    value = np.asarray(value, dtype=np.float32)
    if len(value) < WINDOW:
        value = np.pad(value, (0, WINDOW - len(value)))
    maximum = len(value) - WINDOW
    starts = sorted({0, maximum // 2, maximum})
    return torch.stack([torch.from_numpy(value[start : start + WINDOW].copy()) for start in starts])


def _probabilities(
    model: BlindPresence,
    windows: torch.Tensor,
    calibration: list[dict],
    target: torch.device,
    aggregation: str = "top-two",
) -> np.ndarray:
    with torch.no_grad():
        logits = model(log_mel(windows.to(target)))
        scale = torch.tensor(
            [row["scale"] for row in calibration], dtype=logits.dtype, device=target
        )
        bias = torch.tensor(
            [row["bias"] for row in calibration], dtype=logits.dtype, device=target
        )
        calibrated = torch.sigmoid(logits * scale + bias)
        if aggregation == "top-two":
            aggregated = aggregate_windows(calibrated)
        elif aggregation == "mean":
            aggregated = calibrated.mean(dim=0)
        else:
            raise ValueError(f"unsupported file aggregation: {aggregation}")
        return aggregated.cpu().numpy()


def _acceptance(overall: dict, multi_effect: dict, guitars: dict[str, dict]) -> dict:
    gates = {
        "micro_f1": overall["micro_f1"] >= 0.80,
        "macro_f1": overall["macro_f1"] >= 0.80,
        "each_label_recall": all(row["recall"] >= 0.80 for row in overall["per_label"].values()),
        "clean_false_positive_rate": overall["clean_false_positive_rate"] is not None
        and overall["clean_false_positive_rate"] <= 0.05,
        "multi_effect_macro_f1": multi_effect["macro_f1"] >= 0.80,
        "each_guitar_macro_f1": bool(guitars)
        and all(metrics["macro_f1"] >= 0.70 for metrics in guitars.values()),
    }
    return {
        "external_software_chain_veto_passed": all(gates.values()),
        "gates": gates,
        "product_promotion_allowed": False,
        "remaining_blocker": "a real multi-effect physical-hardware veto and listening still remain",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expert", type=Path)
    parser.add_argument("--extra-expert", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/guitar-effects-chains"))
    parser.add_argument("--split", choices=("fit", "calibration", "development"), default="development")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["manifest"]["architecture"] != "compact-audio-resnet18":
        raise ValueError("this evaluator requires a compact Blind checkpoint")
    model = BlindPresence()
    model.load_state_dict(checkpoint["model"])
    target = _device(args.device)
    model.to(target).eval()
    expert_checkpoint = None
    expert = None
    family_index = None
    if args.expert is not None:
        expert_checkpoint = torch.load(args.expert, map_location="cpu", weights_only=False)
        family = expert_checkpoint["manifest"]["family"]
        family_index = LABELS.index(family)
        expert = BlindFamilyPresence(family)
        expert.load_state_dict(expert_checkpoint["model"])
        expert.to(target).eval()
    extra_checkpoint = None
    extra_expert = None
    extra_family_index = None
    if args.extra_expert is not None:
        extra_checkpoint = torch.load(args.extra_expert, map_location="cpu", weights_only=False)
        extra_family = extra_checkpoint["manifest"]["family"]
        if extra_family == "any" or (expert_checkpoint is not None and extra_family == expert_checkpoint["manifest"]["family"]):
            raise ValueError("extra expert must replace a different concrete family")
        extra_family_index = LABELS.index(extra_family)
        extra_expert = BlindFamilyPresence(extra_family)
        extra_expert.load_state_dict(extra_checkpoint["model"])
        extra_expert.to(target).eval()
    gate_checkpoint = None
    gate = None
    if args.gate is not None:
        gate_checkpoint = torch.load(args.gate, map_location="cpu", weights_only=False)
        if gate_checkpoint["manifest"]["family"] != "any":
            raise ValueError("gate checkpoint must be an any-effect family expert")
        gate = BlindFamilyPresence("any")
        gate.load_state_dict(gate_checkpoint["model"])
        gate.to(target).eval()

    rows = discover(args.corpus, args.split)
    if not rows:
        raise ValueError(f"empty guitar-chain split: {args.split}")
    probability_rows = []
    for row in rows:
        windows = _windows(row.path)
        row_probabilities = _probabilities(
            model, windows, checkpoint["calibration"], target
        )
        base_row_probabilities = row_probabilities.copy()
        base_gate_probability = float(np.max(row_probabilities))
        if expert is not None and expert_checkpoint is not None and family_index is not None:
            expert_probability = _probabilities(
                expert,
                windows,
                expert_checkpoint["calibration"],
                target,
                expert_checkpoint.get("file_aggregation", "top-two"),
            )[0]
            expert_weight = float(expert_checkpoint.get("expert_blend_weight", 1.0))
            base_weight = float(expert_checkpoint.get("base_blend_weight", 0.0))
            if abs(base_weight + expert_weight - 1.0) > 1.0e-6:
                raise ValueError("base/expert blend weights must sum to one")
            row_probabilities[family_index] = (
                base_weight * row_probabilities[family_index]
                + expert_weight * expert_probability
            )
        if (
            extra_expert is not None
            and extra_checkpoint is not None
            and extra_family_index is not None
        ):
            extra_probability = _probabilities(
                extra_expert,
                windows,
                extra_checkpoint["calibration"],
                target,
                extra_checkpoint.get("file_aggregation", "top-two"),
            )[0]
            extra_expert_weight = float(extra_checkpoint.get("expert_blend_weight", 1.0))
            extra_base_weight = float(extra_checkpoint.get("base_blend_weight", 0.0))
            row_probabilities[extra_family_index] = (
                extra_base_weight * row_probabilities[extra_family_index]
                + extra_expert_weight * extra_probability
            )
        if gate is not None and gate_checkpoint is not None:
            gate_expert_probability = _probabilities(
                gate,
                windows,
                gate_checkpoint["calibration"],
                target,
                gate_checkpoint.get("file_aggregation", "top-two"),
            )[0]
            gate_expert_weight = float(gate_checkpoint.get("expert_blend_weight", 1.0))
            gate_base_weight = float(gate_checkpoint.get("base_blend_weight", 0.0))
            gate_probability = (
                gate_base_weight * base_gate_probability
                + gate_expert_weight * gate_expert_probability
            )
            gate_active = gate_probability >= float(gate_checkpoint["threshold"])
            if "base_fallback_threshold" in gate_checkpoint:
                gate_active = gate_active or base_gate_probability >= float(
                    gate_checkpoint["base_fallback_threshold"]
                )
            if "family_fallback" in gate_checkpoint:
                fallback = gate_checkpoint["family_fallback"]
                gate_active = gate_active or base_row_probabilities[
                    LABELS.index(fallback["family"])
                ] >= float(fallback["threshold"])
            if not gate_active:
                row_probabilities[:] = 0.0
        probability_rows.append(row_probabilities)
    probabilities = np.stack(probability_rows)
    expected = np.stack([row.target for row in rows])
    thresholds = [float(value) for value in checkpoint["thresholds"]]
    if expert_checkpoint is not None and family_index is not None:
        thresholds[family_index] = float(expert_checkpoint["threshold"])
    if extra_checkpoint is not None and extra_family_index is not None:
        thresholds[extra_family_index] = float(extra_checkpoint["threshold"])
    sources = [f"guitar:{row.guitar}" for row in rows]
    overall = _metrics(probabilities, expected, thresholds, sources)
    multi_indices = np.flatnonzero(expected.sum(axis=1) >= 2)
    multi_effect = _metrics(
        probabilities[multi_indices],
        expected[multi_indices],
        thresholds,
        [sources[index] for index in multi_indices],
    )
    guitars = {
        guitar: _metrics(
            probabilities[[index for index, row in enumerate(rows) if row.guitar == guitar]],
            expected[[index for index, row in enumerate(rows) if row.guitar == guitar]],
            thresholds,
            [sources[index] for index, row in enumerate(rows) if row.guitar == guitar],
        )
        for guitar in sorted({row.guitar for row in rows})
    }
    report = {
        "schema": 1,
        "source_id": SOURCE_ID,
        "scope": "presence-only external random-position software-chain veto; no order supervision",
        "split": args.split,
        "device": str(target),
        "examples": len(rows),
        "groups": len({row.group for row in rows}),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        },
        "expert": None if args.expert is None else {
            "path": str(args.expert.resolve()),
            "sha256": hashlib.sha256(args.expert.read_bytes()).hexdigest(),
            "family": expert_checkpoint["manifest"]["family"],
            "base_blend_weight": expert_checkpoint.get("base_blend_weight", 0.0),
            "expert_blend_weight": expert_checkpoint.get("expert_blend_weight", 1.0),
            "file_aggregation": expert_checkpoint.get("file_aggregation", "top-two"),
        },
        "gate": None if args.gate is None else {
            "path": str(args.gate.resolve()),
            "sha256": hashlib.sha256(args.gate.read_bytes()).hexdigest(),
            "threshold": gate_checkpoint["threshold"],
            "base_fallback_threshold": gate_checkpoint.get("base_fallback_threshold"),
            "family_fallback": gate_checkpoint.get("family_fallback"),
            "file_aggregation": gate_checkpoint.get("file_aggregation", "top-two"),
        },
        "extra_expert": None if args.extra_expert is None else {
            "path": str(args.extra_expert.resolve()),
            "sha256": hashlib.sha256(args.extra_expert.read_bytes()).hexdigest(),
            "family": extra_checkpoint["manifest"]["family"],
        },
        "overall": overall,
        "multi_effect": multi_effect,
        "guitars": guitars,
        "acceptance": _acceptance(overall, multi_effect, guitars),
        "inventory": inventory(args.corpus),
        "boundaries": {
            "presence_labels_only": True,
            "order_labels_used": False,
            "controls_used": False,
            "restoration_gradients_used": False,
            "locked_final_opened": False,
            "physical_hardware_claim": False,
            "family_decisions_independent": expert is not None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
