"""Evaluate a shared Blind checkpoint with one family decision replaced by an expert."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .blind2 import LABELS, BlindFamilyPresence, BlindPresence, log_mel
from .blind_data2 import BlindPresenceData, inventory
from .train_blind2 import SEED, _acceptance, _loader, _metrics, _probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--extra-expert", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    target_device = torch.device(args.device)
    base_checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    expert_checkpoint = torch.load(args.expert, map_location="cpu", weights_only=False)
    family = expert_checkpoint["manifest"]["family"]
    family_index = LABELS.index(family)
    base = BlindPresence().to(target_device)
    base.load_state_dict(base_checkpoint["model"])
    expert = BlindFamilyPresence(family).to(target_device)
    expert.load_state_dict(expert_checkpoint["model"])
    extra_checkpoint = None
    extra_expert = None
    extra_family = None
    if args.extra_expert is not None:
        extra_checkpoint = torch.load(args.extra_expert, map_location="cpu", weights_only=False)
        extra_family = extra_checkpoint["manifest"]["family"]
        if extra_family in {family, "any"}:
            raise ValueError("extra expert must replace a different concrete family")
        extra_expert = BlindFamilyPresence(extra_family).to(target_device)
        extra_expert.load_state_dict(extra_checkpoint["model"])
    gate_checkpoint = None
    gate = None
    if args.gate is not None:
        gate_checkpoint = torch.load(args.gate, map_location="cpu", weights_only=False)
        if gate_checkpoint["manifest"]["family"] != "any":
            raise ValueError("gate checkpoint must be an any-effect family expert")
        gate = BlindFamilyPresence("any").to(target_device)
        gate.load_state_dict(gate_checkpoint["model"])
    data = BlindPresenceData(Path.cwd(), "development", 640, SEED + 37)
    expected_rows, source_rows, base_rows, expert_rows, extra_rows, gate_rows = [], [], [], [], [], []
    base.eval()
    expert.eval()
    if extra_expert is not None:
        extra_expert.eval()
    if gate is not None:
        gate.eval()
    with torch.no_grad():
        for batch in _loader(data, 16, False):
            features = log_mel(batch["audio"].to(target_device))
            base_rows.append(base(features).cpu().numpy())
            expert_rows.append(expert(features).cpu().numpy())
            if extra_expert is not None:
                extra_rows.append(extra_expert(features).cpu().numpy())
            if gate is not None:
                gate_rows.append(gate(features).cpu().numpy())
            expected_rows.append(batch["target"].numpy())
            source_rows.extend(batch["source"])
    expected = np.concatenate(expected_rows)
    base_probabilities = _probabilities(
        np.concatenate(base_rows), base_checkpoint["calibration"]
    )
    expert_probabilities = _probabilities(
        np.concatenate(expert_rows), expert_checkpoint["calibration"]
    )
    probabilities = base_probabilities.copy()
    expert_weight = float(expert_checkpoint.get("expert_blend_weight", 1.0))
    base_weight = float(expert_checkpoint.get("base_blend_weight", 0.0))
    if abs(base_weight + expert_weight - 1.0) > 1.0e-6:
        raise ValueError("base/expert blend weights must sum to one")
    probabilities[:, family_index] = (
        base_weight * base_probabilities[:, family_index]
        + expert_weight * expert_probabilities[:, 0]
    )
    extra_report = None
    if extra_checkpoint is not None and extra_family is not None:
        extra_index = LABELS.index(extra_family)
        extra_probabilities = _probabilities(
            np.concatenate(extra_rows), extra_checkpoint["calibration"]
        )[:, 0]
        extra_expert_weight = float(extra_checkpoint.get("expert_blend_weight", 1.0))
        extra_base_weight = float(extra_checkpoint.get("base_blend_weight", 0.0))
        if abs(extra_base_weight + extra_expert_weight - 1.0) > 1.0e-6:
            raise ValueError("extra expert blend weights must sum to one")
        probabilities[:, extra_index] = (
            extra_base_weight * base_probabilities[:, extra_index]
            + extra_expert_weight * extra_probabilities
        )
        extra_report = {
            "family": extra_family,
            "path": str(args.extra_expert.resolve()),
            "sha256": hashlib.sha256(args.extra_expert.read_bytes()).hexdigest(),
        }
    gate_report = None
    if gate_checkpoint is not None:
        gate_expert_probabilities = _probabilities(
            np.concatenate(gate_rows), gate_checkpoint["calibration"]
        )[:, 0]
        gate_expert_weight = float(gate_checkpoint.get("expert_blend_weight", 1.0))
        gate_base_weight = float(gate_checkpoint.get("base_blend_weight", 0.0))
        if abs(gate_base_weight + gate_expert_weight - 1.0) > 1.0e-6:
            raise ValueError("gate blend weights must sum to one")
        gate_probabilities = (
            gate_base_weight * base_probabilities.max(axis=1)
            + gate_expert_weight * gate_expert_probabilities
        )
        gate_active = gate_probabilities >= float(gate_checkpoint["threshold"])
        if "base_fallback_threshold" in gate_checkpoint:
            gate_active |= base_probabilities.max(axis=1) >= float(
                gate_checkpoint["base_fallback_threshold"]
            )
        if "family_fallback" in gate_checkpoint:
            fallback = gate_checkpoint["family_fallback"]
            gate_active |= base_probabilities[:, LABELS.index(fallback["family"])] >= float(
                fallback["threshold"]
            )
        effect_present = expected.astype(bool).any(axis=1)
        probabilities[~gate_active] = 0.0
        gate_report = {
            "threshold": float(gate_checkpoint["threshold"]),
            "base_fallback_threshold": gate_checkpoint.get("base_fallback_threshold"),
            "family_fallback": gate_checkpoint.get("family_fallback"),
            "effect_recall": float(np.mean(gate_active[effect_present])),
            "clean_false_positive_rate": float(np.mean(gate_active[~effect_present])),
            "path": str(args.gate.resolve()),
            "sha256": hashlib.sha256(args.gate.read_bytes()).hexdigest(),
        }
    thresholds = list(base_checkpoint["thresholds"])
    thresholds[family_index] = float(expert_checkpoint["threshold"])
    if extra_checkpoint is not None and extra_family is not None:
        thresholds[LABELS.index(extra_family)] = float(extra_checkpoint["threshold"])
    metrics = _metrics(probabilities, expected, thresholds, source_rows)
    report = {
        "schema": 1,
        "status": "development-accepted-not-promoted" if _acceptance(metrics)["accepted"] else "development-rejected",
        "device": str(target_device),
        "family_replaced": family,
        "family_blend": {"base": base_weight, "expert": expert_weight},
        "extra_expert": extra_report,
        "any_effect_gate": gate_report,
        "development": metrics,
        "acceptance": _acceptance(metrics),
        "thresholds": dict(zip(LABELS, thresholds, strict=True)),
        "inventory": inventory(Path.cwd()),
        "artifacts": {
            "base": {
                "path": str(args.base.resolve()),
                "sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
            },
            "expert": {
                "path": str(args.expert.resolve()),
                "sha256": hashlib.sha256(args.expert.read_bytes()).hexdigest(),
            },
        },
        "boundaries": {
            "order_labels_used": False,
            "controls_used": False,
            "locked_final_opened": False,
            "weights_changed": False,
            "family_decisions_independent": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
