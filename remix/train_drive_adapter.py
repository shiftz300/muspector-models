#!/usr/bin/env python3
"""Train a tiny Drive adapter from aligned, device-local calibration pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import dry_sources
from .drive_adapter import DriveDeviceAdapter
from .forward_chain import DRIVE_CHECKPOINT, _load_drive
from .forward_drive import FORWARD_RATE, _latin_hypercube, _segment, forward_loss, normalized_drive_controls
from .render import render_chain
from .spec import ChainSpec, Delay, Drive, Reverb
from .train import CORPUS, device


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/drive-adapter-pilot"
CONTEXTS = ("clean", "delay", "reverb", "delay-reverb", "reverb-delay")


def _upstream(rng: random.Random, context: str) -> ChainSpec:
    effects = []
    for kind in context.split("-") if context != "clean" else ():
        if kind == "delay":
            effects.append(
                Delay(
                    40.0 * 25.0 ** rng.random(),
                    rng.uniform(0.05, 0.82),
                    rng.uniform(0.08, 0.65),
                )
            )
        elif kind == "reverb":
            effects.append(
                Reverb(
                    0.2 * 40.0 ** rng.random(),
                    rng.uniform(0.05, 0.95),
                    rng.uniform(0.08, 0.65),
                )
            )
    return ChainSpec(tuple(effects))


@torch.no_grad()
def _examples(
    paths: list[Path],
    base: torch.nn.Module,
    count: int,
    frames: int,
    seed: int,
    domain: str,
) -> list[dict]:
    controls = _latin_hypercube(count, 3, seed ^ 0x41445054)
    result = []
    for index in range(count):
        rng = random.Random(seed + index * 104_729)
        source = _segment(paths[index % len(paths)], frames, rng)
        peak = max(float(np.max(np.abs(source))), 1.0e-6)
        source = np.asarray(source * (rng.uniform(0.04, 0.42) / peak), dtype=np.float32)
        context = CONTEXTS[index % len(CONTEXTS)]
        upstream = _upstream(rng, context)
        stage_input = (
            render_chain(source, upstream, FORWARD_RATE, domain)
            if upstream.effects
            else source.copy()
        )
        values = controls[index]
        effect = Drive(
            float(values[0]) * 30.0,
            float(values[1]),
            float(values[2]) * 30.0 - 18.0,
        )
        control = torch.from_numpy(normalized_drive_controls(effect)).unsqueeze(0)
        dry = torch.from_numpy(stage_input.copy()).unsqueeze(0)
        generic, _ = base(dry, control)
        target = render_chain(stage_input, ChainSpec((effect,)), FORWARD_RATE, domain)
        result.append(
            {
                "dry": dry.squeeze(0),
                "base": generic.squeeze(0),
                "wet": torch.from_numpy(target.copy()),
                "controls": control.squeeze(0),
                "context": context,
            }
        )
    return result


@torch.no_grad()
def _evaluate(adapter: DriveDeviceAdapter, loader: DataLoader, target: torch.device) -> dict:
    adapter.eval()
    totals = defaultdict(lambda: {"base": 0.0, "model": 0.0, "count": 0})
    peak_ratios = []
    for batch in loader:
        dry = batch["dry"].to(target)
        base = batch["base"].to(target)
        wet = batch["wet"].to(target)
        controls = batch["controls"].to(target)
        prediction, _ = adapter(dry, base, controls)
        for index, context in enumerate(batch["context"]):
            energy = wet[index].square().mean().clamp_min(1.0e-8)
            totals[context]["base"] += float((base[index] - wet[index]).square().mean() / energy)
            totals[context]["model"] += float((prediction[index] - wet[index]).square().mean() / energy)
            totals[context]["count"] += 1
        peak_ratios.extend(
            (prediction.abs().amax(dim=1) / wet.abs().amax(dim=1).clamp_min(1.0e-6)).cpu().tolist()
        )
    groups = {}
    for name, values in totals.items():
        base_esr = values["base"] / values["count"]
        model_esr = values["model"] / values["count"]
        groups[name] = {
            "base_esr": base_esr,
            "model_esr": model_esr,
            "relative_improvement": 1.0 - model_esr / max(base_esr, 1.0e-12),
            "examples": values["count"],
        }
    ordered = sorted(float(value) for value in peak_ratios)
    return {
        "groups": groups,
        "mean_base_esr": float(np.mean([value["base_esr"] for value in groups.values()])),
        "mean_model_esr": float(np.mean([value["model_esr"] for value in groups.values()])),
        "mean_relative_improvement": float(
            np.mean([value["relative_improvement"] for value in groups.values()])
        ),
        "worst_context_improvement": min(
            value["relative_improvement"] for value in groups.values()
        ),
        "peak_ratio_p95": ordered[max(0, int(np.ceil(0.95 * len(ordered))) - 1)],
        "worst_peak_ratio": ordered[-1],
    }


def _stream_parity(adapter: DriveDeviceAdapter) -> dict:
    rng = np.random.default_rng(20270204)
    dry = torch.from_numpy((rng.standard_normal((1, 8_207)) * 0.04).astype(np.float32))
    base = torch.tanh(dry * 2.1) * 0.7
    controls = torch.tensor(((0.73, 0.26, 0.61),), dtype=torch.float32)
    with torch.inference_mode():
        complete, _ = adapter(dry, base, controls)
        state = None
        pieces = []
        for start in range(0, dry.shape[1], 1_024):
            piece, state = adapter(
                dry[:, start : start + 1_024],
                base[:, start : start + 1_024],
                controls,
                state,
            )
            pieces.append(piece)
        streamed = torch.cat(pieces, dim=1)
        silence, _ = adapter(torch.zeros_like(dry), torch.zeros_like(base), controls)
    difference = streamed.double() - complete.double()
    return {
        "max_absolute_error": float(torch.max(torch.abs(difference))),
        "rms_error": float(torch.sqrt(torch.mean(torch.square(difference)))),
        "silence_max_absolute_output": float(torch.max(torch.abs(silence))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--base", type=Path, default=DRIVE_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--domain", default="final_challenge")
    parser.add_argument("--calibration-examples", type=int, default=320)
    parser.add_argument("--validation-examples", type=int, default=160)
    parser.add_argument("--frames", type=int, default=8_192)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--peak-penalty-weight", type=float, default=4.0)
    args = parser.parse_args()
    random.seed(20270201)
    np.random.seed(20270201)
    torch.manual_seed(20270201)
    target = device()
    base = _load_drive(args.base).eval()
    calibration_rows = _examples(
        dry_sources(args.corpus, "calibrate"),
        base,
        args.calibration_examples,
        args.frames,
        20270201,
        args.domain,
    )
    validation_rows = _examples(
        dry_sources(args.corpus, "valid"),
        base,
        args.validation_examples,
        args.frames,
        20270202,
        args.domain,
    )
    final_paths = [
        path for path in dry_sources(args.corpus, "train") if path.name.startswith("prs_")
    ]
    if not final_paths:
        raise ValueError("adapter-heldout PRS sources are missing")
    final_rows = _examples(
        final_paths,
        base,
        args.validation_examples,
        args.frames,
        20270203,
        args.domain,
    )
    calibration = DataLoader(calibration_rows, batch_size=args.batch_size, shuffle=True)
    validation = DataLoader(validation_rows, batch_size=args.batch_size)
    adapter = DriveDeviceAdapter().to(target)
    if args.init_adapter is not None:
        initial = torch.load(args.init_adapter, map_location="cpu", weights_only=True)
        if initial.get("schema") != 1 or initial.get("sample_rate") != FORWARD_RATE:
            raise ValueError("initial Drive adapter is incompatible")
        adapter.load_state_dict(initial["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=2.0e-4, weight_decay=1.0e-6)
    best_score = float("inf")
    best_state = None
    history = []
    for epoch in range(args.epochs):
        adapter.train()
        total = 0.0
        batches = 0
        for batch in calibration:
            dry = batch["dry"].to(target)
            generic = batch["base"].to(target)
            wet = batch["wet"].to(target)
            controls = batch["controls"].to(target)
            prediction, _ = adapter(dry, generic, controls)
            loss, _ = forward_loss(prediction, wet)
            peak_ratio = prediction.abs().amax(dim=1) / wet.abs().amax(dim=1).clamp_min(1.0e-6)
            peak_penalty = torch.relu(peak_ratio - 1.10).square().mean()
            loss = loss + args.peak_penalty_weight * peak_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        metrics = _evaluate(adapter, DataLoader(calibration_rows, batch_size=args.batch_size), target)
        row = {
            "epoch": epoch + 1,
            "train_loss": total / max(batches, 1),
            "calibration_mean_model_esr": metrics["mean_model_esr"],
            "calibration_improvement": metrics["mean_relative_improvement"],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if metrics["mean_model_esr"] < best_score:
            best_score = metrics["mean_model_esr"]
            best_state = {
                name: value.detach().cpu().clone() for name, value in adapter.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Drive adapter training produced no checkpoint")
    adapter.load_state_dict(best_state)
    validation_metrics = _evaluate(adapter, validation, target)
    final_metrics = _evaluate(
        adapter,
        DataLoader(final_rows, batch_size=args.batch_size),
        target,
    )
    runtime = _stream_parity(adapter.cpu().eval())
    promotable = bool(
        final_metrics["mean_relative_improvement"] >= 0.20
        and final_metrics["worst_context_improvement"] >= 0.0
        and final_metrics["peak_ratio_p95"] <= 1.35
        and final_metrics["worst_peak_ratio"] <= 1.75
        and runtime["max_absolute_error"] <= 2.0e-6
        and runtime["silence_max_absolute_output"] == 0.0
    )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "drive-device-adapter.pt"
    torch.save(
        {
            "schema": 1,
            "sample_rate": FORWARD_RATE,
            "hidden_size": adapter.hidden_size,
            "state_dict": best_state,
            "base_checkpoint_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
            "device_domain": args.domain,
        },
        checkpoint,
    )
    report = {
        "schema": 1,
        "status": "accepted-synthetic-device-adapter" if promotable else "rejected",
        "promotable": promotable,
        "scope": "synthetic device-domain proof; not a real hardware fidelity claim",
        "domain": args.domain,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "base_checkpoint": str(args.base),
        "base_checkpoint_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
        "calibration_split": "Strat 13-25",
        "validation_split": "Strat 1-12",
        "adapter_final_split": "PRS sources; unseen by adapter training and selection",
        "split_disjoint": True,
        "contexts": list(CONTEXTS),
        "validation": validation_metrics,
        "adapter_final": final_metrics,
        "runtime": runtime,
        "history": history,
        "policy": {
            "base_model_frozen": True,
            "source_files_read_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "locked_tele_test_opened": False,
            "physical_audio_devices_used": False,
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"promotable": promotable, "validation": validation_metrics, "adapter_final": final_metrics, "runtime": runtime}, sort_keys=True))
    if not promotable:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
