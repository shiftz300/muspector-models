#!/usr/bin/env python3
"""Extended guitar-disjoint audit for a Drive forward-renderer candidate."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import dry_sources
from .forward_drive import (
    FORWARD_RATE,
    DriveForwardDataset,
    DriveForwardRenderer,
    error_to_signal_ratio,
    multiresolution_spectral_loss,
)
from .train import CORPUS, device
from .train_forward_drive import VALIDATION_DOMAINS


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/drive-forward-pilot"


def chunked_render(
    model: DriveForwardRenderer,
    dry: torch.Tensor,
    controls: torch.Tensor,
    chunk_frames: int,
) -> torch.Tensor:
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    state = None
    rendered = []
    for start in range(0, dry.shape[1], chunk_frames):
        value, state = model(dry[:, start : start + chunk_frames], controls, state)
        rendered.append(value)
    return torch.cat(rendered, dim=1)


@torch.no_grad()
def detailed_evaluate(
    model: DriveForwardRenderer,
    loader: DataLoader,
    target_device: torch.device,
    chunk_frames: int,
) -> dict:
    model.eval()
    totals = defaultdict(float)
    bins: dict[str, dict[int, dict[str, float]]] = {
        name: {index: defaultdict(float) for index in range(4)}
        for name in ("gain", "tone", "level")
    }
    examples = 0
    finite = True
    peak_ratios = []
    for batch in loader:
        dry = batch["dry"].to(target_device)
        wet = batch["wet"].to(target_device)
        controls = batch["controls"].to(target_device)
        prediction = chunked_render(model, dry, controls, chunk_frames)
        finite = finite and bool(torch.isfinite(prediction).all())
        model_error = (prediction - wet).square().mean(dim=1)
        baseline_error = (dry - wet).square().mean(dim=1)
        target_energy = wet.square().mean(dim=1).clamp_min(1.0e-8)
        model_esr = model_error / target_energy
        baseline_esr = baseline_error / target_energy
        model_mae = torch.abs(prediction - wet).mean(dim=1)
        baseline_mae = torch.abs(dry - wet).mean(dim=1)
        batch_examples = dry.shape[0]
        examples += batch_examples
        totals["model_esr"] += float(model_esr.sum())
        totals["baseline_esr"] += float(baseline_esr.sum())
        totals["model_mae"] += float(model_mae.sum())
        totals["baseline_mae"] += float(baseline_mae.sum())
        totals["model_spectral"] += float(
            multiresolution_spectral_loss(prediction, wet)
        ) * batch_examples
        totals["baseline_spectral"] += float(
            multiresolution_spectral_loss(dry, wet)
        ) * batch_examples
        totals["model_peak"] = max(totals["model_peak"], float(prediction.abs().max()))
        totals["target_peak"] = max(totals["target_peak"], float(wet.abs().max()))
        peak_ratios.extend(
            (
                prediction.abs().amax(dim=1)
                / wet.abs().amax(dim=1).clamp_min(1.0e-6)
            ).cpu().tolist()
        )
        for dimension, name in enumerate(("gain", "tone", "level")):
            indexes = torch.clamp((controls[:, dimension] * 4.0).long(), max=3)
            for bin_index in range(4):
                selected = indexes == bin_index
                count = int(selected.sum())
                if not count:
                    continue
                bucket = bins[name][bin_index]
                bucket["examples"] += count
                bucket["model_esr"] += float(model_esr[selected].sum())
                bucket["baseline_esr"] += float(baseline_esr[selected].sum())
                bucket["model_mae"] += float(model_mae[selected].sum())
                bucket["baseline_mae"] += float(baseline_mae[selected].sum())
    metrics = {
        name: value / examples
        for name, value in totals.items()
        if name not in {"model_peak", "target_peak"}
    }
    metrics["model_peak"] = totals["model_peak"]
    metrics["target_peak"] = totals["target_peak"]
    metrics["examples"] = examples
    metrics["finite"] = finite
    ordered_peak_ratios = sorted(float(value) for value in peak_ratios)
    percentile_index = min(
        len(ordered_peak_ratios) - 1,
        max(0, math.ceil(0.95 * len(ordered_peak_ratios)) - 1),
    )
    metrics["peak_ratio_p95"] = ordered_peak_ratios[percentile_index]
    metrics["worst_peak_ratio"] = ordered_peak_ratios[-1]
    metrics["esr_improvement"] = 1.0 - metrics["model_esr"] / max(
        metrics["baseline_esr"], 1.0e-12
    )
    metrics["mae_improvement"] = 1.0 - metrics["model_mae"] / max(
        metrics["baseline_mae"], 1.0e-12
    )
    metrics["spectral_improvement"] = 1.0 - metrics["model_spectral"] / max(
        metrics["baseline_spectral"], 1.0e-12
    )
    worst_bin_ratio = 0.0
    bin_report = {}
    for name, values in bins.items():
        bin_report[name] = {}
        for index, bucket in values.items():
            count = max(int(bucket["examples"]), 1)
            report = {
                key: value / count
                for key, value in bucket.items()
                if key != "examples"
            }
            report["examples"] = int(bucket["examples"])
            report["esr_ratio"] = report["model_esr"] / max(
                report["baseline_esr"], 1.0e-12
            )
            report["mae_ratio"] = report["model_mae"] / max(
                report["baseline_mae"], 1.0e-12
            )
            worst_bin_ratio = max(
                worst_bin_ratio,
                report["esr_ratio"],
                report["mae_ratio"],
            )
            bin_report[name][str(index)] = report
    metrics["control_quartiles"] = bin_report
    metrics["worst_control_bin_ratio"] = worst_bin_ratio
    metrics["passed"] = bool(
        finite
        and metrics["model_esr"] < 0.9 * metrics["baseline_esr"]
        and metrics["model_mae"] < 0.9 * metrics["baseline_mae"]
        and metrics["model_spectral"] < metrics["baseline_spectral"]
        and worst_bin_ratio <= 1.05
        and metrics["peak_ratio_p95"] <= 1.25
        and metrics["worst_peak_ratio"] <= 1.75
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--checkpoint", type=Path, default=RUN / "drive-forward-candidate.pt")
    parser.add_argument("--output", type=Path, default=RUN / "extended-validation.json")
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--frames", type=int, default=FORWARD_RATE)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-frames", type=int, default=2_048)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = DriveForwardRenderer(hidden_size=int(payload["hidden_size"]))
    model.load_state_dict(payload["state_dict"])
    target_device = device()
    model.to(target_device)
    validation = {}
    for domain in VALIDATION_DOMAINS:
        dataset = DriveForwardDataset(
            dry_sources(args.corpus, "valid"),
            args.samples,
            args.frames,
            seed=20260905,
            domains=(domain,),
        )
        validation[domain] = detailed_evaluate(
            model,
            DataLoader(dataset, batch_size=args.batch_size, num_workers=0),
            target_device,
            args.chunk_frames,
        )
        print(json.dumps({"domain": domain, **validation[domain]}, sort_keys=True))
    passed = all(metrics["passed"] for metrics in validation.values())
    report = {
        "schema": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checkpoint": str(args.checkpoint),
        "sample_rate": FORWARD_RATE,
        "frames": args.frames,
        "chunk_frames": args.chunk_frames,
        "samples_per_domain": args.samples,
        "source_split": "valid",
        "fresh_control_seed": 20260905,
        "locked_tele_test_opened": False,
        "challenge_renderer_opened": False,
        "physical_audio_devices_used": False,
        "validation": validation,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": passed}, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
