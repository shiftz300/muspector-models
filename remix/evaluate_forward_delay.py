#!/usr/bin/env python3
"""Long-horizon audit for the hybrid Delay forward renderer."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import dry_sources
from .forward_delay import DelayForwardDataset, DelayForwardRenderer
from .forward_drive import FORWARD_RATE, multiresolution_spectral_loss
from .train import CORPUS, device
from .train_forward_delay import VALIDATION_DOMAINS


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/delay-forward-pilot"


@torch.no_grad()
def detailed_evaluate(
    model: DelayForwardRenderer,
    loader: DataLoader,
    target_device: torch.device,
) -> dict:
    model.eval()
    totals = defaultdict(float)
    bins: dict[str, dict[int, dict[str, float]]] = {
        name: {index: defaultdict(float) for index in range(4)}
        for name in ("time", "feedback", "mix")
    }
    ratios = []
    examples = 0
    finite = True
    for batch in loader:
        dry = batch["dry"].to(target_device)
        wet = batch["wet"].to(target_device)
        controls = batch["controls"].to(target_device)
        prediction = model(dry, controls)
        count = dry.shape[0]
        examples += count
        model_error = (prediction - wet).square().mean(dim=1)
        baseline_error = (dry - wet).square().mean(dim=1)
        target_energy = wet.square().mean(dim=1).clamp_min(1.0e-8)
        model_esr = model_error / target_energy
        baseline_esr = baseline_error / target_energy
        model_mae = torch.abs(prediction - wet).mean(dim=1)
        baseline_mae = torch.abs(dry - wet).mean(dim=1)
        totals["model_esr"] += float(model_esr.sum())
        totals["baseline_esr"] += float(baseline_esr.sum())
        totals["model_mae"] += float(model_mae.sum())
        totals["baseline_mae"] += float(baseline_mae.sum())
        totals["model_spectral"] += float(multiresolution_spectral_loss(prediction, wet)) * count
        totals["baseline_spectral"] += float(multiresolution_spectral_loss(dry, wet)) * count
        ratios.extend(
            (
                prediction.abs().amax(dim=1)
                / wet.abs().amax(dim=1).clamp_min(1.0e-6)
            ).cpu().tolist()
        )
        finite = finite and bool(torch.isfinite(prediction).all())
        for dimension, name in enumerate(("time", "feedback", "mix")):
            indexes = torch.clamp((controls[:, dimension] * 4.0).long(), max=3)
            for bin_index in range(4):
                selected = indexes == bin_index
                selected_count = int(selected.sum())
                if not selected_count:
                    continue
                bucket = bins[name][bin_index]
                bucket["examples"] += selected_count
                bucket["model_esr"] += float(model_esr[selected].sum())
                bucket["baseline_esr"] += float(baseline_esr[selected].sum())
                bucket["model_mae"] += float(model_mae[selected].sum())
                bucket["baseline_mae"] += float(baseline_mae[selected].sum())
    metrics = {name: value / examples for name, value in totals.items()}
    metrics["examples"] = examples
    metrics["finite"] = finite
    ordered = sorted(float(value) for value in ratios)
    index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    metrics["peak_ratio_p95"] = ordered[index]
    metrics["worst_peak_ratio"] = ordered[-1]
    metrics["esr_improvement"] = 1.0 - metrics["model_esr"] / max(
        metrics["baseline_esr"], 1.0e-12
    )
    metrics["mae_improvement"] = 1.0 - metrics["model_mae"] / max(
        metrics["baseline_mae"], 1.0e-12
    )
    metrics["spectral_improvement"] = 1.0 - metrics["model_spectral"] / max(
        metrics["baseline_spectral"], 1.0e-12
    )
    reports = {}
    worst_bin_ratio = 0.0
    for name, values in bins.items():
        reports[name] = {}
        for bin_index, bucket in values.items():
            count = max(int(bucket["examples"]), 1)
            report = {
                key: value / count for key, value in bucket.items() if key != "examples"
            }
            report["examples"] = int(bucket["examples"])
            report["esr_ratio"] = report["model_esr"] / max(report["baseline_esr"], 1.0e-12)
            report["mae_ratio"] = report["model_mae"] / max(report["baseline_mae"], 1.0e-12)
            worst_bin_ratio = max(worst_bin_ratio, report["esr_ratio"], report["mae_ratio"])
            reports[name][str(bin_index)] = report
    metrics["control_quartiles"] = reports
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


@torch.no_grad()
def invariant_audit(model: DelayForwardRenderer) -> dict:
    frames = 50_000
    dry = torch.zeros((1, frames), dtype=torch.float32)
    dry[:, 0] = 1.0
    timing = {}
    for time_ms in (40.0, 73.0, 240.0, 999.0):
        normalized_time = math.log(time_ms / 40.0) / math.log(25.0)
        controls = torch.tensor(((normalized_time, 0.5, 1.0),), dtype=torch.float32)
        rendered = model(dry, controls)
        expected = round(time_ms * FORWARD_RATE / 1_000.0)
        tail = rendered[0, 1:]
        observed = int(torch.nonzero(torch.abs(tail) > 1.0e-8)[0]) + 1
        timing[str(time_ms)] = {
            "expected_sample": expected,
            "observed_sample": observed,
            "absolute_error_samples": abs(observed - expected),
        }
    random_dry = torch.randn((2, 8_192), dtype=torch.float32) * 0.1
    bypass = model(random_dry, torch.tensor(((0.0, 0.0, 0.0), (1.0, 1.0, 0.0))))
    silence = model(torch.zeros_like(random_dry), torch.tensor(((0.0, 0.0, 1.0), (1.0, 1.0, 1.0))))
    return {
        "timing": timing,
        "worst_timing_error_samples": max(
            values["absolute_error_samples"] for values in timing.values()
        ),
        "bypass_max_absolute_error": float(torch.max(torch.abs(bypass - random_dry))),
        "silence_max_absolute_output": float(torch.max(torch.abs(silence))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--checkpoint", type=Path, default=RUN / "delay-forward-candidate.pt")
    parser.add_argument("--output", type=Path, default=RUN / "extended-validation.json")
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--frames", type=int, default=120_000)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = DelayForwardRenderer(fir_taps=int(payload["fir_taps"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    target_device = device()
    model.to(target_device)
    validation = {}
    for domain in VALIDATION_DOMAINS:
        loader = DataLoader(
            DelayForwardDataset(
                dry_sources(args.corpus, "valid"),
                args.samples,
                args.frames,
                seed=20261025,
                domains=(domain,),
            ),
            batch_size=args.batch_size,
            num_workers=0,
        )
        validation[domain] = detailed_evaluate(model, loader, target_device)
        print(json.dumps({"domain": domain, **validation[domain]}, sort_keys=True))
    invariants = invariant_audit(model.cpu())
    passed = bool(
        all(values["passed"] for values in validation.values())
        and invariants["worst_timing_error_samples"] == 0
        and invariants["bypass_max_absolute_error"] == 0.0
        and invariants["silence_max_absolute_output"] == 0.0
    )
    report = {
        "schema": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checkpoint": str(args.checkpoint),
        "sample_rate": FORWARD_RATE,
        "frames": args.frames,
        "samples_per_domain": args.samples,
        "source_split": "valid",
        "fresh_control_seed": 20261025,
        "invariants": invariants,
        "validation": validation,
        "physical_audio_devices_used": False,
        "runtime_automatic_normalization": False,
        "challenge_renderer_opened": False,
        "locked_tele_test_opened": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": passed}, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
