#!/usr/bin/env python3
"""Audit Drive forward-model generalization outside its training input levels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import dry_sources
from .evaluate_forward_drive import detailed_evaluate
from .forward_drive import FORWARD_RATE, DriveForwardDataset, DriveForwardRenderer
from .train import CORPUS, device
from .train_forward_drive import VALIDATION_DOMAINS


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/drive-forward-pilot"
AMPLITUDE_BANDS = {
    "very_quiet": (0.01, 0.04),
    "quiet": (0.04, 0.08),
    "hot": (0.24, 0.50),
    "very_hot": (0.50, 0.90),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--checkpoint", type=Path, default=RUN / "drive-forward-candidate.pt")
    parser.add_argument("--output", type=Path, default=RUN / "amplitude-validation.json")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--frames", type=int, default=FORWARD_RATE // 2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-frames", type=int, default=2_048)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = DriveForwardRenderer(hidden_size=int(payload["hidden_size"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    target_device = device()
    model.to(target_device)
    sources = dry_sources(args.corpus, "valid")
    validation = {}
    for band_index, (band, peak_range) in enumerate(AMPLITUDE_BANDS.items()):
        validation[band] = {}
        for domain_index, domain in enumerate(VALIDATION_DOMAINS):
            dataset = DriveForwardDataset(
                sources,
                args.samples,
                args.frames,
                seed=20260920 + band_index * 101 + domain_index,
                domains=(domain,),
                input_peak_range=peak_range,
            )
            metrics = detailed_evaluate(
                model,
                DataLoader(dataset, batch_size=args.batch_size, num_workers=0),
                target_device,
                args.chunk_frames,
            )
            validation[band][domain] = metrics
            print(
                json.dumps(
                    {
                        "band": band,
                        "input_peak_range": peak_range,
                        "domain": domain,
                        "passed": metrics["passed"],
                        "esr_improvement": metrics["esr_improvement"],
                        "mae_improvement": metrics["mae_improvement"],
                        "peak_ratio_p95": metrics["peak_ratio_p95"],
                        "worst_peak_ratio": metrics["worst_peak_ratio"],
                    },
                    sort_keys=True,
                )
            )
    passed = all(
        metrics["passed"]
        for domains in validation.values()
        for metrics in domains.values()
    )
    report = {
        "schema": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checkpoint": str(args.checkpoint),
        "sample_rate": FORWARD_RATE,
        "frames": args.frames,
        "chunk_frames": args.chunk_frames,
        "samples_per_band_domain": args.samples,
        "source_split": "valid",
        "training_input_peak_range": [0.08, 0.24],
        "audited_input_peak_ranges": {
            name: list(values) for name, values in AMPLITUDE_BANDS.items()
        },
        "runtime_automatic_normalization": False,
        "physical_audio_devices_used": False,
        "validation": validation,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": passed}, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
