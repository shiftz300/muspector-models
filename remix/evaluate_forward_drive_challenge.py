#!/usr/bin/env python3
"""One-way final synthetic-domain audit for a frozen Drive candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import dry_sources
from .evaluate_forward_drive import detailed_evaluate
from .evaluate_forward_drive_amplitude import AMPLITUDE_BANDS
from .forward_drive import FORWARD_RATE, DriveForwardDataset, DriveForwardRenderer
from .train import CORPUS, device


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/drive-forward-pilot"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluate(
    model: DriveForwardRenderer,
    sources: list[Path],
    target_device: torch.device,
    *,
    samples: int,
    frames: int,
    seed: int,
    batch_size: int,
    chunk_frames: int,
    peak_range: tuple[float, float] = (0.08, 0.24),
) -> dict:
    dataset = DriveForwardDataset(
        sources,
        samples,
        frames,
        seed=seed,
        domains=("challenge",),
        input_peak_range=peak_range,
    )
    return detailed_evaluate(
        model,
        DataLoader(dataset, batch_size=batch_size, num_workers=0),
        target_device,
        chunk_frames,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--checkpoint", type=Path, default=RUN / "drive-forward-candidate.pt")
    parser.add_argument("--output", type=Path, default=RUN / "challenge-validation.json")
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--frames", type=int, default=FORWARD_RATE)
    parser.add_argument("--amplitude-samples", type=int, default=64)
    parser.add_argument("--amplitude-frames", type=int, default=FORWARD_RATE // 2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-frames", type=int, default=2_048)
    args = parser.parse_args()

    frozen_hash = _sha256(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = DriveForwardRenderer(hidden_size=int(payload["hidden_size"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    target_device = device()
    model.to(target_device)
    sources = dry_sources(args.corpus, "valid")

    nominal = _evaluate(
        model,
        sources,
        target_device,
        samples=args.samples,
        frames=args.frames,
        seed=20261001,
        batch_size=args.batch_size,
        chunk_frames=args.chunk_frames,
    )
    print(json.dumps({"challenge": "nominal", **nominal}, sort_keys=True))
    amplitude = {}
    for index, (name, peak_range) in enumerate(AMPLITUDE_BANDS.items()):
        amplitude[name] = _evaluate(
            model,
            sources,
            target_device,
            samples=args.amplitude_samples,
            frames=args.amplitude_frames,
            seed=20261011 + index,
            batch_size=args.batch_size,
            chunk_frames=args.chunk_frames,
            peak_range=peak_range,
        )
        print(
            json.dumps(
                {
                    "challenge": name,
                    "input_peak_range": peak_range,
                    "passed": amplitude[name]["passed"],
                    "esr_improvement": amplitude[name]["esr_improvement"],
                    "mae_improvement": amplitude[name]["mae_improvement"],
                    "peak_ratio_p95": amplitude[name]["peak_ratio_p95"],
                    "worst_peak_ratio": amplitude[name]["worst_peak_ratio"],
                },
                sort_keys=True,
            )
        )
    passed = nominal["passed"] and all(values["passed"] for values in amplitude.values())
    report = {
        "schema": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256_before_challenge": frozen_hash,
        "checkpoint_sha256_after_challenge": _sha256(args.checkpoint),
        "checkpoint_immutable_during_audit": frozen_hash == _sha256(args.checkpoint),
        "renderer": "challenge",
        "source_split": "valid",
        "locked_tele_test_opened": False,
        "challenge_renderer_opened": True,
        "challenge_result_used_for_training": False,
        "physical_audio_devices_used": False,
        "runtime_automatic_normalization": False,
        "nominal": nominal,
        "amplitude": amplitude,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": passed}, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
