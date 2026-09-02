#!/usr/bin/env python3
"""One-way challenge audit for the frozen hybrid Delay candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import dry_sources
from .evaluate_forward_delay import detailed_evaluate, invariant_audit
from .evaluate_forward_drive_amplitude import AMPLITUDE_BANDS
from .forward_delay import DelayForwardDataset, DelayForwardRenderer
from .train import CORPUS, device


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/delay-forward-pilot"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit(
    model: DelayForwardRenderer,
    sources: list[Path],
    target_device: torch.device,
    *,
    samples: int,
    frames: int,
    seed: int,
    peak_range: tuple[float, float],
    batch_size: int,
) -> dict:
    return detailed_evaluate(
        model,
        DataLoader(
            DelayForwardDataset(
                sources,
                samples,
                frames,
                seed=seed,
                domains=("challenge",),
                input_peak_ranges=(peak_range,),
            ),
            batch_size=batch_size,
            num_workers=0,
        ),
        target_device,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--checkpoint", type=Path, default=RUN / "delay-forward-candidate.pt")
    parser.add_argument("--output", type=Path, default=RUN / "challenge-validation.json")
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--amplitude-samples", type=int, default=48)
    parser.add_argument("--frames", type=int, default=120_000)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    frozen_hash = _sha256(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = DelayForwardRenderer(fir_taps=int(payload["fir_taps"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    target_device = device()
    model.to(target_device)
    sources = dry_sources(args.corpus, "valid")
    nominal = _audit(
        model,
        sources,
        target_device,
        samples=args.samples,
        frames=args.frames,
        seed=20261030,
        peak_range=(0.08, 0.24),
        batch_size=args.batch_size,
    )
    print(json.dumps({"challenge": "nominal", **nominal}, sort_keys=True))
    amplitude = {}
    for index, (name, peak_range) in enumerate(AMPLITUDE_BANDS.items()):
        amplitude[name] = _audit(
            model,
            sources,
            target_device,
            samples=args.amplitude_samples,
            frames=args.frames,
            seed=20261031 + index,
            peak_range=peak_range,
            batch_size=args.batch_size,
        )
        print(
            json.dumps(
                {
                    "challenge": name,
                    "passed": amplitude[name]["passed"],
                    "esr_improvement": amplitude[name]["esr_improvement"],
                    "mae_improvement": amplitude[name]["mae_improvement"],
                    "peak_ratio_p95": amplitude[name]["peak_ratio_p95"],
                    "worst_peak_ratio": amplitude[name]["worst_peak_ratio"],
                },
                sort_keys=True,
            )
        )
    invariants = invariant_audit(model.cpu())
    passed = bool(
        nominal["passed"]
        and all(values["passed"] for values in amplitude.values())
        and invariants["worst_timing_error_samples"] == 0
        and invariants["bypass_max_absolute_error"] == 0.0
        and invariants["silence_max_absolute_output"] == 0.0
    )
    after_hash = _sha256(args.checkpoint)
    report = {
        "schema": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256_before_challenge": frozen_hash,
        "checkpoint_sha256_after_challenge": after_hash,
        "checkpoint_immutable_during_audit": frozen_hash == after_hash,
        "renderer": "challenge",
        "source_split": "valid",
        "nominal": nominal,
        "amplitude": amplitude,
        "invariants": invariants,
        "challenge_renderer_opened": True,
        "challenge_result_used_for_training": False,
        "locked_tele_test_opened": False,
        "physical_audio_devices_used": False,
        "runtime_automatic_normalization": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": passed}, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
