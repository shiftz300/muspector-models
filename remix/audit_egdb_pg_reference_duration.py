#!/usr/bin/env python3
"""Audit Wet-only tone observation duration on revealed EGDB-PG development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .egdb_pg_amp_data import EgdbPgAmpPairs, RATE
from .egdb_pg_amp_model10 import WetTonePhaseGrayBoxAmpCabInverse
from .train_egdb_pg_amp import SEED, _quality


def audit(args: argparse.Namespace) -> dict:
    run = args.run.resolve()
    checkpoint = run / "model.pt"
    training = json.loads((run / "metrics.json").read_text())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != training["model"]["sha256"]:
        raise PermissionError("checkpoint hash differs from frozen training report")
    if training["model"].get("schema") != 12:
        raise ValueError("reference-duration audit requires phase gray-box schema 12")
    if training["quality"].get("locked_final_audio_opened") is not False:
        raise PermissionError("training report did not preserve locked-final")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    architecture = payload["architecture"]
    model = WetTonePhaseGrayBoxAmpCabInverse(
        architecture["channels"], architecture["depth"], architecture["condition_size"]
    )
    model.load_state_dict(payload["state_dict"])
    model.tone_encoder_sha256 = architecture["tone_encoder_sha256"]
    model.eval()

    durations = {}
    for seconds in args.seconds:
        reference_frames = round(seconds * RATE)
        dataset = EgdbPgAmpPairs(
            args.workspace.resolve(),
            "development",
            args.samples,
            args.target_frames,
            args.context_frames,
            SEED + 3,
            reference_frames,
        )
        durations[f"{seconds:g}"] = _quality(model, dataset)
    report = {
        "schema": 1,
        "status": "revealed-development-diagnostic",
        "checkpoint_sha256": digest,
        "split": "development",
        "fresh_validation_opened": False,
        "fresh_validation_v2_opened": False,
        "locked_final_opened": False,
        "tone_reference_source": "same Wet recording only",
        "durations_seconds": durations,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace duration audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        seconds: {
            "pass_fraction": quality["pass_fraction"],
            "metrics": quality["metrics"],
        }
        for seconds, quality in durations.items()
    }, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, nargs="+", default=(3.0, 5.5, 8.0))
    parser.add_argument("--samples", type=int, default=72)
    parser.add_argument("--target-frames", type=int, default=16384)
    parser.add_argument("--context-frames", type=int, default=4096)
    args = parser.parse_args()
    if any(seconds < 3.0 or seconds > 12.0 for seconds in args.seconds):
        raise ValueError("tone observation duration must be between 3 and 12 seconds")
    audit(args)


if __name__ == "__main__":
    main()
