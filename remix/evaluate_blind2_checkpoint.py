"""Recompute Blind v2 development metrics for an existing frozen checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .blind2 import BlindPresence
from .blind_data2 import BlindPresenceData, inventory
from .train_blind2 import (
    SEED,
    _acceptance,
    _collect,
    _loader,
    _metrics,
    _probabilities,
    _thresholds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--recalibrate-thresholds", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    target = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["manifest"]["architecture"] != "compact-audio-resnet18":
        raise ValueError("only compact Blind checkpoints are supported")
    model = BlindPresence().to(target)
    model.load_state_dict(checkpoint["model"])
    thresholds = checkpoint["thresholds"]
    calibration_metrics = None
    if args.recalibrate_thresholds:
        calibration_data = BlindPresenceData(Path.cwd(), "calibration", 480, SEED + 23)
        calibration_logits, calibration_expected, calibration_sources = _collect(
            model, _loader(calibration_data, 16, False), target
        )
        calibration_probabilities = _probabilities(
            calibration_logits, checkpoint["calibration"]
        )
        thresholds = _thresholds(calibration_probabilities, calibration_expected)
        calibration_metrics = _metrics(
            calibration_probabilities,
            calibration_expected,
            thresholds,
            calibration_sources,
        )
    data = BlindPresenceData(Path.cwd(), "development", 640, SEED + 37)
    logits, expected, sources = _collect(model, _loader(data, 16, False), target)
    probabilities = _probabilities(logits, checkpoint["calibration"])
    metrics = _metrics(probabilities, expected, thresholds, sources)
    report = {
        "schema": 1,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        },
        "device": str(target),
        "development": metrics,
        "calibration": calibration_metrics,
        "thresholds": dict(zip(model.labels, thresholds, strict=True)),
        "acceptance": _acceptance(metrics),
        "inventory": inventory(Path.cwd()),
        "boundaries": {
            "development_only": True,
            "locked_final_opened": False,
            "order_labels_used": False,
            "controls_used": False,
            "weights_changed": False,
            "thresholds_recalibrated": args.recalibrate_thresholds,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
