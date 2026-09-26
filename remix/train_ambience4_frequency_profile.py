#!/usr/bin/env python3
"""Train the Product4 frequency-dependent known-profile Reverb selector on MPS."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .ambience4 import (
    AmbiencePairsV4FrequencyProfileBank,
    AmbiencePairsV4MixedFrequencyProfileBank,
    AmbiencePairsV4OpenSLR26FrequencyProfileBank,
)
from .ambience_model4 import AmbienceFrequencyProfileBankExpert
from .train_ambience2 import train


SEED = 20260908


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/foundation/product4-reverb-frequency-profile-v1"),
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--rir-source", choices=("but", "openslr26", "mixed"), default="but"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=192)
    parser.add_argument("--calibration-samples", type=int, default=96)
    parser.add_argument("--development-samples", type=int, default=96)
    parser.add_argument("--target-frames", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--uncertainty-weight", type=float, default=0.0)
    parser.add_argument("--tail-weight", type=float, default=3.0)
    parser.add_argument("--gate-oracle-weight", type=float, default=5.0)
    args = parser.parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.train_samples = min(args.train_samples, 16)
        args.calibration_samples = min(args.calibration_samples, 12)
        args.development_samples = min(args.development_samples, 12)
        args.channels = min(args.channels, 4)
        args.depth = min(args.depth, 6)
    args.seed = SEED
    args.select_by_calibration_quality = True
    args.quality_selection_interval = 2
    if args.output.exists():
        raise FileExistsError(f"refusing to replace Product4 frequency profile run: {args.output}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))
    dataset_class = {
        "but": AmbiencePairsV4FrequencyProfileBank,
        "openslr26": AmbiencePairsV4OpenSLR26FrequencyProfileBank,
        "mixed": AmbiencePairsV4MixedFrequencyProfileBank,
    }[args.rir_source]
    report = train(
        args,
        dataset_class=dataset_class,
        model_class=AmbienceFrequencyProfileBankExpert,
    )
    development_gate_passed = bool(report["accepted"])
    report["development_gate_passed"] = development_gate_passed
    report["accepted"] = False
    report["status"] = (
        "development-gates-passed-not-promoted"
        if development_gate_passed
        else "diagnostic-not-promoted"
    )
    report["promotion_blockers"] = [
        "fresh-v2 and locked-final remain unopened",
        "streaming profile-candidate generation is not sealed",
        "listening acceptance is not complete",
    ]
    report["rir_training_source"] = args.rir_source
    metrics = args.output.resolve() / "ambience/metrics.json"
    metrics.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "accelerator": report["training"]["accelerator"],
        "selected_epoch": report["training"]["selected_epoch"],
        "selected_calibration_loss": report["training"]["selected_calibration_loss"],
        "development_gate_passed": development_gate_passed,
        "development_tail": report["development"]["tail"],
        "metrics": str(metrics),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
