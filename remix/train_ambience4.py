#!/usr/bin/env python3
"""Run a non-promotable MPS screen of Ambience v2 on room-disjoint BUT RIRs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .ambience4 import (
    AmbiencePairsV4,
    AmbiencePairsV4DecayBank,
    AmbiencePairsV4ProfileBank,
    AmbiencePairsV4WetBase,
)
from .ambience_model2 import AmbienceExpert
from .ambience_model4 import (
    AmbienceDecayBankExpert,
    AmbienceGrayboxExpert,
    AmbienceProfileBankExpert,
    AmbienceSparseMaskExpert,
)
from .train_ambience2 import train


SEED = 20260904


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product4-reverb-screen-v1"),
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=24)
    parser.add_argument("--calibration-samples", type=int, default=16)
    parser.add_argument("--development-samples", type=int, default=24)
    parser.add_argument("--target-frames", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument(
        "--uncertainty-weight", type=float, default=0.0,
        help="fit restoration first; uncertainty is calibrated only after viability",
    )
    parser.add_argument("--tail-weight", type=float, default=3.0)
    parser.add_argument("--gate-oracle-weight", type=float, default=5.0)
    parser.add_argument(
        "--base", choices=("wet", "blind-late"), default="wet",
        help="artifact-safe Wet residual base is the Product4 default",
    )
    parser.add_argument(
        "--architecture",
        choices=("profile-bank", "decay-bank", "graybox", "sparse-mask", "residual"),
        default="decay-bank",
    )
    args = parser.parse_args()
    args.quick = True
    args.seed = SEED
    args.select_by_calibration_quality = True
    args.quality_selection_interval = 2
    if args.output.exists():
        raise FileExistsError(f"refusing to replace Product4 Reverb screen: {args.output}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))
    if args.architecture in {"profile-bank", "decay-bank", "graybox", "sparse-mask"}:
        if args.architecture == "profile-bank":
            dataset_class = AmbiencePairsV4ProfileBank
            model_class = AmbienceProfileBankExpert
            residual_base = "known-profile-bounded-response-shortening-bank"
        elif args.architecture == "decay-bank":
            dataset_class = AmbiencePairsV4DecayBank
            model_class = AmbienceDecayBankExpert
            residual_base = "bounded-multiband-exponential-decay-bank"
        else:
            dataset_class = AmbiencePairsV4
            model_class = (
                AmbienceSparseMaskExpert
                if args.architecture == "sparse-mask"
                else AmbienceGrayboxExpert
            )
            residual_base = "bounded-blind-late-mask"
    else:
        dataset_class = AmbiencePairsV4WetBase if args.base == "wet" else AmbiencePairsV4
        model_class = AmbienceExpert
        residual_base = args.base
    report = train(args, dataset_class=dataset_class, model_class=model_class)
    report["status"] = "diagnostic-room-disjoint-screen-not-promotable"
    report["accepted"] = False
    report["screen"] = {
        "architecture": args.architecture,
        "residual_base": residual_base,
        "room_balanced_sampling": True,
        "product4_decay_strata_seconds": [0.40, 0.75],
    }
    metrics = args.output.resolve() / "ambience/metrics.json"
    metrics.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "accelerator": report["training"]["accelerator"],
        "selected_calibration_loss": report["training"]["selected_calibration_loss"],
        "metrics": str(metrics),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
