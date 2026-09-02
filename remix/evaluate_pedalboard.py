#!/usr/bin/env python3
"""Evaluate the frozen four-model bundle on Spotify Pedalboard only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import dry_sources
from .delay_model import DelayControlEstimator
from .drive_model import DriveControlEstimator
from .model import PairedEstimator
from .pedalboard_data import PedalboardControlDataset
from .pedalboard_renderer import pedalboard_manifest
from .reverb_model import ReverbControlEstimator
from .train import CORPUS, RUN, device, evaluate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=RUN / "paired-estimator.pt")
    parser.add_argument("--drive-checkpoint", type=Path, default=RUN / "drive-estimator.pt")
    parser.add_argument("--delay-checkpoint", type=Path, default=RUN / "delay-estimator.pt")
    parser.add_argument("--reverb-checkpoint", type=Path, default=RUN / "reverb-estimator.pt")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=RUN / "pedalboard-unseen-baseline.json")
    parser.add_argument("--samples", type=int, default=1_600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--order-equivalence-db",
        type=float,
        default=None,
        help="mask order relations reversed by a quieter counterfactual permutation",
    )
    args = parser.parse_args()

    target = device()
    model = PairedEstimator().to(target)
    model.load_state_dict(torch.load(args.checkpoint, map_location=target, weights_only=True))
    drive_model = DriveControlEstimator().to(target)
    drive_model.load_state_dict(
        torch.load(args.drive_checkpoint, map_location=target, weights_only=True)
    )
    delay_model = DelayControlEstimator().to(target)
    delay_model.load_state_dict(
        torch.load(args.delay_checkpoint, map_location=target, weights_only=True)
    )
    reverb_model = ReverbControlEstimator().to(target)
    reverb_model.load_state_dict(
        torch.load(args.reverb_checkpoint, map_location=target, weights_only=True)
    )

    sources = dry_sources(args.corpus, "valid")
    dataset = PedalboardControlDataset(
        sources,
        args.samples,
        seed=args.seed,
        order_equivalence_db=args.order_equivalence_db,
    )
    metrics = evaluate(
        model,
        DataLoader(dataset, batch_size=args.batch_size),
        target,
        drive_model,
        delay_model,
        reverb_model,
    )
    checkpoints = {
        "main": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint)},
        "drive": {
            "path": str(args.drive_checkpoint),
            "sha256": sha256(args.drive_checkpoint),
        },
        "delay": {
            "path": str(args.delay_checkpoint),
            "sha256": sha256(args.delay_checkpoint),
        },
        "reverb": {
            "path": str(args.reverb_checkpoint),
            "sha256": sha256(args.reverb_checkpoint),
        },
    }
    report = {
        "schema": 1,
        "evaluation_only": True,
        "trained_on_domain": False,
        "samples": len(dataset),
        "seed": args.seed,
        "order_equivalence_db": args.order_equivalence_db,
        "dry_split": "Stratocaster takes 1-12 validation sources",
        "dry_sources": [path.name for path in sources],
        "checkpoints": checkpoints,
        "domain": pedalboard_manifest(),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
