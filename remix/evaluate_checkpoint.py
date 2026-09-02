#!/usr/bin/env python3
"""Evaluate an existing paired checkpoint without retraining or touching test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import DAFxOrderDataset, SyntheticControlDataset, dry_sources
from .delay_model import DelayControlEstimator
from .drive_model import DriveControlEstimator
from .model import PairedEstimator
from .quality import contract_manifest
from .reverb_model import ReverbControlEstimator
from .train import CORPUS, RUN, device, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--drive-checkpoint",
        type=Path,
        default=RUN / "drive-estimator.pt",
    )
    parser.add_argument(
        "--delay-checkpoint",
        type=Path,
        default=RUN / "delay-estimator.pt",
    )
    parser.add_argument(
        "--reverb-checkpoint",
        type=Path,
        default=RUN / "reverb-estimator.pt",
    )
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=RUN / "evaluated-metrics.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--synthetic-samples", type=int, default=1_600)
    parser.add_argument(
        "--stress-samples",
        type=int,
        default=0,
        help="evaluate the third augmentation renderer",
    )
    parser.add_argument(
        "--challenge-samples",
        type=int,
        default=0,
        help="evaluate the final renderer excluded from training and calibration",
    )
    parser.add_argument(
        "--pedalboard-samples",
        type=int,
        default=0,
        help="evaluate Spotify Pedalboard as an external DSP holdout",
    )
    args = parser.parse_args()

    target = device()
    model = PairedEstimator().to(target)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=target, weights_only=True)
    )
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
    paths = dry_sources(args.corpus, "valid")
    real = DAFxOrderDataset(args.corpus, "valid")
    reference = SyntheticControlDataset(
        paths,
        args.synthetic_samples,
        seed=20260831,
        renderers=("reference",),
    )
    alternate = SyntheticControlDataset(
        paths,
        args.synthetic_samples,
        seed=20260831,
        renderers=("alternate",),
    )
    stress = (
        SyntheticControlDataset(
            paths,
            args.stress_samples,
            seed=20260902,
            renderers=("stress",),
        )
        if args.stress_samples
        else None
    )
    challenge = (
        SyntheticControlDataset(
            paths,
            args.challenge_samples,
            seed=20260903,
            renderers=("challenge",),
        )
        if args.challenge_samples
        else None
    )
    pedalboard = None
    if args.pedalboard_samples:
        from .pedalboard_data import PedalboardControlDataset

        pedalboard = PedalboardControlDataset(paths, args.pedalboard_samples)
    validation = {
        "real_order": evaluate(
            model,
            DataLoader(real, batch_size=args.batch_size),
            target,
            drive_model,
            delay_model,
            reverb_model,
        ),
        "synthetic_reference": evaluate(
            model,
            DataLoader(reference, batch_size=args.batch_size),
            target,
            drive_model,
            delay_model,
            reverb_model,
        ),
        "synthetic_alternate": evaluate(
            model,
            DataLoader(alternate, batch_size=args.batch_size),
            target,
            drive_model,
            delay_model,
            reverb_model,
        ),
    }
    if stress is not None:
        validation["synthetic_stress_unseen"] = evaluate(
            model,
            DataLoader(stress, batch_size=args.batch_size),
            target,
            drive_model,
            delay_model,
            reverb_model,
        )
    if challenge is not None:
        validation["synthetic_challenge_unseen"] = evaluate(
            model,
            DataLoader(challenge, batch_size=args.batch_size),
            target,
            drive_model,
            delay_model,
            reverb_model,
        )
    if pedalboard is not None:
        validation["external_pedalboard_unseen"] = evaluate(
            model,
            DataLoader(pedalboard, batch_size=args.batch_size),
            target,
            drive_model,
            delay_model,
            reverb_model,
        )
    report = {
        "schema": 1,
        "evaluation_only": True,
        "audio_quality": contract_manifest(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "drive_checkpoint": str(args.drive_checkpoint),
        "drive_checkpoint_sha256": hashlib.sha256(
            args.drive_checkpoint.read_bytes()
        ).hexdigest(),
        "delay_checkpoint": str(args.delay_checkpoint),
        "delay_checkpoint_sha256": hashlib.sha256(
            args.delay_checkpoint.read_bytes()
        ).hexdigest(),
        "reverb_checkpoint": str(args.reverb_checkpoint),
        "reverb_checkpoint_sha256": hashlib.sha256(
            args.reverb_checkpoint.read_bytes()
        ).hexdigest(),
        "parameter_counts": {
            "main": sum(parameter.numel() for parameter in model.parameters()),
            "drive": sum(parameter.numel() for parameter in drive_model.parameters()),
            "delay": sum(parameter.numel() for parameter in delay_model.parameters()),
            "reverb": sum(parameter.numel() for parameter in reverb_model.parameters()),
            "total": sum(parameter.numel() for parameter in model.parameters())
            + sum(parameter.numel() for parameter in drive_model.parameters())
            + sum(parameter.numel() for parameter in delay_model.parameters())
            + sum(parameter.numel() for parameter in reverb_model.parameters()),
        },
        "validation_examples": len(real)
        + len(reference)
        + len(alternate)
        + (len(stress) if stress is not None else 0)
        + (len(challenge) if challenge is not None else 0)
        + (len(pedalboard) if pedalboard is not None else 0),
        "validation": validation,
    }
    if pedalboard is not None:
        from .pedalboard_renderer import pedalboard_manifest

        report["external_domain"] = pedalboard_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
