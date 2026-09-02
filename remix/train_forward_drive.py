#!/usr/bin/env python3
"""Train and audit the control-conditioned Drive forward renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import dry_sources
from .forward_drive import (
    FORWARD_RATE,
    DriveForwardDataset,
    DriveForwardRenderer,
    error_to_signal_ratio,
    forward_loss,
    multiresolution_spectral_loss,
    pre_emphasis,
)
from .train import CORPUS, device


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/drive-forward-pilot"
TRAIN_DOMAINS = ("reference", "alternate", "stress")
VALIDATION_DOMAINS = (*TRAIN_DOMAINS, "pedalboard")
WIDE_INPUT_PEAK_RANGES = (
    (0.01, 0.04),
    (0.04, 0.08),
    (0.08, 0.24),
    (0.24, 0.50),
    (0.50, 0.90),
)


def _tensor_batch(batch: dict, target: torch.device) -> tuple[torch.Tensor, ...]:
    return batch["dry"].to(target), batch["wet"].to(target), batch["controls"].to(target)


@torch.no_grad()
def evaluate(
    model: DriveForwardRenderer,
    loader: DataLoader,
    target: torch.device,
) -> dict:
    model.eval()
    totals = {
        "examples": 0,
        "model_esr": 0.0,
        "model_preemphasis_esr": 0.0,
        "model_spectral": 0.0,
        "baseline_esr": 0.0,
        "baseline_preemphasis_esr": 0.0,
        "baseline_spectral": 0.0,
        "model_mae": 0.0,
        "baseline_mae": 0.0,
        "finite": True,
    }
    peak_ratios = []
    for batch in loader:
        dry, wet, controls = _tensor_batch(batch, target)
        prediction, _ = model(dry, controls)
        examples = dry.shape[0]
        totals["examples"] += examples
        totals["model_esr"] += float(error_to_signal_ratio(prediction, wet)) * examples
        totals["model_preemphasis_esr"] += float(
            error_to_signal_ratio(pre_emphasis(prediction), pre_emphasis(wet))
        ) * examples
        totals["model_spectral"] += float(
            multiresolution_spectral_loss(prediction, wet)
        ) * examples
        totals["baseline_esr"] += float(error_to_signal_ratio(dry, wet)) * examples
        totals["baseline_preemphasis_esr"] += float(
            error_to_signal_ratio(pre_emphasis(dry), pre_emphasis(wet))
        ) * examples
        totals["baseline_spectral"] += float(
            multiresolution_spectral_loss(dry, wet)
        ) * examples
        totals["model_mae"] += float(torch.mean(torch.abs(prediction - wet))) * examples
        totals["baseline_mae"] += float(torch.mean(torch.abs(dry - wet))) * examples
        totals["finite"] = totals["finite"] and bool(torch.isfinite(prediction).all())
        peak_ratios.extend(
            (
                prediction.abs().amax(dim=1)
                / wet.abs().amax(dim=1).clamp_min(1.0e-6)
            ).cpu().tolist()
        )
    count = max(int(totals.pop("examples")), 1)
    finite = bool(totals.pop("finite"))
    metrics = {name: value / count for name, value in totals.items()}
    metrics["examples"] = count
    metrics["finite"] = finite
    ordered = sorted(float(value) for value in peak_ratios)
    percentile_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    metrics["peak_ratio_p95"] = ordered[percentile_index]
    metrics["worst_peak_ratio"] = ordered[-1]
    metrics["esr_improvement"] = 1.0 - metrics["model_esr"] / max(
        metrics["baseline_esr"], 1.0e-12
    )
    metrics["mae_improvement"] = 1.0 - metrics["model_mae"] / max(
        metrics["baseline_mae"], 1.0e-12
    )
    return metrics


def _dataset(
    corpus: Path,
    split: str,
    samples: int,
    frames: int,
    seed: int,
    domains: tuple[str, ...],
    input_peak_ranges: tuple[tuple[float, float], ...] | None = None,
) -> DriveForwardDataset:
    return DriveForwardDataset(
        dry_sources(corpus, split),
        samples,
        frames,
        seed=seed,
        domains=domains,
        input_peak_ranges=input_peak_ranges,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--train-samples", type=int, default=4_096)
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--frames", type=int, default=16_384)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--wide-amplitude", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path)
    args = parser.parse_args()

    if args.smoke:
        args.train_samples = 24
        args.calibration_samples = 6
        args.validation_samples = 6
        args.frames = 4_096
        args.epochs = 2
        args.batch_size = 4
    elif args.pilot:
        args.train_samples = 768
        args.calibration_samples = 48
        args.validation_samples = 48
        args.frames = 8_192
        args.epochs = 16
        args.batch_size = 8

    random.seed(20260830)
    np.random.seed(20260830)
    torch.manual_seed(20260830)
    target = device()
    input_peak_ranges = WIDE_INPUT_PEAK_RANGES if args.wide_amplitude else None
    model = DriveForwardRenderer(hidden_size=32).to(target)
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if (
            initial.get("schema") != 1
            or initial.get("sample_rate") != FORWARD_RATE
            or int(initial.get("hidden_size", -1)) != model.hidden_size
        ):
            raise ValueError("initial forward Drive checkpoint is incompatible")
        model.load_state_dict(initial["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=1.0e-5)
    train = _dataset(
        args.corpus,
        "train",
        args.train_samples,
        args.frames,
        20260830,
        TRAIN_DOMAINS,
        input_peak_ranges,
    )
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    calibration = {
        domain: DataLoader(
            _dataset(
                args.corpus,
                "calibrate",
                args.calibration_samples,
                args.frames,
                20260901,
                (domain,),
                input_peak_ranges,
            ),
            batch_size=args.batch_size,
            num_workers=0,
        )
        for domain in TRAIN_DOMAINS
    }

    best_score = float("inf")
    best_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        totals = {
            "loss": 0.0,
            "esr": 0.0,
            "preemphasis_esr": 0.0,
            "spectral": 0.0,
            "normalized_l1": 0.0,
            "relative_peak": 0.0,
            "amplitude_overshoot": 0.0,
        }
        batches = 0
        for batch in train_loader:
            dry, wet, controls = _tensor_batch(batch, target)
            prediction, _ = model(dry, controls)
            loss, parts = forward_loss(prediction, wet)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for name in parts:
                totals[name] += parts[name]
            batches += 1
        calibration_metrics = {
            domain: evaluate(model, loader, target)
            for domain, loader in calibration.items()
        }
        score = max(
            max(
                metrics["model_esr"] / max(metrics["baseline_esr"], 1.0e-12),
                metrics["model_mae"] / max(metrics["baseline_mae"], 1.0e-12),
                metrics["peak_ratio_p95"] / 1.25,
                metrics["worst_peak_ratio"] / 1.75,
            )
            for metrics in calibration_metrics.values()
        )
        epoch_report = {
            "epoch": epoch + 1,
            "train": {name: value / max(batches, 1) for name, value in totals.items()},
            "calibration_model_esr": {
                domain: metrics["model_esr"] for domain, metrics in calibration_metrics.items()
            },
            "selection_score": score,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True))
        if score < best_score:
            best_score = score
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("forward Drive training produced no checkpoint")
    model.load_state_dict(best_state)

    validation = {
        domain: evaluate(
            model,
            DataLoader(
                _dataset(
                    args.corpus,
                    "valid",
                    args.validation_samples,
                    args.frames,
                    20260831,
                    (domain,),
                    input_peak_ranges,
                ),
                batch_size=args.batch_size,
                num_workers=0,
            ),
            target,
        )
        for domain in VALIDATION_DOMAINS
    }
    promotable = all(
        metrics["finite"]
        and metrics["model_esr"] < metrics["baseline_esr"]
        and metrics["model_mae"] < metrics["baseline_mae"]
        and metrics["peak_ratio_p95"] <= 1.25
        and metrics["worst_peak_ratio"] <= 1.75
        for metrics in validation.values()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "drive-forward-candidate.pt"
    torch.save(
        {
            "schema": 1,
            "sample_rate": FORWARD_RATE,
            "hidden_size": model.hidden_size,
            "controls": ["gain_db", "tone", "level_db"],
            "state_dict": best_state,
        },
        checkpoint,
    )
    report = {
        "schema": 1,
        "status": "research-candidate" if promotable else "rejected-candidate",
        "promotable_to_bundle": promotable,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "sample_rate": FORWARD_RATE,
        "segment_frames": args.frames,
        "train_examples": args.train_samples,
        "calibration_examples_per_domain": args.calibration_samples,
        "validation_examples_per_domain": args.validation_samples,
        "epochs": args.epochs,
        "training_domains": list(TRAIN_DOMAINS),
        "input_peak_ranges": input_peak_ranges or ((0.08, 0.24),),
        "initial_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "initial_checkpoint_sha256": (
            hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest()
            if args.init_checkpoint
            else None
        ),
        "external_validation_domain": "pedalboard",
        "selection_score": best_score,
        "validation": validation,
        "history": history,
        "data_policy": {
            "source_files_read_only": True,
            "training_copy_level_augmentation": True,
            "runtime_automatic_normalization": False,
            "guitar_disjoint_splits": True,
            "real_hardware_capture": False,
            "locked_tele_test_opened": False,
            "challenge_renderer_opened": False,
        },
        "bundle_policy": "never export unless every validation domain beats dry bypass",
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
