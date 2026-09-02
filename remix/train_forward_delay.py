#!/usr/bin/env python3
"""Train and reject-or-promote the hybrid Delay forward renderer."""

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
from .forward_delay import DelayForwardDataset, DelayForwardRenderer
from .forward_drive import FORWARD_RATE, error_to_signal_ratio, forward_loss
from .train import CORPUS, device


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/delay-forward-pilot"
TRAIN_DOMAINS = ("reference", "alternate", "stress")
VALIDATION_DOMAINS = (*TRAIN_DOMAINS, "pedalboard")


@torch.no_grad()
def evaluate(model: DelayForwardRenderer, loader: DataLoader, target: torch.device) -> dict:
    model.eval()
    totals = {"model_esr": 0.0, "baseline_esr": 0.0, "model_mae": 0.0, "baseline_mae": 0.0}
    ratios = []
    examples = 0
    finite = True
    for batch in loader:
        dry = batch["dry"].to(target)
        wet = batch["wet"].to(target)
        controls = batch["controls"].to(target)
        prediction = model(dry, controls)
        count = dry.shape[0]
        examples += count
        totals["model_esr"] += float(error_to_signal_ratio(prediction, wet)) * count
        totals["baseline_esr"] += float(error_to_signal_ratio(dry, wet)) * count
        totals["model_mae"] += float(torch.mean(torch.abs(prediction - wet))) * count
        totals["baseline_mae"] += float(torch.mean(torch.abs(dry - wet))) * count
        ratios.extend(
            (
                prediction.abs().amax(dim=1)
                / wet.abs().amax(dim=1).clamp_min(1.0e-6)
            ).cpu().tolist()
        )
        finite = finite and bool(torch.isfinite(prediction).all())
    metrics = {name: value / examples for name, value in totals.items()}
    ordered = sorted(float(value) for value in ratios)
    index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    metrics.update(
        {
            "examples": examples,
            "finite": finite,
            "peak_ratio_p95": ordered[index],
            "worst_peak_ratio": ordered[-1],
            "esr_improvement": 1.0 - metrics["model_esr"] / max(metrics["baseline_esr"], 1.0e-12),
            "mae_improvement": 1.0 - metrics["model_mae"] / max(metrics["baseline_mae"], 1.0e-12),
        }
    )
    return metrics


def dataset(
    corpus: Path,
    split: str,
    samples: int,
    frames: int,
    seed: int,
    domains: tuple[str, ...],
) -> DelayForwardDataset:
    return DelayForwardDataset(
        dry_sources(corpus, split),
        samples,
        frames,
        seed=seed,
        domains=domains,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--train-samples", type=int, default=96)
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--frames", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.train_samples = 12
        args.calibration_samples = 6
        args.validation_samples = 6
        args.frames = 49_000
        args.epochs = 2
        args.batch_size = 2

    random.seed(20261020)
    np.random.seed(20261020)
    torch.manual_seed(20261020)
    target = device()
    model = DelayForwardRenderer(fir_taps=64).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-2, weight_decay=0.0)
    train_loader = DataLoader(
        dataset(args.corpus, "train", args.train_samples, args.frames, 20261020, TRAIN_DOMAINS),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    calibration = {
        domain: DataLoader(
            dataset(args.corpus, "calibrate", args.calibration_samples, args.frames, 20261021, (domain,)),
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
        totals = {"loss": 0.0, "esr": 0.0, "relative_peak": 0.0}
        batches = 0
        for batch in train_loader:
            dry = batch["dry"].to(target)
            wet = batch["wet"].to(target)
            controls = batch["controls"].to(target)
            prediction = model(dry, controls)
            loss, parts = forward_loss(prediction, wet)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["esr"] += parts["esr"]
            totals["relative_peak"] += parts["relative_peak"]
            batches += 1
        calibration_metrics = {
            domain: evaluate(model, loader, target) for domain, loader in calibration.items()
        }
        score = max(
            max(
                values["model_esr"] / max(values["baseline_esr"], 1.0e-12),
                values["model_mae"] / max(values["baseline_mae"], 1.0e-12),
                values["peak_ratio_p95"] / 1.25,
                values["worst_peak_ratio"] / 1.75,
            )
            for values in calibration_metrics.values()
        )
        report = {
            "epoch": epoch + 1,
            "selection_score": score,
            "train": {name: value / batches for name, value in totals.items()},
            "learned": model.learned_parameters(),
        }
        history.append(report)
        print(json.dumps(report, sort_keys=True))
        if score < best_score:
            best_score = score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Delay forward training produced no checkpoint")
    model.load_state_dict(best_state)
    validation = {
        domain: evaluate(
            model,
            DataLoader(
                dataset(args.corpus, "valid", args.validation_samples, args.frames, 20261022, (domain,)),
                batch_size=args.batch_size,
                num_workers=0,
            ),
            target,
        )
        for domain in VALIDATION_DOMAINS
    }
    promotable = all(
        values["finite"]
        and values["model_esr"] < values["baseline_esr"]
        and values["model_mae"] < values["baseline_mae"]
        and values["peak_ratio_p95"] <= 1.25
        and values["worst_peak_ratio"] <= 1.75
        for values in validation.values()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "delay-forward-candidate.pt"
    torch.save(
        {
            "schema": 1,
            "sample_rate": FORWARD_RATE,
            "fir_taps": model.fir_taps,
            "controls": ["time_ms", "feedback", "mix"],
            "state_dict": best_state,
        },
        checkpoint,
    )
    result = {
        "schema": 1,
        "status": "research-candidate" if promotable else "rejected-candidate",
        "promotable": promotable,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "sample_rate": FORWARD_RATE,
        "frames": args.frames,
        "train_examples": args.train_samples,
        "calibration_examples_per_domain": args.calibration_samples,
        "validation_examples_per_domain": args.validation_samples,
        "selection_score": best_score,
        "learned": model.learned_parameters(),
        "validation": validation,
        "history": history,
        "policy": {
            "time_is_exact_physical_delay": True,
            "feedback_is_exact": True,
            "mix_is_exact": True,
            "source_files_read_only": True,
            "runtime_automatic_normalization": False,
            "physical_audio_devices_used": False,
            "challenge_renderer_opened": False,
            "locked_tele_test_opened": False,
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
