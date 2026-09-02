#!/usr/bin/env python3
"""Train the production Reverb-control companion head."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import SyntheticControlDataset, TOPOLOGIES, dry_sources
from .reverb_model import ReverbControlEstimator, reverb_features
from .train import CORPUS, RUN, device


def feature_matrix(
    corpus: Path,
    split: str,
    samples: int,
    seed: int,
    renderers: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = SyntheticControlDataset(
        dry_sources(corpus, split), samples, seed=seed, renderers=renderers
    )
    rows, targets = [], []
    pending_dry, pending_wet, pending_targets = [], [], []

    def flush() -> None:
        if not pending_dry:
            return
        rows.append(reverb_features(torch.stack(pending_dry), torch.stack(pending_wet)).cpu())
        targets.append(torch.stack(pending_targets))
        pending_dry.clear()
        pending_wet.clear()
        pending_targets.clear()

    for index in range(len(dataset)):
        if "reverb" not in TOPOLOGIES[index % len(TOPOLOGIES)]:
            continue
        item = dataset[index]
        pending_dry.append(item["dry"])
        pending_wet.append(item["wet"])
        pending_targets.append(item["controls"][6:9])
        if len(pending_dry) == 8:
            flush()
    flush()
    return torch.cat(rows), torch.cat(targets)


@torch.no_grad()
def evaluate(
    model: ReverbControlEstimator,
    features: torch.Tensor,
    targets: torch.Tensor,
    target_device: torch.device,
) -> dict[str, float]:
    model.eval()
    prediction = []
    for (batch,) in DataLoader(TensorDataset(features), batch_size=128):
        prediction.append(torch.sigmoid(model(batch.to(target_device))).cpu())
    absolute = torch.abs(torch.cat(prediction) - targets).mean(dim=0)
    return {
        name: float(absolute[index])
        for index, name in enumerate(("decay", "damping", "mix"))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--train-samples", type=int, default=19_200)
    parser.add_argument("--valid-samples", type=int, default=1_600)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    random.seed(20260830)
    np.random.seed(20260830)
    torch.manual_seed(20260830)
    train_x, train_y = feature_matrix(
        args.corpus,
        "train",
        args.train_samples,
        20260830,
        ("reference", "alternate", "stress"),
    )
    calibration = {
        renderer: feature_matrix(
            args.corpus,
            "calibrate",
            args.valid_samples,
            20260901,
            (renderer,),
        )
        for renderer in ("reference", "alternate", "stress")
    }
    validation = {
        renderer: feature_matrix(
            args.corpus,
            "valid",
            args.valid_samples,
            20260831,
            (renderer,),
        )
        for renderer in ("reference", "alternate", "stress")
    }
    target_device = device()
    model = ReverbControlEstimator().to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=1.0e-4)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
    best_score = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        for features, targets in loader:
            prediction = model(features.to(target_device))
            loss = torch.nn.functional.smooth_l1_loss(
                torch.sigmoid(prediction), targets.to(target_device), beta=0.05
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = {
            renderer: evaluate(model, *values, target_device)
            for renderer, values in calibration.items()
        }
        score = max(
            sum(renderer.values()) / len(renderer) for renderer in metrics.values()
        )
        if score < best_score:
            best_score = score
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(json.dumps({"epoch": epoch + 1, "calibration": metrics}, sort_keys=True))

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    calibration_metrics = {
        renderer: evaluate(model, *values, target_device)
        for renderer, values in calibration.items()
    }
    validation_metrics = {
        renderer: evaluate(model, *values, target_device)
        for renderer, values in validation.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "reverb-estimator.pt"
    torch.save(best_state, checkpoint)
    report = {
        "schema": 1,
        "train_examples": len(train_x),
        "calibration_examples_per_renderer": len(next(iter(calibration.values()))[0]),
        "validation_examples_per_renderer": len(next(iter(validation.values()))[0]),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "calibration_normalized_mae": calibration_metrics,
        "validation_normalized_mae": validation_metrics,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    (args.output / "reverb-metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
