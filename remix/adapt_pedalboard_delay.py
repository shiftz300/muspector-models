#!/usr/bin/env python3
"""Fine-tune a Delay residual candidate with Pedalboard and domain replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import TOPOLOGIES, dry_sources
from .delay_model import DelayControlEstimator, delay_features
from .pedalboard_data import PedalboardControlDataset
from .train import CORPUS, RUN, device
from .train_delay import evaluate, feature_matrix


@torch.no_grad()
def pedalboard_feature_matrix(
    corpus: Path, split: str, samples: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = PedalboardControlDataset(dry_sources(corpus, split), samples, seed=seed)
    rows, targets = [], []
    pending_dry, pending_wet, pending_targets = [], [], []

    def flush() -> None:
        if not pending_dry:
            return
        rows.append(delay_features(torch.stack(pending_dry), torch.stack(pending_wet)).cpu())
        targets.append(torch.stack(pending_targets))
        pending_dry.clear()
        pending_wet.clear()
        pending_targets.clear()

    for index in range(len(dataset)):
        if "delay" not in TOPOLOGIES[index % len(TOPOLOGIES)]:
            continue
        item = dataset[index]
        pending_dry.append(item["dry"])
        pending_wet.append(item["wet"])
        pending_targets.append(item["controls"][4:6])
        if len(pending_dry) == 8:
            flush()
    flush()
    return torch.cat(rows), torch.cat(targets)


def domain_metrics(
    model: DelayControlEstimator,
    domains: dict[str, tuple[torch.Tensor, torch.Tensor]],
    target_device: torch.device,
) -> dict[str, dict[str, float]]:
    return {
        name: evaluate(model, *values, target_device)
        for name, values in domains.items()
    }


def selection_score(metrics: dict[str, dict[str, float]]) -> float:
    return max(sum(domain.values()) / len(domain) for domain in metrics.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=RUN / "delay-estimator.pt")
    parser.add_argument("--output", type=Path, default=RUN / "delay-pedalboard-candidate.pt")
    parser.add_argument("--metrics", type=Path, default=RUN / "delay-pedalboard-candidate.json")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--pedalboard-train-samples", type=int, default=6_400)
    parser.add_argument("--replay-train-samples", type=int, default=9_600)
    parser.add_argument("--calibration-samples", type=int, default=800)
    parser.add_argument("--pedalboard-calibration-samples", type=int, default=1_600)
    parser.add_argument("--validation-samples", type=int, default=800)
    parser.add_argument("--pedalboard-validation-samples", type=int, default=1_600)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    random.seed(20260910)
    np.random.seed(20260910)
    torch.manual_seed(20260910)
    replay = feature_matrix(
        args.corpus,
        "train",
        args.replay_train_samples,
        20260830,
        ("reference", "alternate", "stress"),
    )
    pedalboard_train = pedalboard_feature_matrix(
        args.corpus, "train", args.pedalboard_train_samples, 20260910
    )
    train = (
        torch.cat((replay[0], pedalboard_train[0])),
        torch.cat((replay[1], pedalboard_train[1])),
    )
    calibration = {
        renderer: feature_matrix(
            args.corpus,
            "calibrate",
            args.calibration_samples,
            20260901,
            (renderer,),
        )
        for renderer in ("reference", "alternate", "stress")
    }
    calibration["pedalboard"] = pedalboard_feature_matrix(
        args.corpus, "calibrate", args.pedalboard_calibration_samples, 20260911
    )
    validation = {
        renderer: feature_matrix(
            args.corpus,
            "valid",
            args.validation_samples,
            20260831,
            (renderer,),
        )
        for renderer in ("reference", "alternate", "stress")
    }
    validation["pedalboard"] = pedalboard_feature_matrix(
        args.corpus, "valid", args.pedalboard_validation_samples, 20260904
    )

    target_device = device()
    model = DelayControlEstimator().to(target_device)
    model.load_state_dict(torch.load(args.source, map_location=target_device, weights_only=True))
    baseline_calibration = domain_metrics(model, calibration, target_device)
    baseline_validation = domain_metrics(model, validation, target_device)
    best_score = selection_score(baseline_calibration)
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1.0e-4)
    loader = DataLoader(TensorDataset(*train), batch_size=64, shuffle=True)
    for epoch in range(args.epochs):
        model.train()
        for features, targets in loader:
            prediction = torch.sigmoid(model(features.to(target_device)))
            loss = torch.nn.functional.smooth_l1_loss(
                prediction, targets.to(target_device), beta=0.05
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        current = domain_metrics(model, calibration, target_device)
        score = selection_score(current)
        if score < best_score:
            best_score = score
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(json.dumps({"epoch": epoch + 1, "calibration": current}, sort_keys=True))

    model.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output)
    report = {
        "schema": 1,
        "candidate_only": True,
        "source_checkpoint_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "candidate_checkpoint_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "train_examples": len(train[0]),
        "replay_examples": len(replay[0]),
        "pedalboard_examples": len(pedalboard_train[0]),
        "calibration_baseline": baseline_calibration,
        "calibration_selected": domain_metrics(model, calibration, target_device),
        "validation_baseline": baseline_validation,
        "validation_selected": domain_metrics(model, validation, target_device),
    }
    args.metrics.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
