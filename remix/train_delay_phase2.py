#!/usr/bin/env python3
"""Fine-tune Delay inverse controls for both isolated and mixed chains."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .adapt_pedalboard_delay import pedalboard_feature_matrix
from .delay_model import DelayControlEstimator
from .train import CORPUS, RUN, device
from .train_delay import delay_only_feature_matrix, evaluate, feature_matrix


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "remix/runs/delay-inverse-phase2"
DOMAINS = ("reference", "alternate", "stress", "pedalboard")


def _metrics(
    model: DelayControlEstimator,
    matrices: dict[str, tuple[torch.Tensor, torch.Tensor]],
    target: torch.device,
) -> dict[str, dict[str, float]]:
    return {name: evaluate(model, *matrix, target) for name, matrix in matrices.items()}


def _score(
    current: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
) -> float:
    ratios = []
    for name, values in current.items():
        for control, value in values.items():
            ratios.append(value / max(baseline[name][control], 1.0e-6))
    return max(ratios) + float(np.mean(ratios))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=RUN / "delay-estimator.pt")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--replay-samples", type=int, default=6_400)
    parser.add_argument("--pedalboard-replay-samples", type=int, default=3_200)
    parser.add_argument("--single-samples-per-domain", type=int, default=1_000)
    parser.add_argument("--calibration-samples", type=int, default=320)
    parser.add_argument("--validation-samples", type=int, default=480)
    parser.add_argument("--epochs", type=int, default=35)
    args = parser.parse_args()

    random.seed(20261110)
    np.random.seed(20261110)
    torch.manual_seed(20261110)
    args.output.mkdir(parents=True, exist_ok=True)

    replay = feature_matrix(
        args.corpus,
        "train",
        args.replay_samples,
        20261110,
        ("reference", "alternate", "stress"),
    )
    pedalboard_replay = pedalboard_feature_matrix(
        args.corpus, "train", args.pedalboard_replay_samples, 20261111
    )
    single_train = {
        domain: delay_only_feature_matrix(
            args.corpus,
            "train",
            args.single_samples_per_domain,
            20261120 + index,
            domain,
        )
        for index, domain in enumerate(DOMAINS)
    }
    train = (
        torch.cat((replay[0], pedalboard_replay[0], *(value[0] for value in single_train.values()))),
        torch.cat((replay[1], pedalboard_replay[1], *(value[1] for value in single_train.values()))),
    )

    calibration = {}
    validation = {}
    for index, domain in enumerate(DOMAINS):
        if domain == "pedalboard":
            calibration[f"mixed_{domain}"] = pedalboard_feature_matrix(
                args.corpus, "calibrate", args.calibration_samples, 20261140
            )
            validation[f"mixed_{domain}"] = pedalboard_feature_matrix(
                args.corpus, "valid", args.validation_samples, 20261150
            )
        else:
            calibration[f"mixed_{domain}"] = feature_matrix(
                args.corpus, "calibrate", args.calibration_samples, 20261140, (domain,)
            )
            validation[f"mixed_{domain}"] = feature_matrix(
                args.corpus, "valid", args.validation_samples, 20261150, (domain,)
            )
        calibration[f"single_{domain}"] = delay_only_feature_matrix(
            args.corpus, "calibrate", args.calibration_samples, 20261160 + index, domain
        )
        validation[f"single_{domain}"] = delay_only_feature_matrix(
            args.corpus, "valid", args.validation_samples, 20261170 + index, domain
        )

    target = device()
    model = DelayControlEstimator().to(target)
    model.load_state_dict(torch.load(args.source, map_location=target, weights_only=True))
    baseline_calibration = _metrics(model, calibration, target)
    baseline_validation = _metrics(model, validation, target)
    best_score = _score(baseline_calibration, baseline_calibration)
    best_epoch = 0
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
    generator = torch.Generator().manual_seed(20261110)
    loader = DataLoader(
        TensorDataset(*train), batch_size=64, shuffle=True, generator=generator
    )
    weights = torch.tensor((1.25, 1.0), device=target)
    for epoch in range(args.epochs):
        model.train()
        for features, targets in loader:
            prediction = torch.sigmoid(model(features.to(target)))
            element_loss = torch.nn.functional.smooth_l1_loss(
                prediction, targets.to(target), beta=0.04, reduction="none"
            )
            loss = (element_loss * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        current = _metrics(model, calibration, target)
        score = _score(current, baseline_calibration)
        if score < best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(json.dumps({"epoch": epoch + 1, "score": score}, sort_keys=True))

    model.load_state_dict(best_state)
    selected_calibration = _metrics(model, calibration, target)
    selected_validation = _metrics(model, validation, target)
    checkpoint = args.output / "delay-estimator-candidate.pt"
    torch.save(best_state, checkpoint)
    regressions = {
        name: {
            control: selected_validation[name][control] / max(value, 1.0e-6)
            for control, value in baseline_validation[name].items()
        }
        for name in baseline_validation
    }
    single_ratios = [
        value
        for name, values in regressions.items()
        if name.startswith("single_")
        for value in values.values()
    ]
    mixed_ratios = [
        value
        for name, values in regressions.items()
        if name.startswith("mixed_")
        for value in values.values()
    ]
    promotable = bool(
        best_epoch > 0
        and float(np.mean(single_ratios)) <= 0.92
        and max(single_ratios) <= 1.05
        and float(np.mean(mixed_ratios)) <= 1.0
        and max(mixed_ratios) <= 1.10
    )
    report = {
        "schema": 1,
        "status": "candidate" if promotable else "rejected",
        "promotable": promotable,
        "best_epoch": best_epoch,
        "train_examples": len(train[0]),
        "source_checkpoint": str(args.source),
        "source_checkpoint_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "candidate_checkpoint": str(checkpoint),
        "candidate_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "baseline_calibration": baseline_calibration,
        "selected_calibration": selected_calibration,
        "baseline_validation": baseline_validation,
        "selected_validation": selected_validation,
        "validation_ratios": regressions,
        "mean_single_validation_ratio": float(np.mean(single_ratios)),
        "mean_mixed_validation_ratio": float(np.mean(mixed_ratios)),
        "policy": {
            "challenge_renderer_opened": False,
            "locked_tele_test_opened": False,
            "physical_audio_devices_used": False,
            "source_files_read_only": True,
            "automatic_normalization": False,
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))
    if not promotable:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
