#!/usr/bin/env python3
"""Check that the differentiable surrogate prefers true over wrong controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import SyntheticControlDataset, dry_sources
from .differentiable import audio_loss
from .model import Estimate
from .train import CORPUS


def logits(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(1.0e-4, 1.0 - 1.0e-4)
    return torch.log(value / (1.0 - value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()

    dataset = SyntheticControlDataset(
        dry_sources(args.corpus, "valid"), args.samples, seed=20260832
    )
    rows = []
    for index in range(len(dataset)):
        item = dataset[index]
        if not item["reconstruction_mask"]:
            continue
        batch = {name: value.unsqueeze(0) for name, value in item.items()}
        true = item["controls"]
        wrong = torch.remainder(true + 0.35, 1.0)
        zeros = torch.zeros((1, 3))
        true_loss = float(audio_loss(Estimate(zeros, logits(true)[None]), batch, 1))
        wrong_loss = float(audio_loss(Estimate(zeros, logits(wrong)[None]), batch, 1))
        rows.append(
            {
                "index": index,
                "topology": item["topology"].tolist(),
                "renderer": int(item["renderer"]),
                "true": true_loss,
                "wrong": wrong_loss,
                "true_is_better": true_loss < wrong_loss,
            }
        )
    report = {
        "schema": 1,
        "eligible": len(rows),
        "true_better": sum(row["true_is_better"] for row in rows),
        "true_better_rate": sum(row["true_is_better"] for row in rows) / max(len(rows), 1),
        "mean_true_loss": sum(row["true"] for row in rows) / max(len(rows), 1),
        "mean_wrong_loss": sum(row["wrong"] for row in rows) / max(len(rows), 1),
        "rows": rows,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
