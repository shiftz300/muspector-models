#!/usr/bin/env python3
"""Check whether deconvolved multiscale energy carries learnable Reverb controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

from .audit_reverb_decay import ANALYSIS_RATE, deconvolved_impulse
from .data import SyntheticControlDataset, TOPOLOGIES, dry_sources


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/corpus/guitar-effects-chains"
OUTPUT = ROOT / "remix/runs/reverb-feature-audit.json"


def features(impulse: np.ndarray) -> np.ndarray:
    energy = np.square(impulse, dtype=np.float64)
    duration = len(energy) / ANALYSIS_RATE
    logarithmic_edges = np.unique(
        np.clip(
            np.round(np.geomspace(0.002, duration, 97) * ANALYSIS_RATE).astype(int),
            0,
            len(energy),
        )
    )
    linear_edges = np.linspace(0, len(energy), 65, dtype=int)

    def pooled(edges: np.ndarray) -> list[float]:
        return [float(np.mean(energy[left:right])) for left, right in zip(edges, edges[1:])]

    values = np.asarray(pooled(logarithmic_edges) + pooled(linear_edges), dtype=np.float64)
    reference = max(float(np.max(values)), 1.0e-20)
    db = 10.0 * np.log10(np.maximum(values / reference, 1.0e-12))
    return np.clip(db, -100.0, 10.0).astype(np.float32) / 100.0


def matrix(
    root: Path,
    split: str,
    samples: int,
    seed: int,
    renderers: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    dataset = SyntheticControlDataset(
        dry_sources(root, split), samples, seed=seed, renderers=renderers
    )
    rows, targets = [], []
    for index in range(len(dataset)):
        if "reverb" not in TOPOLOGIES[index % len(TOPOLOGIES)]:
            continue
        item = dataset[index]
        impulse = deconvolved_impulse(item["dry"], item["wet"], 1.0e-5)
        rows.append(features(impulse))
        targets.append(item["controls"][6:9].numpy())
    return np.stack(rows), np.stack(targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--train-samples", type=int, default=1_600)
    parser.add_argument("--valid-samples", type=int, default=800)
    args = parser.parse_args()

    train_x, train_y = matrix(
        args.corpus,
        "train",
        args.train_samples,
        20260830,
        ("reference", "alternate"),
    )
    model = ExtraTreesRegressor(
        n_estimators=256,
        min_samples_leaf=2,
        max_features=0.75,
        n_jobs=-1,
        random_state=20260830,
    ).fit(train_x, train_y)
    validation = {}
    for renderer in ("reference", "alternate"):
        valid_x, valid_y = matrix(
            args.corpus,
            "valid",
            args.valid_samples,
            20260831,
            (renderer,),
        )
        prediction = np.clip(model.predict(valid_x), 0.0, 1.0)
        validation[renderer] = {
            name: float(mean_absolute_error(valid_y[:, index], prediction[:, index]))
            for index, name in enumerate(("decay", "damping", "mix"))
        }
    report = {
        "schema": 1,
        "purpose": "feature feasibility only; ExtraTrees is not a production model",
        "train_examples": len(train_x),
        "validation_examples_per_renderer": len(valid_x),
        "validation_normalized_mae": validation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
