"""Refit independent convex effect heads on a frozen Blind v2 encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .blind2 import LABELS, BlindPresence, log_mel
from .blind_data2 import BlindPresenceData, inventory
from .train_blind2 import (
    SEED,
    _acceptance,
    _fit_calibration,
    _loader,
    _metrics,
    _probabilities,
    _thresholds,
)


def _collect_features(
    model: BlindPresence,
    dataset: BlindPresenceData,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    features, targets, masks, sources = [], [], [], []
    with torch.no_grad():
        for batch in _loader(dataset, 16, False):
            value = model.features(log_mel(batch["audio"].to(device)))
            features.append(value.cpu().numpy())
            targets.append(batch["target"].numpy())
            masks.append(batch["target_mask"].numpy())
            sources.extend(batch["source"])
    return np.concatenate(features), np.concatenate(targets), np.concatenate(masks), sources


def _fit_heads(
    features: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    weights, biases = [], []
    for index in range(len(LABELS)):
        admitted = masks[:, index].astype(bool)
        expected = targets[admitted, index].astype(np.int64)
        if len(np.unique(expected)) != 2:
            raise ValueError(f"head {LABELS[index]} does not have both classes")
        head = LogisticRegression(
            C=regularization,
            solver="lbfgs",
            max_iter=2_000,
            random_state=SEED,
        )
        head.fit(features[admitted], expected)
        weights.append(head.coef_[0])
        biases.append(head.intercept_[0])
    return np.asarray(weights, dtype=np.float32), np.asarray(biases, dtype=np.float32)


def _evaluate(
    features: np.ndarray,
    targets: np.ndarray,
    sources: list[str],
    weights: np.ndarray,
    biases: np.ndarray,
    calibration: list[dict] | None = None,
    thresholds: list[float] | None = None,
) -> tuple[dict, list[dict], list[float]]:
    logits = features @ weights.T + biases
    calibration = calibration or _fit_calibration(logits, targets)
    probabilities = _probabilities(logits, calibration)
    thresholds = thresholds or _thresholds(probabilities, targets)
    return _metrics(probabilities, targets, thresholds, sources), calibration, thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--fit-epochs", type=int, default=4)
    args = parser.parse_args()
    if args.fit_epochs < 1:
        raise ValueError("fit epochs must be positive")
    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = BlindPresence()
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    started = time.perf_counter()

    fit_data = BlindPresenceData(Path.cwd(), "fit", 2_400, SEED + 11)
    fit_parts = []
    for epoch in range(args.fit_epochs):
        fit_data.set_epoch(epoch)
        fit_parts.append(_collect_features(model, fit_data, device))
        print(json.dumps({"stage": "features", "fit_epoch": epoch + 1}), flush=True)
    fit_features = np.concatenate([row[0] for row in fit_parts])
    fit_targets = np.concatenate([row[1] for row in fit_parts])
    fit_masks = np.concatenate([row[2] for row in fit_parts])

    calibration_data = BlindPresenceData(Path.cwd(), "calibration", 480, SEED + 23)
    development_data = BlindPresenceData(Path.cwd(), "development", 640, SEED + 37)
    calibration_features, calibration_targets, _, calibration_sources = _collect_features(
        model, calibration_data, device
    )
    development_features, development_targets, _, development_sources = _collect_features(
        model, development_data, device
    )

    candidates = []
    for regularization in (0.01, 0.1, 1.0):
        weights, biases = _fit_heads(fit_features, fit_targets, fit_masks, regularization)
        metrics, calibration, thresholds = _evaluate(
            calibration_features,
            calibration_targets,
            calibration_sources,
            weights,
            biases,
        )
        candidates.append(
            {
                "regularization": regularization,
                "weights": weights,
                "biases": biases,
                "metrics": metrics,
                "calibration": calibration,
                "thresholds": thresholds,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "head-search",
                    "C": regularization,
                    "macro_f1": metrics["macro_f1"],
                    "minimum_recall": min(row["recall"] for row in metrics["per_label"].values()),
                }
            ),
            flush=True,
        )
    selected = max(
        candidates,
        key=lambda row: (
            min(item["recall"] for item in row["metrics"]["per_label"].values()),
            row["metrics"]["macro_f1"],
            -row["regularization"],
        ),
    )
    development, _, _ = _evaluate(
        development_features,
        development_targets,
        development_sources,
        selected["weights"],
        selected["biases"],
        selected["calibration"],
        selected["thresholds"],
    )
    acceptance = _acceptance(development)
    model.head.weight.data.copy_(torch.from_numpy(selected["weights"]).to(device))
    model.head.bias.data.copy_(torch.from_numpy(selected["biases"]).to(device))
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    args.output.mkdir(parents=True, exist_ok=True)
    output_checkpoint = args.output / "model.pt"
    torch.save(
        {
            "schema": 1,
            "model": state,
            "manifest": model.manifest(),
            "calibration": selected["calibration"],
            "thresholds": selected["thresholds"],
        },
        output_checkpoint,
    )
    report = {
        "schema": 1,
        "status": "development-accepted-not-promoted" if acceptance["accepted"] else "development-rejected",
        "scope": "Wet-only independent effect-presence heads; no order, controls or inverse changes",
        "device": args.device,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "fit_feature_examples": len(fit_features),
        "fit_feature_epochs": args.fit_epochs,
        "head": {
            "kind": "four independent convex logistic heads copied into the PyTorch linear layer",
            "selected_C": selected["regularization"],
            "search": [
                {
                    "C": row["regularization"],
                    "macro_f1": row["metrics"]["macro_f1"],
                    "minimum_recall": min(
                        item["recall"] for item in row["metrics"]["per_label"].values()
                    ),
                }
                for row in candidates
            ],
        },
        "calibration": selected["metrics"],
        "development": development,
        "acceptance": acceptance,
        "inventory": inventory(Path.cwd()),
        "boundaries": {
            "encoder_frozen": True,
            "heads_independent": True,
            "order_labels_used": False,
            "controls_used": False,
            "inverse_weights_changed": False,
            "locked_final_opened": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "artifact": {
            "path": str(output_checkpoint.resolve()),
            "sha256": hashlib.sha256(output_checkpoint.read_bytes()).hexdigest(),
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
