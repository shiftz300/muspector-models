"""One bounded hard-example pass for the order-independent Drive presence expert.

This is deliberately narrower than the generic Blind family trainer:

* it accepts only an existing one-head nonlinear expert;
* it trains at most four epochs and includes epoch zero in calibration selection;
* weak, low-drive, and single-effect synthetic positives receive fixed weights;
* checkpoint selection uses calibration only, then opens development exactly once;
* rejected weights are not written to disk.

Ambience and other effects can occur as immutable mixed-chain context in the
existing product-safe renderer, but only the nonlinear head receives gradients.
No order or control labels are consumed or emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from .blind2 import BlindFamilyPresence, log_mel
from .blind_data2 import BlindPresenceData, inventory
from .train_blind2 import (
    SEED,
    _device,
    _fit_calibration,
    _loader,
    _probabilities,
    _thresholds,
)
from .train_blind2_family import _collect, _family_target, _weighted_masked_bce


MAX_EPOCHS = 4
WEIGHT_RULES = {
    "synthetic_count_1_positive": 3.0,
    "synthetic_low_drive_positive": 2.0,
    "synthetic_weak_residual_positive": 2.0,
    "maximum_combined_weight": 6.0,
}
FROZEN_GATES = {
    "f1": 0.80,
    "recall": 0.80,
    "negative_false_positive_rate": 0.05,
    "synthetic_count_1_recall": 0.75,
    "synthetic_low_drive_recall": 0.75,
    "synthetic_weak_residual_recall": 0.70,
    "egfxset_real_recall": 0.80,
    "dafx_random_position_recall": 0.80,
}


def _parse_nonlinear_source(source: str) -> dict[str, str] | None:
    prefix = "synthetic-product-render:nonlinear:"
    if not source.startswith(prefix):
        return None
    parts = source.split(":")
    if len(parts) != 7:
        raise ValueError(f"invalid nonlinear diagnostic source: {source}")
    return {
        "implementation": parts[2],
        "drive": parts[3],
        "residual": parts[4],
        "chain_count": parts[5],
        "clean_source": parts[6],
    }


def _hard_weights(expected: torch.Tensor, sources: list[str]) -> torch.Tensor:
    """Return the frozen positive-only weighting vector for one batch."""

    if expected.ndim != 2 or expected.shape[1] != 1:
        raise ValueError("nonlinear hard weights require a [batch, 1] target")
    if expected.shape[0] != len(sources):
        raise ValueError("nonlinear hard weights require one source per target")
    values = torch.ones_like(expected)
    for index, source in enumerate(sources):
        if float(expected[index, 0].detach().cpu()) < 0.5:
            continue
        details = _parse_nonlinear_source(source)
        if details is None:
            continue
        weight = 1.0
        if details["chain_count"] == "count-1":
            weight *= WEIGHT_RULES["synthetic_count_1_positive"]
        if details["drive"] == "low":
            weight *= WEIGHT_RULES["synthetic_low_drive_positive"]
        if details["residual"] == "weak":
            weight *= WEIGHT_RULES["synthetic_weak_residual_positive"]
        values[index, 0] = min(weight, WEIGHT_RULES["maximum_combined_weight"])
    return values


def _stratum_metrics(expected: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> dict:
    selected_expected = expected[mask]
    selected_predicted = predicted[mask]
    precision, recall, f1, _ = precision_recall_fscore_support(
        selected_expected,
        selected_predicted,
        average="binary",
        zero_division=0,
    )
    return {
        "examples": int(np.sum(mask)),
        "positive_examples": int(np.sum(selected_expected)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    sources: list[str],
) -> dict:
    if probabilities.shape != targets.shape or targets.ndim != 2 or targets.shape[1] != 1:
        raise ValueError("nonlinear metrics require matching [examples, 1] matrices")
    if len(sources) != targets.shape[0]:
        raise ValueError("nonlinear metrics require one source per example")
    expected = targets[:, 0].astype(bool)
    predicted = probabilities[:, 0] >= threshold
    source_array = np.asarray(sources, dtype=str)
    parsed = [_parse_nonlinear_source(source) for source in sources]
    masks = {
        "egfxset-real": source_array == "egfxset-real",
        "dafx-random-position-presence": source_array == "dafx-random-position-presence",
        "synthetic-nonlinear-chain-count:count-1": np.asarray(
            [row is not None and row["chain_count"] == "count-1" for row in parsed]
        ),
        "synthetic-nonlinear-drive:low": np.asarray(
            [row is not None and row["drive"] == "low" for row in parsed]
        ),
        "synthetic-nonlinear-residual:weak": np.asarray(
            [row is not None and row["residual"] == "weak" for row in parsed]
        ),
    }
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected, predicted, average="binary", zero_division=0
    )
    negative = ~expected
    return {
        "examples": len(expected),
        "positive_examples": int(np.sum(expected)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "negative_false_positive_rate": (
            float(np.mean(predicted[negative])) if np.any(negative) else None
        ),
        "strata": {
            name: _stratum_metrics(expected, predicted, mask)
            for name, mask in masks.items()
            if np.any(mask)
        },
    }


def _acceptance(metrics: dict) -> dict:
    strata = metrics["strata"]

    def recall(name: str) -> float:
        return float(strata.get(name, {}).get("recall", 0.0))

    false_positive_rate = metrics["negative_false_positive_rate"]
    gates = {
        "f1": metrics["f1"] >= FROZEN_GATES["f1"],
        "recall": metrics["recall"] >= FROZEN_GATES["recall"],
        "negative_false_positive_rate": (
            false_positive_rate is not None
            and false_positive_rate <= FROZEN_GATES["negative_false_positive_rate"]
        ),
        "synthetic_count_1_recall": recall(
            "synthetic-nonlinear-chain-count:count-1"
        )
        >= FROZEN_GATES["synthetic_count_1_recall"],
        "synthetic_low_drive_recall": recall("synthetic-nonlinear-drive:low")
        >= FROZEN_GATES["synthetic_low_drive_recall"],
        "synthetic_weak_residual_recall": recall("synthetic-nonlinear-residual:weak")
        >= FROZEN_GATES["synthetic_weak_residual_recall"],
        "egfxset_real_recall": recall("egfxset-real")
        >= FROZEN_GATES["egfxset_real_recall"],
        "dafx_random_position_recall": recall("dafx-random-position-presence")
        >= FROZEN_GATES["dafx_random_position_recall"],
    }
    return {"accepted": all(gates.values()), "gates": gates}


def _selection_key(metrics: dict) -> tuple:
    acceptance = _acceptance(metrics)
    hard_recalls = (
        metrics["recall"],
        metrics["strata"].get("synthetic-nonlinear-chain-count:count-1", {}).get(
            "recall", 0.0
        ),
        metrics["strata"].get("synthetic-nonlinear-drive:low", {}).get("recall", 0.0),
        metrics["strata"].get("synthetic-nonlinear-residual:weak", {}).get(
            "recall", 0.0
        ),
        metrics["strata"].get("egfxset-real", {}).get("recall", 0.0),
        metrics["strata"].get("dafx-random-position-presence", {}).get("recall", 0.0),
    )
    false_positive_rate = metrics["negative_false_positive_rate"]
    return (
        acceptance["accepted"],
        sum(acceptance["gates"].values()),
        min(hard_recalls),
        metrics["f1"],
        -(float(false_positive_rate) if false_positive_rate is not None else 1.0),
    )


def _load_model(checkpoint: dict) -> BlindFamilyPresence:
    manifest = checkpoint.get("manifest", {})
    if manifest.get("architecture") != "compact-audio-resnet18-family-expert":
        raise ValueError("source must be a compact Blind family expert")
    if manifest.get("family") != "nonlinear" or manifest.get("labels") != ["nonlinear"]:
        raise ValueError("source must be the standalone nonlinear expert")
    model = BlindFamilyPresence("nonlinear")
    model.load_state_dict(checkpoint["model"])
    return model


def _calibration_candidate(
    model: BlindFamilyPresence,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    loss: float | None,
) -> dict:
    logits, expected, sources = _collect(model, loader, device, "nonlinear")
    calibration = _fit_calibration(logits, expected)
    probabilities = _probabilities(logits, calibration)
    threshold = _thresholds(probabilities, expected)[0]
    metrics = _metrics(probabilities, expected, threshold, sources)
    return {
        "epoch": epoch,
        "loss": loss,
        "calibration": calibration,
        "threshold": threshold,
        "metrics": metrics,
        "acceptance": _acceptance(metrics),
        "selection_key": list(_selection_key(metrics)),
        "state": {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    if not 1 <= args.epochs <= MAX_EPOCHS:
        raise ValueError(f"epochs must be between 1 and {MAX_EPOCHS}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = _device(args.device)
    source_sha256 = hashlib.sha256(args.source.read_bytes()).hexdigest()
    source_checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    model = _load_model(source_checkpoint).to(device)

    fit = BlindPresenceData(Path.cwd(), "fit", 2_400, SEED + 11)
    calibration_data = BlindPresenceData(Path.cwd(), "calibration", 480, SEED + 23)
    development_data = BlindPresenceData(Path.cwd(), "development", 640, SEED + 37)
    fit_loader = _loader(fit, 16, True)
    calibration_loader = _loader(calibration_data, 16, False)
    development_loader = _loader(development_data, 16, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    started = time.perf_counter()
    candidates = [_calibration_candidate(model, calibration_loader, device, 0, None)]
    print(json.dumps({key: value for key, value in candidates[0].items() if key != "state"}), flush=True)
    for epoch in range(args.epochs):
        fit.set_epoch(epoch)
        model.train()
        total = 0.0
        batches = 0
        for batch in fit_loader:
            audio = batch["audio"].to(device)
            expected = _family_target(batch["target"], "nonlinear").to(device)
            mask = _family_target(batch["target_mask"], "nonlinear").to(device)
            weights = _hard_weights(expected, batch["source"])
            optimizer.zero_grad(set_to_none=True)
            logits = model(log_mel(audio))
            loss = _weighted_masked_bce(logits, expected, mask, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        candidate = _calibration_candidate(
            model, calibration_loader, device, epoch + 1, total / batches
        )
        candidates.append(candidate)
        print(json.dumps({key: value for key, value in candidate.items() if key != "state"}), flush=True)

    best = max(candidates, key=lambda row: tuple(row["selection_key"]))
    model.load_state_dict(best["state"])
    logits, expected, sources = _collect(model, development_loader, device, "nonlinear")
    probabilities = _probabilities(logits, best["calibration"])
    development = _metrics(probabilities, expected, best["threshold"], sources)
    acceptance = _acceptance(development)

    args.output.mkdir(parents=True, exist_ok=True)
    artifact = args.output / "model.pt"
    if acceptance["accepted"]:
        torch.save(
            {
                "schema": 1,
                "model": best["state"],
                "manifest": model.manifest(),
                "calibration": best["calibration"],
                "threshold": best["threshold"],
                "source_checkpoint_sha256": source_sha256,
            },
            artifact,
        )

    history = []
    for candidate in candidates:
        history.append({key: value for key, value in candidate.items() if key != "state"})
    report = {
        "schema": 1,
        "status": "development-accepted-not-promoted" if acceptance["accepted"] else "development-rejected",
        "scope": "bounded hard-example fine-tune of the Wet-only standalone nonlinear presence expert",
        "device": str(device),
        "model": model.manifest(),
        "training": {
            "selected_epoch": best["epoch"],
            "completed_epochs": args.epochs,
            "elapsed_seconds": time.perf_counter() - started,
            "weight_rules": WEIGHT_RULES,
            "history": history,
        },
        "frozen_gates": FROZEN_GATES,
        "calibration": {
            "parameters": best["calibration"],
            "threshold": best["threshold"],
            "metrics": best["metrics"],
            "acceptance": best["acceptance"],
        },
        "development": development,
        "acceptance": acceptance,
        "inventory": inventory(Path.cwd()),
        "boundaries": {
            "order_labels_used": False,
            "controls_used": False,
            "locked_final_opened": False,
            "standalone_nonlinear_head_only": True,
            "amp_head_trained": False,
            "ambience_head_trained": False,
            "mixed_chain_context_is_immutable_input_only": True,
            "development_opened_after_selection_only": True,
        },
        "source": {"path": str(args.source.resolve()), "sha256": source_sha256},
        "artifact": (
            {
                "path": str(artifact.resolve()),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
            if artifact.exists()
            else None
        ),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
