"""Fine-tune one order-independent Blind family expert from a shared checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .blind2 import LABELS, BlindFamilyPresence, log_mel
from .blind_data2 import BlindPresenceData, inventory
from .train_blind2 import (
    SEED,
    _device,
    _fit_calibration,
    _loader,
    _masked_bce,
    _probabilities,
    _thresholds,
)


FAMILIES = (*LABELS, "any")


def _family_target(value: torch.Tensor, family: str) -> torch.Tensor:
    if family == "any":
        return value.amax(dim=1, keepdim=True)
    index = LABELS.index(family)
    return value[:, index : index + 1]


def _weighted_masked_bce(
    logits: torch.Tensor,
    expected: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != expected.shape or mask.shape != expected.shape or weights.shape != expected.shape:
        raise ValueError("weighted BCE tensors must have matching shapes")
    elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, expected, reduction="none"
    )
    effective = mask * weights
    denominator = effective.sum()
    if float(denominator.detach().cpu()) <= 0.0:
        raise ValueError("weighted target mask must supervise at least one label")
    return (elementwise * effective).sum() / denominator


def _collect(
    model: BlindFamilyPresence,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    family: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    logits, targets, sources = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits.append(model(log_mel(batch["audio"].to(device))).cpu().numpy())
            targets.append(_family_target(batch["target"], family).numpy())
            sources.extend(batch["source"])
    return np.concatenate(logits), np.concatenate(targets), sources


def _binary_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    sources: list[str],
) -> dict:
    expected = targets[:, 0].astype(bool)
    predicted = probabilities[:, 0] >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected, predicted, average="binary", zero_division=0
    )
    source_array = np.asarray(sources, dtype=str)
    masks = {
        "egfxset-real": source_array == "egfxset-real",
        "dafx-random-position-presence": source_array == "dafx-random-position-presence",
        "synthetic-product-render": np.char.startswith(
            source_array, "synthetic-product-render:"
        ),
    }
    strata = {}
    for name, mask in masks.items():
        if not np.any(mask):
            continue
        stratum_expected = expected[mask]
        stratum_predicted = predicted[mask]
        row_precision, row_recall, row_f1, _ = precision_recall_fscore_support(
            stratum_expected, stratum_predicted, average="binary", zero_division=0
        )
        strata[name] = {
            "examples": int(np.sum(mask)),
            "positive_examples": int(np.sum(stratum_expected)),
            "precision": float(row_precision),
            "recall": float(row_recall),
            "f1": float(row_f1),
        }
    negative = ~expected
    return {
        "examples": len(expected),
        "positive_examples": int(np.sum(expected)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "negative_false_positive_rate": float(np.mean(predicted[negative])) if np.any(negative) else None,
        "strata": strata,
    }


def _initialized_model(checkpoint: dict, family: str) -> BlindFamilyPresence:
    if checkpoint["manifest"]["architecture"] != "compact-audio-resnet18":
        raise ValueError("family expert initialization requires a compact Blind checkpoint")
    family_index = None if family == "any" else LABELS.index(family)
    source = checkpoint["model"]
    model = BlindFamilyPresence(family)
    state = model.state_dict()
    for name in state:
        if name == "head.weight":
            state[name] = (
                source[name].mean(dim=0, keepdim=True).clone()
                if family_index is None
                else source[name][family_index : family_index + 1].clone()
            )
        elif name == "head.bias":
            state[name] = (
                source[name].mean(dim=0, keepdim=True).clone()
                if family_index is None
                else source[name][family_index : family_index + 1].clone()
            )
        else:
            state[name] = source[name].clone()
    model.load_state_dict(state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--any-hard-family", choices=LABELS)
    parser.add_argument("--any-hard-weight", type=float, default=2.0)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.any_hard_family is not None and args.family != "any":
        raise ValueError("--any-hard-family requires --family any")
    if args.any_hard_weight < 1.0:
        raise ValueError("--any-hard-weight must be at least one")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    target_device = _device(args.device)
    source_checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    model = _initialized_model(source_checkpoint, args.family).to(target_device)
    fit = BlindPresenceData(Path.cwd(), "fit", 2_400, SEED + 11)
    calibration_data = BlindPresenceData(Path.cwd(), "calibration", 480, SEED + 23)
    development_data = BlindPresenceData(Path.cwd(), "development", 640, SEED + 37)
    fit_loader = _loader(fit, 16, True)
    calibration_loader = _loader(calibration_data, 16, False)
    development_loader = _loader(development_data, 16, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    history = []
    best = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        fit.set_epoch(epoch)
        model.train()
        total = 0.0
        batches = 0
        for batch in fit_loader:
            audio = batch["audio"].to(target_device)
            expected = _family_target(batch["target"], args.family).to(target_device)
            mask = _family_target(batch["target_mask"], args.family).to(target_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(log_mel(audio))
            if args.any_hard_family is None:
                loss = _masked_bce(logits, expected, mask)
            else:
                hard_index = LABELS.index(args.any_hard_family)
                hard_positive = batch["target"][:, hard_index : hard_index + 1].to(
                    target_device
                )
                weights = 1.0 + hard_positive * (args.any_hard_weight - 1.0)
                loss = _weighted_masked_bce(logits, expected, mask, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        logits, expected, sources = _collect(
            model, calibration_loader, target_device, args.family
        )
        calibration = _fit_calibration(logits, expected)
        probabilities = _probabilities(logits, calibration)
        threshold = _thresholds(probabilities, expected)[0]
        metrics = _binary_metrics(probabilities, expected, threshold, sources)
        row = {"epoch": epoch + 1, "loss": total / batches, "calibration": metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        score = metrics["f1"]
        if best is None or score > best["score"] + 1.0e-4:
            best = {
                "score": score,
                "epoch": epoch + 1,
                "state": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
                "calibration": calibration,
                "threshold": threshold,
            }
            stale = 0
        else:
            stale += 1
        if stale >= 3:
            break
    assert best is not None
    model.load_state_dict(best["state"])
    logits, expected, sources = _collect(model, development_loader, target_device, args.family)
    probabilities = _probabilities(logits, best["calibration"])
    development = _binary_metrics(probabilities, expected, best["threshold"], sources)
    acceptance = {
        "accepted": (
            development["f1"] >= 0.80
            and development["recall"] >= 0.80
            and development["negative_false_positive_rate"] <= 0.05
        ),
        "gates": {
            "f1": development["f1"] >= 0.80,
            "recall": development["recall"] >= 0.80,
            "negative_false_positive_rate": development["negative_false_positive_rate"] <= 0.05,
        },
    }
    artifact = args.output / "model.pt"
    torch.save(
        {
            "schema": 1,
            "model": best["state"],
            "manifest": model.manifest(),
            "calibration": best["calibration"],
            "threshold": best["threshold"],
            "source_checkpoint_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        },
        artifact,
    )
    report = {
        "schema": 1,
        "status": "development-accepted-not-promoted" if acceptance["accepted"] else "development-rejected",
        "scope": f"Wet-only {args.family} presence expert; no order and no controls",
        "device": str(target_device),
        "model": model.manifest(),
        "training": {
            "selected_epoch": best["epoch"],
            "completed_epochs": len(history),
            "elapsed_seconds": time.perf_counter() - started,
            "history": history,
            "any_hard_family": args.any_hard_family,
            "any_hard_weight": args.any_hard_weight if args.any_hard_family else None,
        },
        "calibration": best["calibration"],
        "threshold": best["threshold"],
        "development": development,
        "acceptance": acceptance,
        "inventory": inventory(Path.cwd()),
        "boundaries": {
            "order_labels_used": False,
            "controls_used": False,
            "locked_final_opened": False,
            "standalone_family_expert_only": True,
        },
        "artifact": {
            "path": str(artifact.resolve()),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
