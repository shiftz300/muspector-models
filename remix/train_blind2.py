"""Train and calibrate the product-safe Wet-only multi-label Blind v2 model."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .blind2 import LABELS, BlindPresence, BlindPresenceMultiAxis, log_mel
from .blind_data2 import BlindPresenceData, inventory


SEED = 20260902


def _masked_bce(logits: torch.Tensor, expected: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != expected.shape or mask.shape != expected.shape:
        raise ValueError("masked BCE tensors must have matching shapes")
    if not torch.all((mask == 0.0) | (mask == 1.0)):
        raise ValueError("target mask must be binary")
    denominator = mask.sum()
    if float(denominator.detach().cpu()) <= 0.0:
        raise ValueError("target mask must supervise at least one label")
    elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, expected, reduction="none"
    )
    return (elementwise * mask).sum() / denominator


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(requested)


def _loader(dataset: BlindPresenceData, batch: int, shuffle: bool) -> torch.utils.data.DataLoader:
    generator = torch.Generator().manual_seed(SEED + dataset.seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _collect(model: BlindPresence, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    logits, targets, sources = [], [], []
    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            logits.append(model(log_mel(audio)).cpu().numpy())
            targets.append(batch["target"].numpy())
            sources.extend(batch["source"])
    return np.concatenate(logits), np.concatenate(targets), sources


def _fit_calibration(logits: np.ndarray, targets: np.ndarray) -> list[dict]:
    result = []
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("calibration logits and targets must have matching matrices")
    for index in range(logits.shape[1]):
        model = LogisticRegression(C=1.0e6, solver="lbfgs", random_state=SEED)
        model.fit(logits[:, index, None], targets[:, index].astype(np.int64))
        result.append({"scale": float(model.coef_[0, 0]), "bias": float(model.intercept_[0])})
    return result


def _probabilities(logits: np.ndarray, calibration: list[dict]) -> np.ndarray:
    scale = np.asarray([row["scale"] for row in calibration])
    bias = np.asarray([row["bias"] for row in calibration])
    values = np.clip(logits * scale + bias, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def _thresholds(probabilities: np.ndarray, targets: np.ndarray) -> list[float]:
    expected_all = targets.astype(bool)
    clean = ~expected_all.any(axis=1)
    maximum_clean_false_positives = int(np.floor(0.05 * int(np.sum(clean)) + 1.0e-12))
    choices = []
    if probabilities.shape != targets.shape or probabilities.ndim != 2:
        raise ValueError("threshold probabilities and targets must have matching matrices")
    for index in range(probabilities.shape[1]):
        expected = expected_all[:, index]
        candidates = []
        for threshold in np.linspace(0.05, 0.995, 190):
            predicted = probabilities[:, index] >= threshold
            false_positive = float(np.mean(predicted[~expected])) if np.any(~expected) else 0.0
            if false_positive > 0.05:
                continue
            true_positive = int(np.sum(predicted & expected))
            false_positive_count = int(np.sum(predicted & ~expected))
            false_negative = int(np.sum(~predicted & expected))
            recall = true_positive / max(1, true_positive + false_negative)
            f1 = 2 * true_positive / max(1, 2 * true_positive + false_positive_count + false_negative)
            candidates.append({
                "threshold": float(threshold),
                "clean": predicted[clean],
                "tp": true_positive,
                "fp": false_positive_count,
                "fn": false_negative,
                "recall": recall,
                "f1": f1,
            })
        if not candidates:
            label = LABELS[index] if index < len(LABELS) else f"label-{index}"
            raise ValueError(f"no false-positive-safe threshold for {label}")
        # Keep the search bounded while retaining both low-recall-cost and
        # high-precision ends of every per-label frontier.
        count = min(18, len(candidates))
        indices = set(np.linspace(0, len(candidates) - 1, count).round().astype(int).tolist())
        indices.add(max(range(len(candidates)), key=lambda row: candidates[row]["f1"]))
        choices.append([candidates[row] for row in sorted(indices)])

    best = None
    for rows in itertools.product(*choices):
        if np.any(clean):
            joint_clean = np.logical_or.reduce([row["clean"] for row in rows])
            if int(np.sum(joint_clean)) > maximum_clean_false_positives:
                continue
        true_positive = sum(row["tp"] for row in rows)
        false_positive = sum(row["fp"] for row in rows)
        false_negative = sum(row["fn"] for row in rows)
        micro_f1 = 2 * true_positive / max(
            1, 2 * true_positive + false_positive + false_negative
        )
        objective = (
            min(row["recall"] for row in rows),
            float(np.mean([row["f1"] for row in rows])),
            micro_f1,
            -sum(row["threshold"] for row in rows),
        )
        if best is None or objective > best[0]:
            best = (objective, rows)
    if best is None:
        raise ValueError("no threshold vector satisfies the joint Clean false-positive gate")
    return [float(row["threshold"]) for row in best[1]]


def _metrics(probabilities: np.ndarray, targets: np.ndarray, thresholds: list[float], sources: list[str]) -> dict:
    expected = targets.astype(bool)
    predicted = probabilities >= np.asarray(thresholds)
    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, average=None, zero_division=0
    )
    clean = ~expected.any(axis=1)
    report = {
        "examples": len(expected),
        "micro_f1": float(f1_score(expected, predicted, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
        "exact_match": float(np.mean(np.all(expected == predicted, axis=1))),
        "clean_examples": int(np.sum(clean)),
        "clean_false_positive_rate": float(np.mean(predicted[clean].any(axis=1))) if np.any(clean) else None,
        "per_label": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "positive_examples": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
    }
    strata = {}
    source_array = np.asarray(sources, dtype=str)
    primary_strata = {
        "egfxset-real": source_array == "egfxset-real",
        "dafx-random-position-presence": source_array == "dafx-random-position-presence",
        "synthetic-product-render": np.char.startswith(source_array, "synthetic-product-render:"),
    }
    nonlinear_parts = [
        source.split(":") if source.startswith("synthetic-product-render:nonlinear:") else None
        for source in sources
    ]
    diagnostic_strata = {}
    for index, axis in (
        (2, "implementation"),
        (3, "drive"),
        (4, "residual"),
        (5, "chain-count"),
        (6, "clean-source"),
    ):
        values = sorted({parts[index] for parts in nonlinear_parts if parts is not None})
        for value in values:
            diagnostic_strata[f"synthetic-nonlinear-{axis}:{value}"] = np.asarray([
                parts is not None and parts[index] == value for parts in nonlinear_parts
            ])
    for name, mask in {**primary_strata, **diagnostic_strata}.items():
        if np.any(mask):
            stratum_precision, stratum_recall, stratum_f1, stratum_support = (
                precision_recall_fscore_support(
                    expected[mask], predicted[mask], average=None, zero_division=0
                )
            )
            strata[name] = {
                "examples": int(np.sum(mask)),
                "micro_f1": float(f1_score(expected[mask], predicted[mask], average="micro", zero_division=0)),
                "macro_f1": float(f1_score(expected[mask], predicted[mask], average="macro", zero_division=0)),
                "exact_match": float(np.mean(np.all(expected[mask] == predicted[mask], axis=1))),
                "per_label": {
                    label: {
                        "precision": float(stratum_precision[index]),
                        "recall": float(stratum_recall[index]),
                        "f1": float(stratum_f1[index]),
                        "positive_examples": int(stratum_support[index]),
                    }
                    for index, label in enumerate(LABELS)
                },
            }
    report["strata"] = strata
    return report


def _acceptance(metrics: dict) -> dict:
    gates = {
        "micro_f1": metrics["micro_f1"] >= 0.80,
        "macro_f1": metrics["macro_f1"] >= 0.80,
        "each_label_recall": all(row["recall"] >= 0.80 for row in metrics["per_label"].values()),
        "clean_false_positive_rate": metrics["clean_false_positive_rate"] is not None
        and metrics["clean_false_positive_rate"] <= 0.05,
        "real_hardware_anchor_macro_f1": metrics["strata"].get("egfxset-real", {}).get("macro_f1", 0.0) >= 0.70,
        "synthetic_chain_macro_f1": metrics["strata"].get("synthetic-product-render", {}).get("macro_f1", 0.0) >= 0.80,
        "random_position_chain_macro_f1": metrics["strata"].get(
            "dafx-random-position-presence", {}
        ).get("macro_f1", 0.0) >= 0.80,
    }
    return {"accepted": all(gates.values()), "gates": gates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--architecture", choices=("compact", "multiaxis"), default="compact")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(
        "runs/foundation/product2/blind2-quick" if args.quick else "runs/foundation/product2/blind2"
    )
    output.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    target = _device(args.device)
    fit_count, calibration_count, development_count = (
        (384, 128, 160) if args.quick else (2_400, 480, 640)
    )
    epochs = 3 if args.quick else 12
    batch_size = 16
    fit = BlindPresenceData(Path.cwd(), "fit", fit_count, SEED + 11)
    calibration_data = BlindPresenceData(Path.cwd(), "calibration", calibration_count, SEED + 23)
    development_data = BlindPresenceData(Path.cwd(), "development", development_count, SEED + 37)
    fit_loader = _loader(fit, batch_size, True)
    calibration_loader = _loader(calibration_data, batch_size, False)
    development_loader = _loader(development_data, batch_size, False)
    model_class = BlindPresence if args.architecture == "compact" else BlindPresenceMultiAxis
    model = model_class().to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    history = []
    best = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        fit.set_epoch(epoch)
        model.train()
        total = 0.0
        batches = 0
        for batch in fit_loader:
            audio = batch["audio"].to(target)
            expected = batch["target"].to(target)
            target_mask = batch["target_mask"].to(target)
            optimizer.zero_grad(set_to_none=True)
            logits = model(log_mel(audio))
            loss = _masked_bce(logits, expected, target_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        calibration_logits, calibration_targets, calibration_sources = _collect(model, calibration_loader, target)
        calibrators = _fit_calibration(calibration_logits, calibration_targets)
        calibration_probabilities = _probabilities(calibration_logits, calibrators)
        thresholds = _thresholds(calibration_probabilities, calibration_targets)
        calibration_metrics = _metrics(
            calibration_probabilities, calibration_targets, thresholds, calibration_sources
        )
        score = calibration_metrics["macro_f1"]
        row = {"epoch": epoch + 1, "loss": total / batches, "calibration": calibration_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if best is None or score > best["score"] + 1.0e-4:
            best = {
                "score": score,
                "epoch": epoch + 1,
                "state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                "calibration": calibrators,
                "thresholds": thresholds,
            }
            stale = 0
        else:
            stale += 1
        if not args.quick and stale >= 3:
            break
    assert best is not None
    model.load_state_dict(best["state"])
    development_logits, development_targets, development_sources = _collect(model, development_loader, target)
    development_probabilities = _probabilities(development_logits, best["calibration"])
    development_metrics = _metrics(
        development_probabilities, development_targets, best["thresholds"], development_sources
    )
    acceptance = _acceptance(development_metrics)
    checkpoint = output / "model.pt"
    torch.save(
        {
            "schema": 1,
            "model": best["state"],
            "manifest": model.manifest(),
            "calibration": best["calibration"],
            "thresholds": best["thresholds"],
        },
        checkpoint,
    )
    report = {
        "schema": 1,
        "status": "development-accepted-not-promoted" if acceptance["accepted"] else "development-rejected",
        "quick": args.quick,
        "scope": "Wet-only multi-label effect presence; no order and no controls",
        "device": str(target),
        "device_selection_evidence": {
            "benchmark_batch": 8,
            "cpu_seconds": 0.15384601679979823,
            "mps_seconds": 0.04963484159961808,
            "mps_speedup": 3.0998,
        },
        "model": model.manifest(),
        "training": {
            "fit_examples_per_epoch": fit_count,
            "calibration_examples": calibration_count,
            "development_examples": development_count,
            "maximum_epochs": epochs,
            "selected_epoch": best["epoch"],
            "completed_epochs": len(history),
            "elapsed_seconds": time.perf_counter() - started,
            "history": history,
            "positive_only_source": "multimodal-electric-guitar-data",
            "positive_only_labels": "repository-added effects only; all absent labels masked unknown",
            "random_position_presence_source": "dafx25-guitar-effects-chains",
        },
        "calibration": best["calibration"],
        "thresholds": dict(zip(LABELS, best["thresholds"], strict=True)),
        "development": development_metrics,
        "acceptance": acceptance,
        "inventory": inventory(Path.cwd()),
        "boundaries": {
            "order_model_included": False,
            "order_labels_used": False,
            "controls_model_included": False,
            "inverse_weights_changed": False,
            "multi_effect_presence_supported": True,
            "low_confidence_runtime_action": "abstain",
        },
        "provenance": {
            "product_sources_only": True,
            "required_attribution": fit.authorization["required_attribution"],
            "egfxset_wet_use": "family-recognition single-effect anchor only",
            "multimodal_amplifier_use": "fit-only positive-label family-presence augmentation; never restoration Clean",
            "random_position_chain_use": "presence bits only; author order metadata remains isolated in the order package",
            "generated_wet_written": False,
            "locked_final_opened": False,
            "research_data_used_for_gradients_calibration_or_selection": False,
            "physical_audio_devices_used": False,
        },
        "artifact": {
            "path": str(checkpoint.resolve()),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
    }
    report_path = output / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
