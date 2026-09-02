#!/usr/bin/env python3
"""Train product-only Wet family admission and per-effect knob estimators."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from .ambience2 import AmbiencePairsV2
from .product2 import ProductPairsV2, _condition_clean
from .product_data import Clean, _read, discover_clean


SEED = 20260911
RATE = 48_000
FAMILIES = ("identity", "nonlinear", "dynamics", "echo", "ambience")
CONTROL_WIDTHS = {"nonlinear": 5, "dynamics": 5, "echo": 3, "ambience": 3}
ANALYSIS_FRAMES = 48_000


def features(audio: np.ndarray) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float64)
    if value.ndim != 1 or len(value) < ANALYSIS_FRAMES or not np.isfinite(value).all():
        raise ValueError("wet admission expects at least one finite mono second")
    value = value[-ANALYSIS_FRAMES:]
    absolute = np.abs(value)
    rms = math.sqrt(float(np.mean(value * value)) + 1.0e-12)
    peak = float(np.max(absolute))
    difference = np.diff(value, prepend=value[0])
    frame = value[: (len(value) // 480) * 480].reshape(-1, 480)
    frame_rms = np.sqrt(np.mean(frame * frame, axis=1) + 1.0e-12)
    windowed = value * np.hanning(len(value))
    magnitude = np.abs(rfft(windowed)) + 1.0e-12
    power = magnitude * magnitude
    frequencies = rfftfreq(len(value), 1.0 / RATE)
    total_power = float(np.sum(power)) + 1.0e-12
    result = [
        math.log10(rms + 1.0e-8),
        math.log10(peak + 1.0e-8),
        peak / max(rms, 1.0e-8),
        float(np.mean(absolute)),
        float(np.std(value)),
        float(np.mean(np.abs(difference))) / max(rms, 1.0e-8),
        math.sqrt(float(np.mean(difference * difference)) + 1.0e-12) / max(rms, 1.0e-8),
        float(np.mean(np.signbit(value[1:]) != np.signbit(value[:-1]))),
        *np.quantile(absolute, (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)).tolist(),
        *np.quantile(frame_rms, (0.1, 0.25, 0.5, 0.75, 0.9)).tolist(),
        float(np.std(frame_rms)) / max(float(np.mean(frame_rms)), 1.0e-8),
    ]
    edges = np.geomspace(40.0, 20_000.0, 17)
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        band = (frequencies >= low) & (frequencies < high)
        result.append(math.log10(float(np.sum(power[band])) / total_power + 1.0e-10))
    centroid = float(np.sum(frequencies * power) / total_power)
    cumulative = np.cumsum(power)
    rolloff = float(frequencies[min(len(frequencies) - 1, int(np.searchsorted(cumulative, 0.85 * cumulative[-1])))])
    flatness = float(np.exp(np.mean(np.log(magnitude))) / np.mean(magnitude))
    result.extend((centroid / 24_000.0, rolloff / 24_000.0, flatness))
    # Echo and room evidence: normalized autocorrelation over 40-650 ms.
    fft_length = 1 << (2 * len(value) - 1).bit_length()
    spectrum = rfft(value, fft_length)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), fft_length)[: len(value)]
    correlation /= max(float(correlation[0]), 1.0e-12)
    region = correlation[round(0.04 * RATE) : round(0.65 * RATE) + 1]
    peak_index = int(np.argmax(np.abs(region)))
    result.extend((float(region[peak_index]), peak_index / max(len(region) - 1, 1), float(np.mean(np.abs(region)))))
    output = np.asarray(result, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("wet admission emitted non-finite features")
    return output


class _IdentityRows:
    def __init__(self, workspace: Path, split: str, samples: int, seed: int) -> None:
        buckets: dict[str, list[Clean]] = defaultdict(list)
        for row in discover_clean(workspace):
            if row.split == split:
                buckets[row.source_id].append(row)
        self.buckets = {name: tuple(rows) for name, rows in sorted(buckets.items())}
        self.source_ids = tuple(self.buckets)
        self.samples = samples
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict:
        source = self.source_ids[index % len(self.source_ids)]
        rows = self.buckets[source]
        cycle = index // len(self.source_ids)
        selected = rows[(cycle * 104729 + self.seed) % len(rows)]
        rng = random.Random(self.seed + index * 104729)
        clean = _read(selected, ANALYSIS_FRAMES, rng.getrandbits(63))
        clean, _ = _condition_clean(clean, rng)
        return {"wet": clean, "source_id": source, "controls": np.zeros(0, dtype=np.float32)}


def _dataset(workspace: Path, family: str, split: str, samples: int, seed: int):
    if family == "identity":
        return _IdentityRows(workspace, split, samples, seed)
    if family == "ambience":
        return AmbiencePairsV2(
            workspace, split, samples, 16384, seed, include_late_base=False
        )
    return ProductPairsV2(workspace, family, split, samples, ANALYSIS_FRAMES, seed)


def _rows(workspace: Path, split: str, samples_per_family: int, seed: int) -> list[dict]:
    result = []
    for family_index, family in enumerate(FAMILIES):
        dataset = _dataset(workspace, family, split, samples_per_family, seed + family_index * 1009)
        for index in range(len(dataset)):
            row = dataset[index]
            wet = row["wet"].numpy() if hasattr(row["wet"], "numpy") else row["wet"]
            controls = row["controls"].numpy() if hasattr(row["controls"], "numpy") else row["controls"]
            if family in CONTROL_WIDTHS:
                controls = np.asarray(controls)[: CONTROL_WIDTHS[family]]
            result.append({
                "family": family,
                "features": features(wet),
                "controls": np.asarray(controls, dtype=np.float32),
                "source_id": row["source_id"],
            })
    return result


def _matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return np.stack([row["features"] for row in rows]), np.asarray([row["family"] for row in rows])


def _family_thresholds(model: ExtraTreesClassifier, rows: list[dict]) -> dict[str, float]:
    matrix, expected = _matrix(rows)
    probabilities = model.predict_proba(matrix)
    predicted = model.classes_[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
    thresholds = {}
    for family in FAMILIES:
        wrong = confidence[(predicted == family) & (expected != family)]
        thresholds[family] = min(1.0, float(np.max(wrong) + 1.0e-6)) if len(wrong) else 0.50
    return thresholds


def _tree_predictions(model: ExtraTreesRegressor, matrix: np.ndarray) -> np.ndarray:
    return np.stack([tree.predict(matrix) for tree in model.estimators_])


def _knob_threshold(model: ExtraTreesRegressor, rows: list[dict]) -> float:
    matrix = np.stack([row["features"] for row in rows])
    expected = np.stack([row["controls"] for row in rows])
    trees = _tree_predictions(model, matrix)
    prediction = trees.mean(axis=0)
    uncertainty = trees.std(axis=0).max(axis=1)
    bad = np.max(np.abs(prediction - expected), axis=1) > 0.25
    return (
        max(0.0, float(np.min(uncertainty[bad]) - 1.0e-7))
        if np.any(bad)
        else float(np.max(uncertainty) + 1.0e-6)
    )


def _evaluate(
    family_model: ExtraTreesClassifier,
    knob_models: dict[str, ExtraTreesRegressor],
    family_thresholds: dict[str, float],
    knob_thresholds: dict[str, float],
    rows: list[dict],
) -> dict:
    matrix, expected = _matrix(rows)
    probabilities = family_model.predict_proba(matrix)
    indices = np.argmax(probabilities, axis=1)
    predicted = family_model.classes_[indices]
    confidence = probabilities[np.arange(len(rows)), indices]
    accepted = np.asarray([
        confidence[index] >= family_thresholds[str(predicted[index])] for index in range(len(rows))
    ])
    family_report = {
        "examples": len(rows),
        "raw_accuracy": float(np.mean(predicted == expected)),
        "accepted_coverage": float(np.mean(accepted)),
        "accepted_accuracy": float(np.mean(predicted[accepted] == expected[accepted])) if np.any(accepted) else 0.0,
        "per_family": {},
        "confusion": {
            true: {guess: int(np.sum((expected == true) & (predicted == guess))) for guess in FAMILIES}
            for true in FAMILIES
        },
    }
    for family in FAMILIES:
        mask = expected == family
        family_report["per_family"][family] = {
            "examples": int(np.sum(mask)),
            "raw_accuracy": float(np.mean(predicted[mask] == family)),
            "accepted_coverage": float(np.mean(accepted[mask])),
            "accepted_accuracy": float(np.mean(predicted[mask & accepted] == family)) if np.any(mask & accepted) else 0.0,
        }
    knob_report = {}
    for family, model in knob_models.items():
        selected = [row for row in rows if row["family"] == family]
        values = np.stack([row["features"] for row in selected])
        expected_controls = np.stack([row["controls"] for row in selected])
        trees = _tree_predictions(model, values)
        prediction = np.clip(trees.mean(axis=0), 0.0, 1.0)
        uncertainty = trees.std(axis=0).max(axis=1)
        admitted = uncertainty <= knob_thresholds[family]
        error = np.abs(prediction - expected_controls)
        admitted_error = error[admitted]
        knob_report[family] = {
            "examples": len(selected),
            "accepted_coverage": float(np.mean(admitted)),
            "mae": float(np.mean(admitted_error)) if len(admitted_error) else None,
            "p95": float(np.quantile(admitted_error, 0.95)) if len(admitted_error) else None,
            "all_mae": float(np.mean(error)),
            "all_p95": float(np.quantile(error, 0.95)),
        }
    family_report["gates"] = {
        "raw_accuracy": family_report["raw_accuracy"] >= 0.75,
        "accepted_accuracy": family_report["accepted_accuracy"] >= 0.95,
        "accepted_coverage": family_report["accepted_coverage"] >= 0.35,
        "each_family_raw_accuracy": all(
            row["raw_accuracy"] >= 0.55 for row in family_report["per_family"].values()
        ),
    }
    family_report["accepted"] = all(family_report["gates"].values())
    for row in knob_report.values():
        row["accepted"] = bool(
            row["accepted_coverage"] >= 0.25
            and row["mae"] is not None and row["mae"] <= 0.15
            and row["p95"] is not None and row["p95"] <= 0.35
        )
    return {"family": family_report, "knobs": knob_report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/product2/admission"))
    parser.add_argument("--fit-per-family", type=int, default=180)
    parser.add_argument("--calibration-per-family", type=int, default=60)
    parser.add_argument("--development-per-family", type=int, default=80)
    parser.add_argument("--trees", type=int, default=240)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.fit_per_family = 30
        args.calibration_per_family = 12
        args.development_per_family = 16
        args.trees = 80
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    fit = _rows(workspace, "fit", args.fit_per_family, SEED + 1)
    calibration = _rows(workspace, "calibration", args.calibration_per_family, SEED + 2)
    development = _rows(workspace, "development", args.development_per_family, SEED + 3)
    fit_matrix, fit_labels = _matrix(fit)
    family_model = ExtraTreesClassifier(
        n_estimators=args.trees,
        min_samples_leaf=2,
        class_weight="balanced",
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    ).fit(fit_matrix, fit_labels)
    knob_models = {}
    knob_thresholds = {}
    for family in CONTROL_WIDTHS:
        selected_fit = [row for row in fit if row["family"] == family]
        model = ExtraTreesRegressor(
            n_estimators=args.trees,
            min_samples_leaf=2,
            max_features=0.75,
            random_state=SEED + FAMILIES.index(family),
            n_jobs=-1,
        ).fit(
            np.stack([row["features"] for row in selected_fit]),
            np.stack([row["controls"] for row in selected_fit]),
        )
        knob_models[family] = model
        knob_thresholds[family] = _knob_threshold(
            model, [row for row in calibration if row["family"] == family]
        )
    family_thresholds = _family_thresholds(family_model, calibration)
    evaluation = _evaluate(
        family_model, knob_models, family_thresholds, knob_thresholds, development
    )
    output.mkdir(parents=True, exist_ok=True)
    family_path = output / "family.joblib"
    knobs_path = output / "knobs.joblib"
    joblib.dump({
        "schema": 1,
        "model": family_model,
        "thresholds": family_thresholds,
        "families": FAMILIES,
        "feature_width": fit_matrix.shape[1],
    }, family_path)
    joblib.dump({
        "schema": 1,
        "models": knob_models,
        "thresholds": knob_thresholds,
        "control_widths": CONTROL_WIDTHS,
        "feature_width": fit_matrix.shape[1],
    }, knobs_path)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "schema": 1,
        "status": "development-gates-passed-not-promoted" if (
            not args.quick
            and evaluation["family"]["accepted"]
            and all(row["accepted"] for row in evaluation["knobs"].values())
        ) else "diagnostic-not-promoted",
        "quick": args.quick,
        "scope": "single-effect unknown Wet admission only",
        "artifacts": {
            "family": {"path": str(family_path), "sha256": digest(family_path)},
            "knobs": {"path": str(knobs_path), "sha256": digest(knobs_path)},
        },
        "training": {
            "fit_per_family": args.fit_per_family,
            "calibration_per_family": args.calibration_per_family,
            "development_per_family": args.development_per_family,
            "trees": args.trees,
            "feature_width": fit_matrix.shape[1],
        },
        "development": evaluation,
        "provenance": {
            "product_sources_only": True,
            "research_data_used_for_gradients_or_selection": False,
            "physical_audio_devices_used": False,
            "locked_final_audio_opened": False,
            "source_audio_modified": False,
        },
        "boundaries": {
            "family_and_knobs_are_separate_artifacts": True,
            "order_model_included": False,
            "inverse_weights_receive_order": False,
            "multi_effect_graph_supported": False,
            "unknown_or_low_confidence_action": "abstain",
        },
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
