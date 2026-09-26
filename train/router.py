#!/usr/bin/env python3
"""Train a gain-invariant RAT-vs-DFZ shadow router on frozen evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from remix.asrnn_effects import effect_files, read_effect_pair
from remix.identity import IdentityRuntime
from remix.packages import digest
from remix.router import FEATURES, encode


SEED = 20260901
GATES = {"recall": 0.80, "false_route": 0.05, "balanced_accuracy": 0.85, "input_mutations": 0}


def spread(paths: list[Path], count: int) -> list[Path]:
    indices = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[int(index)] for index in indices]


def rows(root: Path, split: str) -> list[tuple[str, Path]]:
    rat, dfz = effect_files(root, "rat", split), effect_files(root, "dfz", split)
    count = min(len(rat), len(dfz))
    return [("rat", path) for path in spread(rat, count)] + [("dfz", path) for path in spread(dfz, count)]


def encode_rows(runtime: IdentityRuntime, records: list[tuple[str, Path]]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    values, truth, audit = [], [], []
    for index, (device, path) in enumerate(records):
        source_hash = digest(path)
        dry, wet, _ = read_effect_pair(path, device)
        before = hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest()
        report = runtime.infer(wet, 48_000)
        values.append(encode(runtime, report, dry, wet, 48_000))
        truth.append(device == "rat")
        audit.append({
            "device": device, "path": path.as_posix(), "sha256": source_hash,
            "inputs_unchanged": before == hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest(),
        })
        if (index + 1) % 64 == 0 or index + 1 == len(records):
            print(json.dumps({"stage": "encode", "done": index + 1, "total": len(records)}), flush=True)
    return np.stack(values), np.asarray(truth, dtype=bool), audit


def partition(records: list[tuple[str, Path]]) -> np.ndarray:
    return np.asarray([
        int(hashlib.sha256(f"{device}:{path.name}".encode()).hexdigest()[:8], 16) % 5 == 0
        for device, path in records
    ])


def metrics(probability: np.ndarray, truth: np.ndarray, threshold: float) -> dict:
    predicted = probability >= threshold
    recall = float(predicted[truth].mean())
    false_route = float(predicted[~truth].mean())
    return {
        "examples": len(truth), "rat": int(truth.sum()), "dfz": int((~truth).sum()),
        "threshold": threshold, "recall": recall, "false_route": false_route,
        "balanced_accuracy": (recall + 1.0 - false_route) / 2.0,
    }


def choose_threshold(probability: np.ndarray, truth: np.ndarray) -> float:
    candidates = []
    for value in np.linspace(0.0, 1.0, 1001):
        report = metrics(probability, truth, float(value))
        if report["false_route"] <= GATES["false_route"]:
            candidates.append((report["recall"], value))
    best = max(row[0] for row in candidates)
    return float(max(value for recall, value in candidates if recall == best))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"router run already exists: {args.output}")
    runtime = IdentityRuntime(args.identity)
    training, validation = rows(args.corpus, "train"), rows(args.corpus, "eval")
    train_x, train_y, train_audit = encode_rows(runtime, training)
    valid_x, valid_y, valid_audit = encode_rows(runtime, validation)
    calibrate = partition(training)
    fit = ~calibrate
    candidates = {
        "linear": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=SEED)),
        "hist": HistGradientBoostingClassifier(learning_rate=0.06, max_iter=240, max_leaf_nodes=15, min_samples_leaf=12, l2_regularization=0.05, random_state=SEED),
        "trees": ExtraTreesClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=2),
    }
    screens = {}
    for name, candidate in candidates.items():
        candidate.fit(train_x[fit], train_y[fit])
        probability = candidate.predict_proba(train_x[calibrate])[:, 1]
        selected = choose_threshold(probability, train_y[calibrate])
        screens[name] = {"model": candidate, "threshold": selected, "metrics": metrics(probability, train_y[calibrate], selected)}
    selected_name, selected_row = max(
        screens.items(),
        key=lambda row: (row[1]["metrics"]["recall"], row[1]["metrics"]["balanced_accuracy"], row[0]),
    )
    model, selected = selected_row["model"], selected_row["threshold"]
    valid = metrics(model.predict_proba(valid_x)[:, 1], valid_y, selected)
    mutations = sum(not row["inputs_unchanged"] for row in train_audit + valid_audit)
    failures = []
    if valid["recall"] < GATES["recall"]: failures.append("recall")
    if valid["false_route"] > GATES["false_route"]: failures.append("false-route")
    if valid["balanced_accuracy"] < GATES["balanced_accuracy"]: failures.append("balanced-accuracy")
    if mutations: failures.append("input-mutation")
    runtime.assert_artifacts_unchanged()
    for row in train_audit + valid_audit:
        if digest(Path(row["path"])) != row["sha256"]:
            raise ValueError("router source changed during training")
    args.output.mkdir(parents=True)
    artifact = args.output / "router.joblib"
    joblib.dump({
        "schema": 1, "device": "rat", "features": FEATURES, "threshold": selected,
        "identity_package_sha256": runtime.package_hash, "model": model,
    }, artifact)
    fit_report = {
        "schema": 1, "status": "complete", "architecture": "independently level-standardized identity and paired transfer features",
        "features": list(FEATURES), "seed": SEED, "fit": int(fit.sum()), "calibrate": int(calibrate.sum()),
        "calibration": selected_row["metrics"], "selection": selected_name,
        "screens": {name: row["metrics"] for name, row in screens.items()},
        "source": "ASRNN official train split, evenly balanced RAT and DFZ",
        "license": "CC-BY-NC-4.0", "identity_package_sha256": runtime.package_hash,
        "analysis_gain_standardization": True, "source_audio_modified": False, "physical_audio_devices_used": False,
    }
    (args.output / "fit.json").write_text(json.dumps(fit_report, indent=2, sort_keys=True) + "\n")
    report = {
        "schema": 1, "status": "accepted-development" if not failures else "rejected", "accepted": not failures,
        "source": "ASRNN official eval split", "scope": "RAT-vs-DFZ development validation only",
        "gates": GATES, "metrics": valid, "failures": failures,
        "artifact": {"path": "router.joblib", "bytes": artifact.stat().st_size, "sha256": digest(artifact)},
        "input_mutations": mutations, "artifacts_unchanged": True, "analysis_gain_standardization": True,
        "source_audio_modified": False, "physical_audio_devices_used": False, "automatic_delivery": False,
        "automatic_normalization": False, "automatic_limiting": False, "lossy_reencoding": False,
    }
    (args.output / "valid.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": report["accepted"], "failures": failures, "metrics": valid}), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
