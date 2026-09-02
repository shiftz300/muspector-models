"""Extract, train, and evaluate the chain family-set gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch.utils.data import Subset

from .data import DAFxOrderDataset, SyntheticControlDataset, dry_sources
from .evaluate_order_search import evenly_spaced, make_domains
from .family import FamilyRuntime
from .gate import CLASSES, FEATURES, active, mask, vector
from .knobs import KnobRuntime
from .pedalboard_data import PedalboardControlDataset
from .spec import KINDS


MIN_COVERAGE = 0.35
MIN_ACCEPTED_EXACT = 0.98


def datasets(corpus: Path, source: str, samples: int, pairs: Path | None = None):
    if source.startswith("idmt-"):
        if pairs is None:
            raise ValueError("IDMT gate extraction requires --pairs")
        from .idmt import Dataset as IdmtDataset, TARGETS

        dataset = IdmtDataset(corpus, pairs, source.removeprefix("idmt-"), samples)
        return {
            f"idmt-{target}": Subset(
                dataset,
                [index for index, row in enumerate(dataset.records) if row["target"] == target],
            )
            for target in TARGETS
        }
    if source != "train":
        return make_domains(corpus, source, samples, -30.0)
    result = {"real": evenly_spaced(DAFxOrderDataset(corpus, "train"), samples * 2)}
    for renderer in ("reference", "alternate", "stress"):
        result[renderer] = SyntheticControlDataset(
            dry_sources(corpus, "train"), samples, seed=20261001,
            renderers=(renderer,), order_equivalence_db=-30.0,
        )
    result["pedalboard"] = PedalboardControlDataset(
        dry_sources(corpus, "train"), samples, seed=20261011,
        order_equivalence_db=-30.0,
    )
    return result


def truth(item: dict) -> list[str]:
    return [KINDS[int(value)] for value in item["topology"].tolist() if int(value) >= 0]


def extract(args) -> None:
    if args.output.exists():
        raise FileExistsError(f"gate features already exist: {args.output}")
    family = FamilyRuntime(args.family, args.manifest)
    knobs = KnobRuntime(args.bundle, args.evidence, torch.device("cpu"))
    features, labels, domains, indices, hashes = [], [], [], [], []
    for domain, dataset in datasets(args.corpus, args.source, args.samples, args.pairs).items():
        for index in range(len(dataset)):
            item = dataset[index]
            dry, wet = item["dry"].numpy(), item["wet"].numpy()
            before = hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest()
            family_report = family.infer(dry, wet)
            remix_report = knobs.infer(dry, wet, KINDS)
            features.append(vector(family_report, remix_report, dry, wet, knobs))
            labels.append(mask(truth(item)))
            domains.append(domain)
            indices.append(index)
            hashes.append(before)
            if before != hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest():
                raise ValueError("gate extraction mutated input audio")
            if (index + 1) % 16 == 0 or index + 1 == len(dataset):
                print(json.dumps({"domain": domain, "done": index + 1, "total": len(dataset)}), flush=True)
    family.assert_artifacts_unchanged()
    if hashlib.sha256(args.bundle.read_bytes()).hexdigest() != knobs.bundle_hash:
        raise ValueError("inverse bundle changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x=np.stack(features), y=np.asarray(labels, dtype=np.int64),
        domain=np.asarray(domains), index=np.asarray(indices), input_sha256=np.asarray(hashes),
        feature_names=np.asarray(FEATURES),
    )
    print(json.dumps({"rows": len(labels), "features": len(FEATURES), "output": str(args.output)}))


def _load(path: Path):
    with np.load(path) as value:
        if tuple(value["feature_names"].tolist()) != FEATURES:
            raise ValueError("gate feature cache contract changed")
        return {name: value[name].copy() for name in value.files}


def _combine(paths: list[Path]) -> dict:
    values = [_load(path) for path in paths]
    return {
        name: np.concatenate([value[name] for value in values])
        for name in ("x", "y", "domain", "index", "input_sha256")
    }


def _metrics(labels, predictions, confidence, thresholds, domains) -> dict:
    if isinstance(thresholds, dict):
        threshold = np.asarray([thresholds[int(value)] for value in predictions])
    else:
        threshold = float(thresholds)
    selected = confidence >= threshold
    correct = predictions == labels
    result = {
        "examples": int(len(labels)),
        "raw_exact": float(correct.mean()),
        "coverage": float(selected.mean()),
        "accepted": int(selected.sum()),
        "accepted_exact": float(correct[selected].mean()) if selected.any() else 0.0,
        "accepted_errors": int((selected & ~correct).sum()),
    }
    result["domains"] = {
        name: {
            "examples": int(masked.sum()),
            "raw_exact": float(correct[masked].mean()),
            "coverage": float(selected[masked].mean()),
            "accepted_exact": float(correct[masked & selected].mean()) if (masked & selected).any() else 0.0,
            "accepted_errors": int((masked & selected & ~correct).sum()),
        }
        for name in sorted(set(domains.tolist()))
        for masked in [domains == name]
    }
    return result


def _threshold(labels, predictions, confidence) -> float:
    wrong = confidence[predictions != labels]
    return 0.0 if not len(wrong) else float(np.nextafter(wrong.max(), np.inf))


def _thresholds(labels, predictions, confidence) -> dict[int, float]:
    result = {}
    for value in CLASSES:
        selected = predictions == value
        wrong = confidence[selected & (predictions != labels)]
        if len(wrong):
            result[value] = float(np.nextafter(wrong.max(), np.inf))
        elif selected.any():
            result[value] = 0.5
        else:
            result[value] = 1.0000001
    return result


def train(args) -> None:
    if args.output.exists():
        raise FileExistsError(f"gate run already exists: {args.output}")
    train_paths = [args.train, *args.train_extra]
    fit_paths = [args.fit, *args.fit_extra]
    development_paths = ([args.development] if args.development else []) + args.development_extra
    train_data, fit_data = _combine(train_paths), _combine(fit_paths)
    development_data = _combine(development_paths) if development_paths else None
    model = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=240, max_leaf_nodes=15,
        min_samples_leaf=12, l2_regularization=0.05, random_state=20260901,
    )
    model.fit(train_data["x"], train_data["y"])
    if tuple(map(int, model.classes_)) != CLASSES:
        raise ValueError("gate training did not observe every family set")
    probabilities = model.predict_proba(fit_data["x"])
    indices = probabilities.argmax(1)
    predictions = model.classes_[indices]
    confidence = probabilities[np.arange(len(indices)), indices]
    threshold_labels = fit_data["y"]
    threshold_predictions = predictions
    threshold_confidence = confidence
    development_values = None
    if development_data is not None:
        development_probabilities = model.predict_proba(development_data["x"])
        development_indices = development_probabilities.argmax(1)
        development_predictions = model.classes_[development_indices]
        development_confidence = development_probabilities[
            np.arange(len(development_indices)), development_indices
        ]
        threshold_labels = np.concatenate((threshold_labels, development_data["y"]))
        threshold_predictions = np.concatenate((threshold_predictions, development_predictions))
        threshold_confidence = np.concatenate((threshold_confidence, development_confidence))
        development_values = (development_predictions, development_confidence)
    thresholds = _thresholds(threshold_labels, threshold_predictions, threshold_confidence)
    metrics = _metrics(fit_data["y"], predictions, confidence, thresholds, fit_data["domain"])
    development_metrics = None
    if development_data is not None:
        development_metrics = _metrics(
            development_data["y"], development_values[0], development_values[1],
            thresholds, development_data["domain"],
        )
    failures = []
    if metrics["coverage"] < MIN_COVERAGE: failures.append("coverage")
    if metrics["accepted_exact"] < MIN_ACCEPTED_EXACT: failures.append("accepted-exact")
    if development_metrics is not None and development_metrics["coverage"] < MIN_COVERAGE:
        failures.append("development-coverage")
    if development_metrics is not None and development_metrics["accepted_exact"] < MIN_ACCEPTED_EXACT:
        failures.append("development-accepted-exact")
    args.output.mkdir(parents=True)
    artifact = args.output / "gate.joblib"
    joblib.dump({"schema": 1, "features": FEATURES, "classes": CLASSES, "thresholds": thresholds, "model": model}, artifact, compress=3)
    report = {
        "schema": 1, "accepted": not failures, "failures": failures,
        "thresholds": {str(name): value for name, value in thresholds.items()}, "minimum": {"coverage": MIN_COVERAGE, "accepted_exact": MIN_ACCEPTED_EXACT},
        "metrics": metrics, "development_metrics": development_metrics,
        "train_rows": int(len(train_data["y"])), "fit_rows": int(len(fit_data["y"])),
        "development_rows": int(len(development_data["y"])) if development_data is not None else 0,
        "train_sources": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in train_paths],
        "fit_sources": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in fit_paths],
        "development_sources": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in development_paths],
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "source_audio_modified": False, "physical_audio_devices_used": False,
    }
    (args.output / "fit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": report["accepted"], "failures": failures, "thresholds": thresholds, "metrics": metrics, "development_metrics": development_metrics}))
    if failures: raise SystemExit(2)


def evaluate(args) -> None:
    data = _load(args.features)
    payload = joblib.load(args.run / "gate.joblib")
    model = payload["model"]
    thresholds = (
        {int(name): float(value) for name, value in payload["thresholds"].items()}
        if "thresholds" in payload
        else {value: float(payload["threshold"]) for value in CLASSES}
    )
    probabilities = model.predict_proba(data["x"])
    indices = probabilities.argmax(1)
    predictions = model.classes_[indices]
    confidence = probabilities[np.arange(len(indices)), indices]
    metrics = _metrics(data["y"], predictions, confidence, thresholds, data["domain"])
    failures = []
    if metrics["coverage"] < MIN_COVERAGE: failures.append("coverage")
    if metrics["accepted_exact"] < MIN_ACCEPTED_EXACT: failures.append("accepted-exact")
    report = {
        "schema": 1, "accepted": not failures, "failures": failures,
        "thresholds": {str(name): value for name, value in thresholds.items()}, "minimum": {"coverage": MIN_COVERAGE, "accepted_exact": MIN_ACCEPTED_EXACT},
        "metrics": metrics, "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "artifact_sha256": hashlib.sha256((args.run / "gate.joblib").read_bytes()).hexdigest(),
        "source_audio_modified": False, "physical_audio_devices_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": report["accepted"], "failures": failures, "metrics": metrics}))
    if failures: raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    extraction = sub.add_parser("extract")
    extraction.add_argument("--source", choices=("train", "calibrate", "valid", "idmt-train", "idmt-calibrate", "idmt-valid"), required=True)
    extraction.add_argument("--corpus", type=Path, required=True)
    extraction.add_argument("--pairs", type=Path)
    extraction.add_argument("--family", type=Path, required=True)
    extraction.add_argument("--manifest", type=Path, required=True)
    extraction.add_argument("--bundle", type=Path, required=True)
    extraction.add_argument("--evidence", type=Path, required=True)
    extraction.add_argument("--samples", type=int, required=True)
    extraction.add_argument("--output", type=Path, required=True)
    fitting = sub.add_parser("train")
    fitting.add_argument("--train", type=Path, required=True)
    fitting.add_argument("--train-extra", type=Path, action="append", default=[])
    fitting.add_argument("--fit", type=Path, required=True)
    fitting.add_argument("--fit-extra", type=Path, action="append", default=[])
    fitting.add_argument("--development", type=Path)
    fitting.add_argument("--development-extra", type=Path, action="append", default=[])
    fitting.add_argument("--output", type=Path, required=True)
    validation = sub.add_parser("evaluate")
    validation.add_argument("--features", type=Path, required=True)
    validation.add_argument("--run", type=Path, required=True)
    validation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    {"extract": extract, "train": train, "evaluate": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
