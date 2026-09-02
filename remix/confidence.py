"""Train the wet-only information gate for clean restoration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier

from .asrnn_data import rat_files, read_rat_pair
from .clean import SpectralNet, accepted as restoration_accepted, evaluate
from .train_asrnn_rat_adapter import _partition


def features(wet: np.ndarray) -> np.ndarray:
    value = np.asarray(wet, dtype=np.float64)[1024:]
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ValueError("confidence input must be finite mono audio")
    absolute = np.abs(value)
    rms = np.sqrt(np.mean(value * value)) + 1e-12
    peak = absolute.max() + 1e-12
    blocks = np.asarray([np.sqrt(np.mean(block * block)) + 1e-12 for block in np.array_split(value, 32)])
    count = min(len(value), 131072)
    spectrum = np.abs(np.fft.rfft(value[:count] * np.hanning(count))) ** 2 + 1e-18
    edges = np.linspace(0, len(spectrum), 17, dtype=int)
    bands = np.asarray([spectrum[edges[index] : edges[index + 1]].sum() for index in range(16)])
    bands /= bands.sum()
    result = np.r_[
        np.log(rms), np.log(peak), np.log(peak / rms),
        np.quantile(absolute, (.1, .25, .5, .75, .9, .95, .99)),
        np.quantile(np.log(blocks), (0, .1, .25, .5, .75, .9, 1)),
        np.log(bands),
    ]
    if not np.isfinite(result).all():
        raise ValueError("confidence features are non-finite")
    return result.astype(np.float32)


def rows(paths: list[Path], minimum_ratio: float):
    matrix, truth = [], []
    for path in paths:
        dry, wet, _ = read_rat_pair(path)
        dry_rms = np.sqrt(np.mean(np.square(dry[1024:], dtype=np.float64))) + 1e-12
        wet_rms = np.sqrt(np.mean(np.square(wet[1024:], dtype=np.float64))) + 1e-12
        matrix.append(features(wet))
        truth.append(wet_rms / dry_rms >= minimum_ratio)
    return np.stack(matrix), np.asarray(truth, dtype=bool)


def threshold(truth: np.ndarray, probability: np.ndarray) -> float:
    choices = []
    for value in sorted(set(map(float, probability))):
        selected = probability >= value
        false_accepts = int(np.count_nonzero(selected & ~truth))
        recall = float(np.count_nonzero(selected & truth) / max(np.count_nonzero(truth), 1))
        if false_accepts == 0:
            choices.append((recall, -value, value))
    if not choices:
        raise ValueError("calibration cannot form a zero-false-accept threshold")
    return max(choices)[-1]


def metrics(truth: np.ndarray, probability: np.ndarray, selected_threshold: float) -> dict:
    selected = probability >= selected_threshold
    return {
        "examples": len(truth),
        "eligible": int(np.count_nonzero(truth)),
        "ineligible": int(np.count_nonzero(~truth)),
        "eligible_recall": float(np.count_nonzero(selected & truth) / max(np.count_nonzero(truth), 1)),
        "ineligible_false_accept": float(np.count_nonzero(selected & ~truth) / max(np.count_nonzero(~truth), 1)),
        "coverage": float(np.mean(selected)),
        "accepted": int(np.count_nonzero(selected)),
    }


def load_restorer(path: Path, target: torch.device) -> SpectralNet:
    payload = torch.load(path, map_location=target, weights_only=True)
    if payload.get("architecture") != "complex-stft" or payload.get("device") != "rat":
        raise ValueError("confidence seal requires a RAT complex-STFT restorer")
    model = SpectralNet(payload["channels"], payload["n_fft"], payload["hop"]).to(target)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/asrnn-physical-effects"))
    parser.add_argument("--cycle", type=Path, default=Path("cycles/clean5.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/clean/model4/clean.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/clean/model5"))
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to replace confidence run {args.output}")
    cycle_bytes = args.cycle.read_bytes(); cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "clean1" or cycle.get("round") != 5 or cycle.get("status") != "planned": raise ValueError("invalid clean5 cycle")
    args.output.mkdir(parents=True)
    fit, calibration = _partition(rat_files(args.corpus, "train")); development = rat_files(args.corpus, "eval")
    minimum = cycle["eligibility"]["minimum_wet_to_dry_rms_ratio"]
    fit_x, fit_y = rows(fit, minimum); calibration_x, calibration_y = rows(calibration, minimum); development_x, development_y = rows(development, minimum)
    model = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=4, max_features=.8, n_jobs=-1, class_weight="balanced", random_state=20260901)
    model.fit(fit_x, fit_y)
    calibration_probability = model.predict_proba(calibration_x)[:, 1]
    selected_threshold = threshold(calibration_y, calibration_probability)
    development_probability = model.predict_proba(development_x)[:, 1]
    calibration_metrics = metrics(calibration_y, calibration_probability, selected_threshold)
    development_metrics = metrics(development_y, development_probability, selected_threshold)
    selected_paths = [path for path, selected in zip(development, development_probability >= selected_threshold) if selected]
    target = torch.device("cpu"); restorer = load_restorer(args.checkpoint, target)
    restoration = evaluate(restorer, selected_paths, target)["metrics"]
    restoration["coverage"] = development_metrics["coverage"]
    gate_failures = []
    for name, rule in cycle["gates"]["confidence"].items():
        if name.endswith("_minimum"):
            metric = name.removesuffix("_minimum")
            if development_metrics[metric] < rule: gate_failures.append(metric)
        elif name.endswith("_maximum"):
            metric = name.removesuffix("_maximum")
            if development_metrics[metric] > rule: gate_failures.append(metric)
    restoration_passed, restoration_failures = restoration_accepted(restoration, cycle["gates"]["restoration"])
    artifact = args.output / "gate.joblib"; joblib.dump(model, artifact)
    result = {
        "schema": 1,
        "status": "accepted-development" if not gate_failures and restoration_passed else "rejected",
        "accepted": not gate_failures and restoration_passed,
        "failures": {"confidence": gate_failures, "restoration": restoration_failures},
        "threshold": selected_threshold,
        "confidence": {"calibration": calibration_metrics, "development": development_metrics},
        "restoration": restoration,
        "artifacts": {"gate": hashlib.sha256(artifact.read_bytes()).hexdigest(), "restorer": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()},
        "data": {"source_read_only": True, "physical_audio_devices_used": False, "license": "CC-BY-NC-4.0"},
        "quality": {"source_audio_modified": False, "automatic_normalization": False, "automatic_limiting": False, "automatic_dither": False, "lossy_reencoding": False},
        "limitations": ["ASRNN-only development gate", "scikit-learn artifact is not a client runtime package", "cross-device seal remains required"],
        "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest(),
    }
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
