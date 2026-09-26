#!/usr/bin/env python3
"""Train router model2 across raw hardware and independent A2 capture domains."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from remix.identity import IdentityRuntime
from remix.packages import digest
from remix.router import FEATURES, encode
from train.router import GATES, SEED, choose_threshold, encode_rows, metrics, partition, rows


CAPTURE_GATES = {"recall": 0.75, "false_route": 0.10, "models": 3, "input_mutations": 0}
CLIPS = (
    "data/corpus/guitar-techs/P1_chords/P1_chords/audio/directinput/directinput_Set1_min.wav",
    "data/corpus/guitar-techs/P1_chords/P1_chords/audio/directinput/directinput_Set2_maj.wav",
    "data/corpus/guitar-techs/P1_chords/P1_chords/audio/directinput/directinput_Set3_min.wav",
    "data/corpus/guitar-techs/P1_chords/P1_chords/audio/directinput/directinput_Set4_maj.wav",
    "data/corpus/guitar-techs/P1_chords/P1_chords/audio/directinput/directinput_Set1_7.wav",
    "data/corpus/guitar-techs/P1_chords/P1_chords/audio/directinput/directinput_Set2_Maj7.wav",
    "data/corpus/guitar-techs/P1_chords/P1_chords/audio/directinput/directinput_Set3_dim.wav",
    "data/corpus/guitar-techs/P1_chords/P1_chords/audio/directinput/directinput_Set4_aug.wav",
)
MODELS = (
    ("fit", True, "rat40", "rat40_a2.nam", "https://www.tone3000.com/tones/proco-rat-1990s-6335", "CC0-1.0"),
    ("valid", True, "rat15", "rat15_a2.nam", "https://www.tone3000.com/tones/proco-rat-1990s-6335", "CC0-1.0"),
    ("fit", False, "duke", "duke_a2.nam", "https://www.tone3000.com/tones/kwd-drive-001-4718", "CC0-1.0"),
    ("fit", False, "demon3", "demon_g3_a2.nam", "https://www.tone3000.com/tones/demonfx-freedman-be-deluxe-ii-friedman-be-od-clone-60143", "CC0-1.0"),
    ("valid", False, "demon5", "demon_g5_a2.nam", "https://www.tone3000.com/tones/demonfx-freedman-be-deluxe-ii-friedman-be-od-clone-60143", "CC0-1.0"),
    ("fit", False, "demon8", "demon_g8_a2.nam", "https://www.tone3000.com/tones/demonfx-freedman-be-deluxe-ii-friedman-be-od-clone-60143", "CC0-1.0"),
    ("valid", False, "demon10", "demon_g10_a2.nam", "https://www.tone3000.com/tones/demonfx-freedman-be-deluxe-ii-friedman-be-od-clone-60143", "CC0-1.0"),
)


def _nam(source: Path, deps: Path):
    sys.path.append(str(deps.resolve()))
    package = types.ModuleType("nam")
    package.__path__ = [str((source / "nam").resolve())]
    sys.modules["nam"] = package
    from nam.models import init_from_nam
    return init_from_nam


def _windows(path: Path, count: int = 4) -> tuple[list[np.ndarray], dict]:
    source, rate = sf.read(path, dtype="float32", always_2d=True)
    if source.shape[1] == 1:
        value, channel_policy = source[:, 0], "mono"
    elif source.shape[1] == 2 and np.array_equal(source[:, 0], source[:, 1]):
        value, channel_policy = source[:, 0], "exact-dual-mono-left"
    else:
        raise ValueError(f"capture source is not mono or exact dual-mono: {path}")
    if rate != 48_000:
        common = np.gcd(rate, 48_000)
        value = resample_poly(value, 48_000 // common, rate // common).astype(np.float32)
    frames = 240_000
    candidates = list(range(0, len(value) - frames + 1, frames))
    if len(candidates) < count:
        raise ValueError(f"capture source is too short: {path}")
    selected = sorted(candidates, key=lambda start: (-float(np.mean(value[start:start+frames] ** 2)), start))[:count]
    return [value[start:start+frames].copy() for start in sorted(selected)], {
        "path": path.as_posix(), "sha256": digest(path), "rate": int(rate),
        "channels": int(source.shape[1]), "channel_policy": channel_policy,
        "starts": sorted(selected), "frames": frames,
    }


def capture_rows(
    root: Path, model_root: Path, init_from_nam, identity: IdentityRuntime
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], list[dict]]:
    windows, sources = [], []
    for relative in CLIPS:
        values, record = _windows(root / relative)
        record["path"] = relative
        windows.extend((relative, index, value) for index, value in enumerate(values))
        sources.append(record)
    values, truth, splits, audit = [], [], [], []
    for split, positive, name, filename, page, license_id in MODELS:
        path = model_root / filename
        document = json.loads(path.read_text())
        if document.get("architecture") != "SlimmableContainer" or document.get("sample_rate") != 48_000.0:
            raise ValueError(f"unsupported A2 container: {path}")
        nested = dict(document["config"]["submodels"][-1]["model"])
        nested["sample_rate"] = document["sample_rate"]
        model = init_from_nam(nested).eval()
        if model.receptive_field != 6_347:
            raise ValueError("A2 receptive field differs")
        for relative, window_index, dry in windows:
            before = hashlib.sha256(dry.tobytes()).hexdigest()
            with torch.inference_mode():
                wet = model(torch.from_numpy(dry), pad_start=True).numpy().astype(np.float32)
            if wet.shape != dry.shape or not np.isfinite(wet).all():
                raise ValueError("A2 render violated finite geometry")
            report = identity.infer(wet, 48_000)
            values.append(encode(identity, report, dry, wet, 48_000))
            truth.append(positive)
            splits.append(split)
            audit.append({
                "split": split, "scope": "rat" if positive else "other", "model": name,
                "model_path": path.as_posix(), "model_sha256": digest(path), "page": page,
                "license": license_id, "clip": relative, "window": window_index,
                "inputs_unchanged": before == hashlib.sha256(dry.tobytes()).hexdigest(),
            })
        print(json.dumps({"stage": "capture", "model": name, "done": len(windows)}), flush=True)
    return np.stack(values), np.asarray(truth, bool), np.asarray(splits), audit, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--nam", type=Path, required=True)
    parser.add_argument("--deps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"router run already exists: {args.output}")
    identity = IdentityRuntime(args.identity)
    training, validation = rows(args.corpus, "train"), rows(args.corpus, "eval")
    raw_x, raw_y, raw_audit = encode_rows(identity, training)
    eval_x, eval_y, eval_audit = encode_rows(identity, validation)
    cap_x, cap_y, cap_split, cap_audit, sources = capture_rows(
        args.root.resolve(), args.models.resolve(), _nam(args.nam, args.deps), identity
    )
    raw_cal = partition(training)
    cap_fit_domain = cap_split == "fit"
    cap_cal = np.asarray([
        cap_fit_domain[index] and int(hashlib.sha256(f"{row['model']}:{row['clip']}:{row['window']}".encode()).hexdigest()[:8], 16) % 5 == 0
        for index, row in enumerate(cap_audit)
    ])
    train_x = np.concatenate((raw_x[~raw_cal], cap_x[cap_fit_domain & ~cap_cal]))
    train_y = np.concatenate((raw_y[~raw_cal], cap_y[cap_fit_domain & ~cap_cal]))
    cal_x = np.concatenate((raw_x[raw_cal], cap_x[cap_cal]))
    cal_y = np.concatenate((raw_y[raw_cal], cap_y[cap_cal]))
    candidates = {
        "linear": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=SEED)),
        "hist": HistGradientBoostingClassifier(learning_rate=0.06, max_iter=300, max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=0.1, random_state=SEED),
        "trees": ExtraTreesClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=2),
    }
    screens = {}
    for name, candidate in candidates.items():
        candidate.fit(train_x, train_y)
        probability = candidate.predict_proba(cal_x)[:, 1]
        selected = choose_threshold(probability, cal_y)
        screens[name] = {"model": candidate, "threshold": selected, "metrics": metrics(probability, cal_y, selected)}
    selected_name, selected_row = max(
        screens.items(), key=lambda row: (row[1]["metrics"]["recall"], row[1]["metrics"]["balanced_accuracy"], row[0])
    )
    model, selected = selected_row["model"], selected_row["threshold"]
    raw_valid = metrics(model.predict_proba(eval_x)[:, 1], eval_y, selected)
    cap_valid_mask = cap_split == "valid"
    cap_valid = metrics(model.predict_proba(cap_x[cap_valid_mask])[:, 1], cap_y[cap_valid_mask], selected)
    cap_valid["models"] = len({cap_audit[i]["model"] for i in np.flatnonzero(cap_valid_mask)})
    mutations = sum(not row["inputs_unchanged"] for row in raw_audit + eval_audit + cap_audit)
    failures = []
    if raw_valid["recall"] < GATES["recall"]: failures.append("raw.recall")
    if raw_valid["false_route"] > GATES["false_route"]: failures.append("raw.false-route")
    if raw_valid["balanced_accuracy"] < GATES["balanced_accuracy"]: failures.append("raw.balanced-accuracy")
    if cap_valid["recall"] < CAPTURE_GATES["recall"]: failures.append("capture.recall")
    if cap_valid["false_route"] > CAPTURE_GATES["false_route"]: failures.append("capture.false-route")
    if cap_valid["models"] < CAPTURE_GATES["models"]: failures.append("capture.models")
    if mutations: failures.append("input-mutation")
    identity.assert_artifacts_unchanged()
    for record in sources:
        if digest(args.root / record["path"]) != record["sha256"]:
            raise ValueError("capture DI source changed")
    for row in cap_audit:
        if digest(Path(row["model_path"])) != row["model_sha256"]:
            raise ValueError("A2 source model changed")
    args.output.mkdir(parents=True)
    artifact = args.output / "router.joblib"
    joblib.dump({"schema": 1, "device": "rat", "features": FEATURES, "threshold": selected, "identity_package_sha256": identity.package_hash, "model": model}, artifact)
    fit_report = {
        "schema": 1, "status": "complete", "architecture": "gain-invariant identity and paired transfer features",
        "selection": selected_name, "threshold": selected, "screens": {name: row["metrics"] for name, row in screens.items()},
        "raw_fit": int((~raw_cal).sum()), "raw_calibrate": int(raw_cal.sum()),
        "capture_fit": int((cap_fit_domain & ~cap_cal).sum()), "capture_calibrate": int(cap_cal.sum()),
        "capture_sources": cap_audit, "dry_sources": sources,
        "nam": {"repository": "https://github.com/sdatkinson/neural-amp-modeler", "version": "0.13.0", "commit": "f26112906de06ec6b796ad6d1982e29eed83144e"},
        "analysis_gain_standardization": True, "source_audio_modified": False, "physical_audio_devices_used": False,
    }
    (args.output / "fit.json").write_text(json.dumps(fit_report, indent=2, sort_keys=True) + "\n")
    report = {
        "schema": 1, "status": "accepted-development" if not failures else "rejected", "accepted": not failures,
        "gates": {"raw": GATES, "capture": CAPTURE_GATES}, "raw": raw_valid, "capture": cap_valid,
        "failures": failures, "artifact": {"path": "router.joblib", "bytes": artifact.stat().st_size, "sha256": digest(artifact)},
        "input_mutations": mutations, "artifacts_unchanged": True, "analysis_gain_standardization": True,
        "source_audio_modified": False, "physical_audio_devices_used": False, "automatic_delivery": False,
        "automatic_normalization": False, "automatic_limiting": False, "lossy_reencoding": False,
        "scope": "development only; capture-model validation is not raw physical-device evidence",
    }
    (args.output / "valid.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    scores = {
        "schema": 1, "status": "development-scores", "accepted": False,
        "source": {"path": artifact.as_posix(), "sha256": digest(artifact)},
        "raw": [
            {"scope": row[0], "path": row[1].as_posix(), "probability": float(probability)}
            for row, probability in zip(validation, model.predict_proba(eval_x)[:, 1])
        ],
        "capture": [
            {"scope": cap_audit[index]["scope"], "model": cap_audit[index]["model"],
             "clip": cap_audit[index]["clip"], "window": cap_audit[index]["window"],
             "probability": float(model.predict_proba(cap_x[index:index + 1])[:, 1][0])}
            for index in np.flatnonzero(cap_valid_mask)
        ],
        "purpose": "held-out development calibration only; cannot promote",
        "source_audio_modified": False, "physical_audio_devices_used": False,
        "automatic_normalization": False, "automatic_limiting": False,
        "lossy_reencoding": False, "intermediate_audio_retained": False,
    }
    (args.output / "scores.json").write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": report["accepted"], "failures": failures, "raw": raw_valid, "capture": cap_valid}))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
