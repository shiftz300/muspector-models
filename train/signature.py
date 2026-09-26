#!/usr/bin/env python3
"""Train a content-reduced RAT signature with leave-one-device-group-out validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

from remix.a2 import A2Runtime, loader
from remix.asrnn_effects import read_effect_pair
from remix.packages import digest
from remix.signature import FEATURES, RATE, encode
from train.router import rows
from train import router2, router4, router6
from remix import verify3


SEED = 20260901
GATES = {"recall": 0.75, "false_route": 0.10, "positive_group_recall": 0.50,
         "negative_group_false_route": 0.25, "input_mutations": 0}


def analysis(path: Path, seconds: int = 5) -> tuple[np.ndarray, dict]:
    source, rate = sf.read(path, dtype="float32", always_2d=True)
    if source.shape[1] == 1:
        mono, policy = source[:, 0], "mono"
    elif source.shape[1] == 2 and np.array_equal(source[:, 0], source[:, 1]):
        mono, policy = source[:, 0], "exact-dual-mono-left"
    else:
        raise ValueError(f"signature source is not mono or exact dual-mono: {path}")
    if rate != RATE:
        common = np.gcd(rate, RATE)
        mono = resample_poly(mono, RATE // common, rate // common).astype(np.float32)
    frames = seconds * RATE
    starts = list(range(0, len(mono) - frames + 1, RATE))
    if not starts:
        raise ValueError(f"signature source is too short: {path}")
    start = min(starts, key=lambda item: (-float(np.mean(mono[item:item + frames] ** 2)), item))
    value = mono[start:start + frames].copy()
    return value, {"path": path.as_posix(), "sha256": digest(path), "source_rate": int(rate),
                   "source_channels": int(source.shape[1]), "channel_policy": policy,
                   "analysis_rate": RATE, "analysis_start": start, "analysis_frames": frames,
                   "analysis_sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def models() -> list[tuple]:
    training = list(router2.MODELS) + list(router4.EXPANDED) + list(router6.EXPANDED)
    result = [
        (positive, name, filename, page, license_id)
        for _, positive, name, filename, page, license_id in training
    ]
    result.extend(
        (scope == "rat", name, filename, page, license_id)
        for scope, name, filename, _, page, license_id in verify3.MODELS
    )
    return result


def clips() -> tuple[str, ...]:
    return router2.CLIPS + router4.P2 + router6.P3 + verify3.CLIPS


def weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(groups.tolist())
    return np.asarray([1.0 / counts[group] for group in groups], np.float64)


def metrics(scores: np.ndarray, truth: np.ndarray, groups: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    positive_groups = {group: float(predicted[(groups == group) & truth].mean()) for group in sorted(set(groups[truth]))}
    negative_groups = {group: float(predicted[(groups == group) & ~truth].mean()) for group in sorted(set(groups[~truth]))}
    recall = float(predicted[truth].mean())
    false_route = float(predicted[~truth].mean())
    return {"examples": len(truth), "positive": int(truth.sum()), "negative": int((~truth).sum()),
            "threshold": threshold, "recall": recall, "false_route": false_route,
            "balanced_accuracy": (recall + 1.0 - false_route) / 2.0,
            "positive_groups": positive_groups, "negative_groups": negative_groups,
            "worst_positive_group_recall": min(positive_groups.values()),
            "worst_negative_group_false_route": max(negative_groups.values())}


def choose(scores: np.ndarray, truth: np.ndarray, groups: np.ndarray) -> tuple[float, dict, list[str]]:
    feasible = []
    examined = []
    for threshold in np.linspace(0, 1, 1001):
        report = metrics(scores, truth, groups, float(threshold)); examined.append(report)
        if (report["recall"] >= GATES["recall"] and report["false_route"] <= GATES["false_route"]
                and report["worst_positive_group_recall"] >= GATES["positive_group_recall"]
                and report["worst_negative_group_false_route"] <= GATES["negative_group_false_route"]):
            feasible.append(report)
    if feasible:
        selected = max(feasible, key=lambda row: (row["balanced_accuracy"], row["worst_positive_group_recall"], -row["worst_negative_group_false_route"], row["threshold"]))
        return selected["threshold"], selected, []
    selected = max(examined, key=lambda row: (row["balanced_accuracy"], row["worst_positive_group_recall"], -row["worst_negative_group_false_route"]))
    failures = []
    for key, comparator in (("recall", lambda value: value < GATES["recall"]),
                            ("false_route", lambda value: value > GATES["false_route"]),
                            ("worst_positive_group_recall", lambda value: value < GATES["positive_group_recall"]),
                            ("worst_negative_group_false_route", lambda value: value > GATES["negative_group_false_route"])):
        if comparator(selected[key]): failures.append(key)
    return selected["threshold"], selected, failures


def candidate(name: str):
    if name == "trees": return ExtraTreesClassifier(n_estimators=320, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=2)
    if name == "forest": return RandomForestClassifier(n_estimators=320, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=2)
    if name == "hist": return HistGradientBoostingClassifier(learning_rate=.06, max_iter=240, max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=.1, random_state=SEED)
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--nam", type=Path, required=True)
    parser.add_argument("--deps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"signature run already exists: {args.output}")
    values, truth, groups, audit = [], [], [], []
    mutations = 0
    raw_records = rows(args.corpus, "train") + rows(args.corpus, "eval")
    for index, (device, path) in enumerate(raw_records):
        dry, wet, _ = read_effect_pair(path, device); before = hashlib.sha256(dry.tobytes()+wet.tobytes()).hexdigest()
        values.append(encode(dry, wet, RATE)); truth.append(device == "rat"); groups.append(f"asrnn:{device}")
        unchanged = before == hashlib.sha256(dry.tobytes()+wet.tobytes()).hexdigest(); mutations += not unchanged
        audit.append({"domain":"raw","group":groups[-1],"path":path.as_posix(),"sha256":digest(path),"inputs_unchanged":unchanged})
        if (index + 1) % 64 == 0 or index + 1 == len(raw_records): print(json.dumps({"stage":"raw","done":index+1,"total":len(raw_records)}),flush=True)
    dry_values, dry_records = [], []
    for relative in clips():
        value, record = analysis(args.root / relative); record["path"] = relative
        dry_values.append(value); dry_records.append(record)
    init = loader(args.nam, args.deps)
    model_records = []
    for positive, name, filename, page, license_id in models():
        path = args.models / filename; runtime = A2Runtime(path, init); group = page.rsplit("/", 1)[-1]
        model_records.append({"scope":"rat" if positive else "other","model":name,"group":group,"path":path.as_posix(),"sha256":digest(path),"page":page,"license":license_id})
        for dry, source in zip(dry_values, dry_records):
            before = hashlib.sha256(dry.tobytes()).hexdigest(); wet = runtime.render(dry)
            values.append(encode(dry, wet, RATE)); truth.append(positive); groups.append(group)
            unchanged = before == hashlib.sha256(dry.tobytes()).hexdigest(); mutations += not unchanged
            audit.append({"domain":"a2","group":group,"model":name,"clip":source["path"],"inputs_unchanged":unchanged})
        runtime.assert_unchanged(); print(json.dumps({"stage":"a2","model":name,"done":len(dry_values)}),flush=True)
    x, y, group_array = np.stack(values), np.asarray(truth,bool), np.asarray(groups)
    screens = {}
    for name in ("trees","forest","hist"):
        oof = np.zeros(len(y), np.float64)
        for held in sorted(set(groups)):
            valid = group_array == held; train = ~valid; model = candidate(name)
            model.fit(x[train], y[train], sample_weight=weights(group_array[train]))
            oof[valid] = model.predict_proba(x[valid])[:,1]
        threshold, report, failures = choose(oof, y, group_array)
        screens[name] = {"threshold":threshold,"metrics":report,"failures":failures,"oof":oof}
        print(json.dumps({"stage":"screen","candidate":name,"metrics":report,"failures":failures}),flush=True)
    selected_name, selected = max(screens.items(), key=lambda row: (not row[1]["failures"], row[1]["metrics"]["balanced_accuracy"], row[1]["metrics"]["worst_positive_group_recall"], row[0]))
    final = candidate(selected_name); final.fit(x, y, sample_weight=weights(group_array))
    args.output.mkdir(parents=True)
    artifact = args.output / "signature.joblib"
    joblib.dump({"schema":2,"device":"rat","features":FEATURES,"threshold":selected["threshold"],"model":final},artifact)
    np.savez_compressed(args.output / "data.npz", x=x, y=y, groups=group_array)
    data_report = {"schema":1,"status":"complete","examples":len(y),"features":len(FEATURES),"groups":dict(Counter(groups)),
                   "dry_sources":dry_records,"models":model_records,"audit":audit,"input_mutations":mutations,
                   "source_audio_modified":False,"physical_audio_devices_used":False,"rendered_audio_retained":False}
    (args.output/"data.json").write_text(json.dumps(data_report,indent=2,sort_keys=True)+"\n")
    report = {"schema":2,"status":"accepted-development" if not selected["failures"] and not mutations else "rejected",
              "accepted":not selected["failures"] and not mutations,"architecture":"content-reduced paired transfer signature",
              "selection":selected_name,"threshold":selected["threshold"],"gates":GATES,"metrics":selected["metrics"],
              "failures":selected["failures"] + (["input-mutation"] if mutations else []),
              "screens":{name:{"threshold":row["threshold"],"metrics":row["metrics"],"failures":row["failures"]} for name,row in screens.items()},
              "artifact":{"path":"signature.joblib","bytes":artifact.stat().st_size,"sha256":digest(artifact)},
              "scope":"leave-one-device-group-out development evidence; requires a new untouched seal",
              "source_audio_modified":False,"physical_audio_devices_used":False,"automatic_delivery":False,
              "automatic_normalization":False,"automatic_limiting":False,"lossy_reencoding":False}
    (args.output/"valid.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"accepted":report["accepted"],"selection":selected_name,"metrics":selected["metrics"],"failures":report["failures"]}))
    if not report["accepted"]: raise SystemExit(2)


if __name__ == "__main__": main()
