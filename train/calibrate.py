#!/usr/bin/env python3
"""Calibrate a frozen router head at the cross-domain probability margin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from remix.packages import digest


GATES = {
    "raw": {"recall": 0.80, "false_route": 0.05},
    "capture": {"recall": 0.75, "false_route": 0.10},
}


def metrics(rows: list[dict], threshold: float, positive: str) -> dict:
    truth = np.asarray([row["scope"] == positive for row in rows])
    probability = np.asarray([row["probability"] for row in rows])
    predicted = probability >= threshold
    recall = float(predicted[truth].mean())
    false_route = float(predicted[~truth].mean())
    return {
        "examples": len(rows), "positive": int(truth.sum()), "negative": int((~truth).sum()),
        "threshold": threshold, "recall": recall, "false_route": false_route,
        "balanced_accuracy": (recall + 1.0 - false_route) / 2.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"calibrated run already exists: {args.output}")
    score_data = json.loads(args.scores.read_text())
    if score_data["source"]["sha256"] != digest(args.source):
        raise ValueError("router score source differs")
    positive = [row["probability"] for row in score_data["capture"] if row["scope"] == "rat"]
    negative = [row["probability"] for row in score_data["capture"] if row["scope"] != "rat"]
    lower, upper = float(max(negative)), float(min(positive))
    separated = lower < upper
    candidates = []
    for threshold in np.linspace(0.0, 1.0, 1001):
        capture_result = metrics(score_data["capture"], float(threshold), "rat")
        raw_result = metrics(score_data["raw"], float(threshold), "rat")
        if all((
            capture_result["recall"] >= GATES["capture"]["recall"],
            capture_result["false_route"] <= GATES["capture"]["false_route"],
            raw_result["recall"] >= GATES["raw"]["recall"],
            raw_result["false_route"] <= GATES["raw"]["false_route"],
        )):
            candidates.append((capture_result["recall"], -capture_result["false_route"], raw_result["recall"], float(threshold)))
    if separated:
        threshold = (lower + upper) / 2.0
        selection = "midpoint between closest A2 development classes"
    elif candidates:
        threshold = max(candidates)[-1]
        selection = "maximum capture recall under frozen dual-domain gates"
    else:
        raise ValueError("capture development scores have no threshold satisfying frozen gates")
    capture = metrics(score_data["capture"], threshold, "rat")
    raw = metrics(score_data["raw"], threshold, "rat")
    failures = []
    for domain, result in (("raw", raw), ("capture", capture)):
        if result["recall"] < GATES[domain]["recall"]:
            failures.append(f"{domain}.recall")
        if result["false_route"] > GATES[domain]["false_route"]:
            failures.append(f"{domain}.false-route")
    payload = joblib.load(args.source)
    if payload.get("schema") != 1 or payload.get("device") != "rat":
        raise ValueError("router source contract differs")
    payload["threshold"] = threshold
    args.output.mkdir(parents=True)
    artifact = args.output / "router.joblib"
    joblib.dump(payload, artifact)
    fit = {
        "schema": 1, "status": "complete", "operation": "threshold-only cross-domain calibration",
        "source_artifact_sha256": digest(args.source), "source_scores_sha256": digest(args.scores),
        "negative_max": lower, "positive_min": upper, "margin": upper - lower,
        "classes_separated": separated, "selection": selection, "threshold": threshold,
        "weights_changed": False, "source_audio_modified": False, "physical_audio_devices_used": False,
    }
    (args.output / "fit.json").write_text(json.dumps(fit, indent=2, sort_keys=True) + "\n")
    report = {
        "schema": 1, "status": "accepted-development" if not failures else "rejected", "accepted": not failures,
        "gates": GATES, "raw": raw, "capture": capture, "failures": failures,
        "artifact": {"path": "router.joblib", "bytes": artifact.stat().st_size, "sha256": digest(artifact)},
        "scope": "development calibration; requires a new untouched multi-author seal",
        "weights_changed": False, "analysis_gain_standardization": True,
        "source_audio_modified": False, "physical_audio_devices_used": False,
        "automatic_delivery": False, "automatic_normalization": False,
        "automatic_limiting": False, "lossy_reencoding": False,
    }
    (args.output / "valid.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": report["accepted"], "threshold": threshold, "raw": raw, "capture": capture, "failures": failures}))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
