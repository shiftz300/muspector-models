#!/usr/bin/env python3
"""Score model4 on all consumed development domains for calibration audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from remix.identity import IdentityRuntime
from remix.packages import digest
from train import router2, router4
from train.router import encode_rows, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--nam", type=Path, required=True)
    parser.add_argument("--deps", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"score report already exists: {args.output}")
    identity = IdentityRuntime(args.identity)
    payload = joblib.load(args.router)
    raw_records = rows(args.corpus, "eval")
    raw_x, _, raw_audit = encode_rows(identity, raw_records)
    router2.CLIPS = router2.CLIPS + router4.P2
    router2.MODELS = tuple(row for row in router2.MODELS + router4.EXPANDED if row[0] == "valid")
    cap_x, _, _, cap_audit, dry_sources = router2.capture_rows(
        args.root.resolve(), args.models.resolve(), router2._nam(args.nam, args.deps), identity
    )
    raw_probability = payload["model"].predict_proba(raw_x)[:, 1]
    cap_probability = payload["model"].predict_proba(cap_x)[:, 1]
    raw = [
        {"scope": record["device"], "path": record["path"], "probability": float(probability)}
        for record, probability in zip(raw_audit, raw_probability)
    ]
    capture = [
        {"scope": record["scope"], "model": record["model"], "clip": record["clip"],
         "window": record["window"], "probability": float(probability)}
        for record, probability in zip(cap_audit, cap_probability)
    ]
    document = {
        "schema": 1, "status": "development-scores", "accepted": False,
        "source": {"path": args.router.as_posix(), "sha256": digest(args.router)},
        "raw": raw, "capture": capture, "dry_sources": dry_sources,
        "purpose": "consumed development calibration only; cannot promote",
        "source_audio_modified": False, "physical_audio_devices_used": False,
        "automatic_normalization": False, "automatic_limiting": False,
        "lossy_reencoding": False, "intermediate_audio_retained": False,
    }
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    positive = [row["probability"] for row in capture if row["scope"] == "rat"]
    negative = [row["probability"] for row in capture if row["scope"] != "rat"]
    print(json.dumps({"positive_min": min(positive), "negative_max": max(negative),
                      "positive": len(positive), "negative": len(negative)}))


if __name__ == "__main__":
    main()
