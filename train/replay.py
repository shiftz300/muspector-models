#!/usr/bin/env python3
"""Replay a consumed capture seal against a new router for diagnosis only."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from remix.identity import IdentityRuntime
from remix.packages import digest
from remix.proteus import ProteusRuntime, RATE
from remix.route_seal import CLIPS, MODELS, PACK_SHA256, _analysis_clip, gate
from remix.router import RouterRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"replay already exists: {args.output}")
    lock = json.loads(args.lock.read_text())
    if digest(args.archive) != PACK_SHA256 or lock["source"]["sha256"] != PACK_SHA256:
        raise ValueError("capture archive differs from consumed lock")
    clips = []
    for relative, expected in zip(CLIPS, lock["clips"]):
        value, record = _analysis_clip(args.root / relative)
        if record["file_sha256"] != expected["file_sha256"] or record["analysis_sha256"] != expected["analysis_sha256"]:
            raise ValueError("capture replay source differs")
        clips.append((value, Path(relative).stem))
    with zipfile.ZipFile(args.archive) as bundle:
        models = {}
        for row in lock["models"]:
            value = bundle.read(row["entry"])
            if hashlib.sha256(value).hexdigest() != row["sha256"]:
                raise ValueError("capture replay model differs")
            models[row["entry"]] = value
    identity = IdentityRuntime(args.identity)
    router = RouterRuntime(args.router, identity)
    rows = []
    for scope, name, entry in MODELS:
        renderer = ProteusRuntime.bytes(models[entry])
        for dry, clip in clips:
            before = hashlib.sha256(dry.tobytes()).hexdigest()
            wet = renderer.render(dry)
            result = router.infer_pair(dry, wet, RATE)
            rows.append({
                "scope": scope, "model": name, "clip": clip,
                "routed": result["decision"] == "candidate", "score": result["score"],
                "threshold": result["threshold"], "automatic_delivery": result["automatic_delivery"],
                "inputs_unchanged": before == hashlib.sha256(dry.tobytes()).hexdigest(),
                "nonfinite": int(not np.isfinite(wet).all()),
            })
            print(json.dumps({"model": name, "clip": clip, "route": rows[-1]["routed"], "score": rows[-1]["score"]}), flush=True)
    observed, failures, metrics = gate(rows)
    router.assert_artifacts_unchanged()
    report = {
        "schema": 1, "status": "diagnostic-consumed", "accepted": False,
        "observed_gate_pass": observed, "observed_failures": failures, "metrics": metrics,
        "reason": "these sources were already observed in the rejected route/1 seal",
        "can_promote": False, "rows": rows, "router_sha256": digest(args.router),
        "source_lock_sha256": digest(args.lock), "source_audio_modified": False,
        "physical_audio_devices_used": False, "automatic_delivery": False,
        "intermediate_audio_retained": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"observed_gate_pass": observed, "metrics": metrics, "failures": failures}))


if __name__ == "__main__":
    main()
