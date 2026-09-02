#!/usr/bin/env python3
"""Replay the explicit RAT adapter behind the canonical chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .adapter import RAT_NAMES, RatAdapter, analysis
from .asrnn_effects import read_effect_pair
from .chain import ChainRuntime
from .packages import digest


GATES = {
    "coverage": 0.35,
    "macro_mae": 0.08,
    "macro_p95": 0.22,
    "controls": {
        "distortion": {"mae": 0.08, "p95": 0.22},
        "filter": {"mae": 0.08, "p95": 0.22},
        "volume": {"mae": 0.06, "p95": 0.18},
    },
    "input_mutations": 0,
    "artifact_mutations": 0,
}


def evaluate(args) -> dict:
    if args.output.exists():
        raise FileExistsError(f"adapter report already exists: {args.output}")
    locked_bytes = args.lock.read_bytes()
    locked = json.loads(locked_bytes)
    chain = ChainRuntime(
        args.family, args.manifest, args.gate, args.order, args.bundle, args.evidence
    )
    adapter = RatAdapter(args.adapter)
    rows, routed_dry, routed_wet, routed_indices = [], [], [], []
    for index, record in enumerate(locked["records"]):
        path = args.corpus / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError("locked ASRNN source changed")
        dry, wet, controls = read_effect_pair(path, "rat")
        before = hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest()
        canonical = chain.infer(
            analysis(dry, 48_000, 44_100, 220_500),
            analysis(wet, 48_000, 44_100, 220_500),
        )
        routed = canonical["decision"] == "accepted" and canonical["active"] == ["drive"]
        physical_truth = np.asarray((controls[0], 1.0 - controls[1], controls[2]), dtype=np.float32)
        rows.append({
            "index": index,
            "path": record["path"],
            "chain_decision": canonical["decision"],
            "chain_active": canonical["active"],
            "routed": routed,
            "truth": dict(zip(RAT_NAMES, map(float, physical_truth))),
            "adapter": None,
            "controls": {},
            "inputs_unchanged": before == hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest(),
        })
        if routed:
            routed_indices.append(index)
            routed_dry.append(analysis(dry, 48_000, 48_000, 48_000))
            routed_wet.append(analysis(wet, 48_000, 48_000, 48_000))
        if (index + 1) % 8 == 0 or index + 1 == len(locked["records"]):
            print(json.dumps({"stage": "chain", "done": index + 1, "total": len(locked["records"])}), flush=True)
    if routed_indices:
        device = adapter.infer_batch(np.stack(routed_dry), np.stack(routed_wet))
        for index, result in zip(routed_indices, device):
            rows[index]["adapter"] = result["decision"]
            if result["decision"] == "accepted":
                rows[index]["controls"] = {
                    name: abs(result["controls"][name]["normalized"] - rows[index]["truth"][name])
                    for name in RAT_NAMES
                }
    chain.assert_artifacts_unchanged()
    adapter.assert_artifacts_unchanged()
    for record in locked["records"]:
        if digest(args.corpus / record["path"]) != record["sha256"]:
            raise ValueError("ASRNN source changed during adapter replay")
    delivered = [row for row in rows if row["adapter"] == "accepted"]
    errors = {name: [row["controls"][name] for row in delivered] for name in RAT_NAMES}
    all_errors = [value for values in errors.values() for value in values]
    metrics = {
        "examples": len(rows),
        "chain_routed": sum(row["routed"] for row in rows),
        "delivered": len(delivered),
        "coverage": len(delivered) / max(len(rows), 1),
        "macro_mae": float(np.mean(all_errors)) if all_errors else 0.0,
        "macro_p95": float(np.quantile(all_errors, 0.95)) if all_errors else 0.0,
        "controls": {
            name: {
                "values": len(values),
                "mae": float(np.mean(values)) if values else 0.0,
                "p95": float(np.quantile(values, 0.95)) if values else 0.0,
            }
            for name, values in errors.items()
        },
        "input_mutations": sum(not row["inputs_unchanged"] for row in rows),
    }
    failures = []
    if metrics["coverage"] < GATES["coverage"]: failures.append("coverage")
    if metrics["macro_mae"] > GATES["macro_mae"]: failures.append("macro-mae")
    if metrics["macro_p95"] > GATES["macro_p95"]: failures.append("macro-p95")
    for name, gate in GATES["controls"].items():
        values = metrics["controls"][name]
        if not values["values"]: failures.append(f"{name}.missing")
        elif values["mae"] > gate["mae"]: failures.append(f"{name}.mae")
        elif values["p95"] > gate["p95"]: failures.append(f"{name}.p95")
    if metrics["input_mutations"] > GATES["input_mutations"]: failures.append("input-mutation")
    report = {
        "schema": 1,
        "status": "accepted-development-replay" if not failures else "rejected",
        "accepted": not failures,
        "device": "rat",
        "source": "ASRNN official eval reused from ratknobs development; not new sealed evidence",
        "lock_sha256": hashlib.sha256(locked_bytes).hexdigest(),
        "gates": GATES,
        "metrics": metrics,
        "failures": failures,
        "rows": rows,
        "artifacts_unchanged": True,
        "source_audio_modified": False,
        "physical_audio_devices_used": False,
        "automatic_normalization": False,
        "automatic_limiting": False,
        "lossy_reencoding": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = evaluate(args)
    print(json.dumps({"accepted": report["accepted"], "failures": report["failures"], "metrics": report["metrics"]}))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
