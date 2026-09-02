#!/usr/bin/env python3
"""Lock and seal the chain's Drive slice on offline ASRNN RAT recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from .asrnn_effects import effect_controls, effect_files, read_effect_pair
from .chain import ChainRuntime
from .family import FRAMES


NAMES = ("drive.gain_db", "drive.tone", "drive.level_db")
GATES = {
    "coverage": 0.35,
    "accepted_exact": 0.98,
    "knob_mae": 0.13,
    "knob_p95": 0.35,
    "control_mae": 0.28,
    "control_p95": 0.65,
    "input_mutations": 0,
    "runtime_errors": 0,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def spread(paths: list[Path], count: int) -> list[Path]:
    if len(paths) < count:
        raise ValueError(f"not enough ASRNN RAT eval files: {len(paths)} < {count}")
    indices = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[int(index)] for index in indices]


def analysis(path: Path) -> tuple[np.ndarray, np.ndarray]:
    dry, wet, _ = read_effect_pair(path, "rat")
    dry = resample_poly(dry, 147, 160).astype(np.float32)
    wet = resample_poly(wet, 147, 160).astype(np.float32)
    if len(dry) < FRAMES:
        dry = np.pad(dry, (0, FRAMES - len(dry)))
        wet = np.pad(wet, (0, FRAMES - len(wet)))
    return dry[:FRAMES].astype(np.float32, copy=False), wet[:FRAMES].astype(np.float32, copy=False)


def lock(args) -> dict:
    if args.output.exists():
        raise FileExistsError(f"ASRNN lock already exists: {args.output}")
    paths = spread(effect_files(args.corpus, "rat", "eval"), args.samples)
    records = [
        {
            "path": path.relative_to(args.corpus).as_posix(),
            "sha256": digest(path),
            "controls": effect_controls(path, "rat").tolist(),
        }
        for path in paths
    ]
    report = {
        "schema": 1,
        "status": "locked-unopened",
        "source": "ASRNN ProCo RAT official eval",
        "record": "https://zenodo.org/records/20406285",
        "license": "CC-BY-NC-4.0",
        "selection": "evenly spaced by sorted filename before candidate inference",
        "samples": args.samples,
        "records": records,
        "candidate_exposure": "excluded from active family, gate, order, and knob candidates",
        "scope": "external Drive and knob slice; not a complete multi-family locked-final set",
        "source_audio_modified": False,
        "physical_audio_devices_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def seal(args) -> dict:
    if args.output.exists():
        raise FileExistsError(f"ASRNN seal already exists: {args.output}")
    lock_bytes = args.lock.read_bytes()
    locked = json.loads(lock_bytes)
    if locked.get("status") != "locked-unopened":
        raise ValueError("ASRNN source was not locked before inference")
    runtime = ChainRuntime(
        args.family, args.manifest, args.gate, args.order, args.bundle, args.evidence
    )
    rows = []
    for index, record in enumerate(locked["records"]):
        path = args.corpus / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError("locked ASRNN source changed")
        dry, wet = analysis(path)
        before = hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest()
        result = runtime.infer(dry, wet)
        accepted = result["decision"] != "abstain"
        knobs = result["knobs"]
        errors = {}
        if accepted and knobs is not None:
            errors = {
                name: abs(float(knobs["controls"][name]["normalized"]) - float(record["controls"][position]))
                for position, name in enumerate(NAMES)
                if name in knobs["controls"]
            }
        runtime_valid = (
            not result["quality"]["physical_audio_devices_used"]
            and (
                not accepted
                or not result["active"]
                or (knobs is not None and knobs["active"] == result["active"])
            )
        )
        rows.append({
            "index": index,
            "path": record["path"],
            "accepted": accepted,
            "confidence": result["gate"]["confidence"],
            "predicted": result["active"],
            "exact": result["active"] == ["drive"],
            "controls": errors,
            "runtime_valid": bool(runtime_valid),
            "inputs_unchanged": before == hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest(),
        })
        if (index + 1) % 8 == 0 or index + 1 == len(locked["records"]):
            print(json.dumps({"done": index + 1, "total": len(locked["records"])}), flush=True)
    runtime.assert_artifacts_unchanged()
    for record in locked["records"]:
        if digest(args.corpus / record["path"]) != record["sha256"]:
            raise ValueError("ASRNN source changed during seal")
    accepted = [row for row in rows if row["accepted"]]
    controls = {name: [row["controls"][name] for row in accepted if name in row["controls"]] for name in NAMES}
    values = [value for errors in controls.values() for value in errors]
    metrics = {
        "examples": len(rows),
        "accepted": len(accepted),
        "coverage": len(accepted) / max(len(rows), 1),
        "accepted_exact": sum(row["exact"] for row in accepted) / max(len(accepted), 1),
        "accepted_errors": sum(not row["exact"] for row in accepted),
        "knob_mae": float(np.mean(values)) if values else 0.0,
        "knob_p95": float(np.quantile(values, 0.95)) if values else 0.0,
        "controls": {
            name: {
                "values": len(errors),
                "mae": float(np.mean(errors)) if errors else 0.0,
                "p95": float(np.quantile(errors, 0.95)) if errors else 0.0,
            }
            for name, errors in controls.items()
        },
        "runtime_errors": sum(not row["runtime_valid"] for row in rows),
        "input_mutations": sum(not row["inputs_unchanged"] for row in rows),
    }
    failures = []
    if metrics["coverage"] < GATES["coverage"]: failures.append("coverage")
    if metrics["accepted_exact"] < GATES["accepted_exact"]: failures.append("accepted-exact")
    if metrics["knob_mae"] > GATES["knob_mae"]: failures.append("knob-mae")
    if metrics["knob_p95"] > GATES["knob_p95"]: failures.append("knob-p95")
    for name, values in metrics["controls"].items():
        if not values["values"]: failures.append(f"{name}.missing")
        elif values["mae"] > GATES["control_mae"]: failures.append(f"{name}.mae")
        elif values["p95"] > GATES["control_p95"]: failures.append(f"{name}.p95")
    if metrics["runtime_errors"] > GATES["runtime_errors"]: failures.append("runtime-error")
    if metrics["input_mutations"] > GATES["input_mutations"]: failures.append("input-mutation")
    report = {
        "schema": 1,
        "status": "accepted-sealed-drive-slice" if not failures else "rejected",
        "accepted": not failures,
        "source": locked["source"],
        "scope": locked["scope"],
        "license": locked["license"],
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
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
    commands = parser.add_subparsers(dest="mode", required=True)
    locking = commands.add_parser("lock")
    locking.add_argument("--corpus", type=Path, required=True)
    locking.add_argument("--samples", type=int, default=32)
    locking.add_argument("--output", type=Path, required=True)
    sealing = commands.add_parser("seal")
    sealing.add_argument("--lock", type=Path, required=True)
    sealing.add_argument("--corpus", type=Path, required=True)
    sealing.add_argument("--family", type=Path, required=True)
    sealing.add_argument("--manifest", type=Path, required=True)
    sealing.add_argument("--gate", type=Path, required=True)
    sealing.add_argument("--order", type=Path, required=True)
    sealing.add_argument("--bundle", type=Path, required=True)
    sealing.add_argument("--evidence", type=Path, required=True)
    sealing.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = lock(args) if args.mode == "lock" else seal(args)
    print(json.dumps({"status": report["status"], "accepted": report.get("accepted"), "failures": report.get("failures", [])}))
    if args.mode == "seal" and not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
