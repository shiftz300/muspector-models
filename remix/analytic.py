"""Calibrate and seal the analytic blind Reverb inverse."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np

from .blind import matrix
from .deverb import restore
from .forward_chain import ForwardChainRuntime
from .oracle import examples, gates
from .refine import rank
from .stages import CORPUS, metric, sha256


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "cycles/stage13.json"
DRIVE = ROOT / "runs/stage/model8/drive.pt"
DRIVE_CONTROL = ROOT / "runs/stage/model9/drive.candidate.joblib"
REVERB_CONTROL = ROOT / "runs/stage/model9/reverb.candidate.joblib"
PROFILE = ROOT.parent / "muspector/remix/runs/reverb-forward-phase1/reverb-device-profile.npz"
RUN = ROOT / "runs/stage/model13"


def controls(rows: list[dict], estimator) -> tuple[np.ndarray, np.ndarray]:
    features, truth = matrix(rows)
    return np.clip(estimator.predict(features), 0.0, 1.0).astype(np.float32), truth


def evaluate(runtime: ForwardChainRuntime, rows: list[dict], values: np.ndarray, regularization: float) -> dict:
    predictions = [
        restore(runtime.reverb, row["source"].numpy(), control, regularization)
        for row, control in zip(rows, values, strict=True)
    ]
    report = metric(
        [row["source"].numpy() for row in rows],
        predictions,
        [row["target"].numpy() for row in rows],
    )
    report["domains"] = sorted({str(row["domain"]) for row in rows})
    report["regularization"] = float(regularization)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, default=CYCLE)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--drive", type=Path, default=DRIVE)
    parser.add_argument("--drive-control", type=Path, default=DRIVE_CONTROL)
    parser.add_argument("--reverb-control", type=Path, default=REVERB_CONTROL)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--calibrate", type=int, default=192)
    parser.add_argument("--valid", type=int, default=320)
    parser.add_argument("--frames", type=int, default=32768)
    parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace analytic run {args.output}")
    cycle_bytes = args.cycle.read_bytes()
    cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "stage" or cycle.get("round") != 13 or cycle.get("status") != "planned":
        raise ValueError("invalid stage13 cycle")
    paths = {"drive": args.drive, "drive_control": args.drive_control, "reverb_control": args.reverb_control, "reverb_profile": args.profile}
    for name, path in paths.items():
        if sha256(path) != cycle["baseline"][name + "_sha256"]:
            raise ValueError(f"{name} baseline changed")
    args.output.mkdir(parents=True)
    estimator = joblib.load(args.reverb_control)
    runtime = ForwardChainRuntime(reverb_profile=args.profile)
    calibration = examples(runtime, args.corpus, "reverb", "calibrate", args.calibrate, args.frames, 20261190)
    proposed, truth = controls(calibration, estimator)
    candidates = []
    for regularization in cycle["calibration"]["regularization"]:
        report = evaluate(runtime, calibration, proposed, float(regularization))
        candidate = {"regularization": regularization, "rank": rank(report), "report": report, "gates": gates(report)}
        candidates.append(candidate)
        print(json.dumps(candidate), flush=True)
    selected = max(candidates, key=lambda row: row["rank"])
    regularization = float(selected["regularization"])

    valid = examples(runtime, args.corpus, "reverb", "valid", args.valid, args.frames, 20261191)
    proposed, truth = controls(valid, estimator)
    report = evaluate(runtime, valid, proposed, regularization)
    checks = gates(report)
    oracle_report = evaluate(runtime, valid, truth, regularization)
    oracle_checks = gates(oracle_report)
    start = time.perf_counter()
    for row, control in zip(valid[:16], proposed[:16], strict=True):
        restore(runtime.reverb, row["source"].numpy(), control, regularization)
    seconds = time.perf_counter() - start
    audio_seconds = 16 * args.frames / runtime.reverb.sample_rate
    benchmark = {"buffers": 16, "frames": args.frames, "wall_seconds": seconds, "audio_seconds": audio_seconds, "real_time_factor": seconds / audio_seconds, "device": "cpu", "threads": args.threads}
    accepted = all(checks.values()) and all(oracle_checks.values())
    if accepted:
        shutil.copy2(args.drive, args.output / "drive.pt")
        shutil.copy2(args.drive_control, args.output / "drive.joblib")
        shutil.copy2(args.reverb_control, args.output / "reverb.joblib")
        shutil.copy2(args.profile, args.output / "reverb.npz")
    result = {
        "schema": 1,
        "status": "accepted-blind-development" if accepted else "rejected-blind",
        "accepted": accepted,
        "selection": {"candidates": candidates, "selected_regularization": regularization},
        "reports": {"reverb": report, "reverb_oracle": oracle_report},
        "gates": {"reverb": checks, "reverb_oracle": oracle_checks},
        "controls": {"mae": np.mean(np.abs(proposed - truth), axis=0).tolist()},
        "benchmark": benchmark,
        "artifacts": {name: {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for name, path in ({"drive": args.output / "drive.pt", "drive_control": args.output / "drive.joblib", "reverb_control": args.output / "reverb.joblib", "reverb_profile": args.output / "reverb.npz"}.items() if accepted else [])},
        "training": cycle["training"],
        "limitations": ["synthetic forward domain only; real paired pseudo-label and seal are unopened", "Reverb analytic inverse assumes an LTI profile selected for the detected effect family", "absolute Drive level belongs to an explicit clean profile"],
        "data": cycle["data"],
        "quality": cycle["quality"],
        "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest(),
    }
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if accepted:
        model = {"schema": 1, "id": "clean", "version": "0.6.0-blind", "status": "accepted-blind-development", "registry": {"drive": {"processor": "neural", "inverse": "drive.pt", "control": "drive.joblib", "strength": 0.9}, "reverb": {"processor": "wiener", "profile": "reverb.npz", "control": "reverb.joblib", "regularization": regularization, "feature": "base/1", "context_frames": 512}}, "executor": {"policy": "reverse forward order", "fixed_stage_order": False}, "quality": cycle["quality"]}
        (args.output / "model.json").write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": accepted, "regularization": regularization, "reverb": report, "gates": checks, "oracle_gates": oracle_checks, "benchmark": benchmark, "output": str(args.output)}), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
