"""Train a compact long-context wet-only Reverb control estimator."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from .features import context
from .forward_chain import ForwardChainRuntime
from .oracle import evaluate, examples, gates
from .refine import load
from .stages import CORPUS, sha256


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "cycles/stage12.json"
DRIVE = ROOT / "runs/stage/model8/drive.pt"
DRIVE_CONTROL = ROOT / "runs/stage/model9/drive.candidate.joblib"
REVERB = ROOT / "runs/stage/model10/reverb.candidate.pt"
RUN = ROOT / "runs/stage/model12"


def matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([context(row["source"].numpy()) for row in rows]),
        np.stack([row["controls"].numpy() for row in rows]),
    )


def propose(rows: list[dict], estimator) -> tuple[list[dict], np.ndarray, np.ndarray]:
    features, truth = matrix(rows)
    predicted = np.clip(estimator.predict(features), 0.0, 1.0).astype(np.float32)
    proposed = [
        {**row, "controls": torch.from_numpy(control)}
        for row, control in zip(rows, predicted, strict=True)
    ]
    return proposed, predicted, truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, default=CYCLE)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--drive", type=Path, default=DRIVE)
    parser.add_argument("--drive-control", type=Path, default=DRIVE_CONTROL)
    parser.add_argument("--reverb", type=Path, default=REVERB)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--train", type=int, default=1600)
    parser.add_argument("--calibrate", type=int, default=192)
    parser.add_argument("--valid", type=int, default=320)
    parser.add_argument("--frames", type=int, default=65536)
    parser.add_argument("--trees", type=int, default=192)
    parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace context run {args.output}")
    cycle_bytes = args.cycle.read_bytes()
    cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "stage" or cycle.get("round") != 12 or cycle.get("status") != "planned":
        raise ValueError("invalid stage12 cycle")
    paths = {"drive": args.drive, "drive_control": args.drive_control, "reverb": args.reverb}
    for name, path in paths.items():
        if sha256(path) != cycle["baseline"][name + "_sha256"]:
            raise ValueError(f"{name} baseline changed")

    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    target = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    runtime = ForwardChainRuntime()
    fit = examples(runtime, args.corpus, "reverb", "train", args.train, args.frames, 20261170)
    x, y = matrix(fit)
    del fit
    estimator = ExtraTreesRegressor(
        n_estimators=args.trees,
        max_depth=22,
        min_samples_leaf=2,
        max_features=0.85,
        n_jobs=args.threads,
        random_state=20261171,
    ).fit(x, y)
    net = load(args.reverb).to(target)
    calibration = examples(runtime, args.corpus, "reverb", "calibrate", args.calibrate, args.frames, 20261172)
    proposed, predicted, truth = propose(calibration, estimator)
    calibration_report = evaluate(net, proposed, target, 1.0)
    calibration_gates = gates(calibration_report)
    calibration_mae = np.mean(np.abs(predicted - truth), axis=0)
    print(json.dumps({"calibration": calibration_report, "gates": calibration_gates, "control_mae": calibration_mae.tolist()}), flush=True)

    valid = examples(runtime, args.corpus, "reverb", "valid", args.valid, args.frames, 20261173)
    proposed, predicted, truth = propose(valid, estimator)
    blind_report = evaluate(net, proposed, target, 1.0)
    blind_gates = gates(blind_report)
    oracle_report = evaluate(net, valid, target, 1.0)
    oracle_gates = gates(oracle_report)
    mae = np.mean(np.abs(predicted - truth), axis=0)
    accepted = all(blind_gates.values()) and all(oracle_gates.values())
    candidate = args.output / "reverb.candidate.joblib"
    joblib.dump(estimator, candidate, compress=3)
    if accepted:
        shutil.copy2(args.drive, args.output / "drive.pt")
        shutil.copy2(args.drive_control, args.output / "drive.joblib")
        shutil.copy2(args.reverb, args.output / "reverb.pt")
        candidate.rename(args.output / "reverb.joblib")
    control_path = args.output / ("reverb.joblib" if accepted else "reverb.candidate.joblib")
    result = {
        "schema": 1,
        "status": "accepted-blind-development" if accepted else "rejected-blind",
        "accepted": accepted,
        "reports": {"calibration": calibration_report, "reverb": blind_report, "reverb_oracle": oracle_report},
        "gates": {"calibration": calibration_gates, "reverb": blind_gates, "reverb_oracle": oracle_gates},
        "controls": {"mae": mae.tolist(), "mean": float(mae.mean()), "calibration_mae": calibration_mae.tolist(), "feature": "context/1", "context_frames": args.frames},
        "artifacts": {"reverb_control": {"path": control_path.name, "sha256": sha256(control_path), "bytes": control_path.stat().st_size}},
        "training": cycle["training"],
        "limitations": ["synthetic forward domain only; real paired pseudo-label and seal are unopened", "absolute Drive level belongs to an explicit clean profile"],
        "data": cycle["data"],
        "quality": cycle["quality"],
        "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest(),
    }
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if accepted:
        model = {"schema": 1, "id": "clean", "version": "0.5.0-blind", "status": "accepted-blind-development", "registry": {"drive": {"inverse": "drive.pt", "control": "drive.joblib", "strength": 0.9}, "reverb": {"inverse": "reverb.pt", "control": "reverb.joblib", "strength": 1.0, "feature": "context/1", "context_frames": args.frames}}, "executor": {"policy": "reverse forward order", "fixed_stage_order": False}, "physical_audio_devices_used": False}
        (args.output / "model.json").write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": accepted, "reverb": blind_report, "gates": blind_gates, "oracle_gates": oracle_gates, "control_mae": mae.tolist(), "output": str(args.output)}), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
