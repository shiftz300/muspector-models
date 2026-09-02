"""Train wet-only control proposals and evaluate the frozen Stage-8 inverses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from .features import extract
from .forward_chain import ForwardChainRuntime
from .oracle import evaluate, examples, gates
from .refine import load
from .stages import CORPUS, sha256


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "cycles/stage9.json"
BASELINE = ROOT / "runs/stage/model8"
RUN = ROOT / "runs/stage/model9"


def matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return np.stack([extract(row["source"].numpy()) for row in rows]), np.stack([row["controls"].numpy() for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, default=CYCLE); parser.add_argument("--corpus", type=Path, default=CORPUS); parser.add_argument("--baseline", type=Path, default=BASELINE); parser.add_argument("--output", type=Path, default=RUN); parser.add_argument("--train", type=int, default=1200); parser.add_argument("--valid", type=int, default=240); parser.add_argument("--frames", type=int, default=16384); parser.add_argument("--trees", type=int, default=128); parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to replace blind run {args.output}")
    cycle_bytes = args.cycle.read_bytes(); cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "stage" or cycle.get("round") != 9 or cycle.get("status") != "planned": raise ValueError("invalid stage9 cycle")
    if sha256(args.baseline / "drive.pt") != cycle["baseline"]["drive_sha256"] or sha256(args.baseline / "reverb.pt") != cycle["baseline"]["reverb_sha256"]: raise ValueError("stage8 model changed")
    args.output.mkdir(parents=True); torch.set_num_threads(args.threads); target = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); runtime = ForwardChainRuntime()
    strengths = json.loads((args.baseline / "valid.json").read_text())["strengths"]
    reports, checks, controls, artifacts = {}, {}, {}, {}
    for offset, kind in enumerate(("drive", "reverb")):
        fit = examples(runtime, args.corpus, kind, "train", args.train, args.frames, 20261120 + offset)
        valid = examples(runtime, args.corpus, kind, "valid", args.valid, args.frames * 2, 20261130 + offset)
        x, y = matrix(fit); vx, truth = matrix(valid)
        estimator = ExtraTreesRegressor(n_estimators=args.trees, max_depth=18, min_samples_leaf=3, max_features=0.75, n_jobs=args.threads, random_state=20261140 + offset).fit(x, y)
        predicted = np.clip(estimator.predict(vx), 0.0, 1.0).astype(np.float32)
        proposed = [{**row, "controls": torch.from_numpy(control)} for row, control in zip(valid, predicted, strict=True)]
        net = load(args.baseline / f"{kind}.pt").to(target)
        report = evaluate(net, proposed, target, strengths[kind]); check = gates(report)
        oracle = evaluate(net, valid, target, strengths[kind])
        mae = np.mean(np.abs(predicted - truth), axis=0)
        path = args.output / f"{kind}.candidate.joblib"; joblib.dump(estimator, path, compress=3)
        reports[kind], checks[kind] = report, check
        controls[kind] = {"mae": mae.tolist(), "mean": float(mae.mean()), "oracle_report": oracle, "absolute_level_identifiable": False if kind == "drive" else None}
        artifacts[kind] = {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
        print(json.dumps({"stage": kind, "control": controls[kind], "report": report, "gates": check}), flush=True)
    accepted = all(all(values.values()) for values in checks.values())
    result = {"schema": 1, "status": "accepted-blind-development" if accepted else "rejected-blind", "accepted": accepted, "reports": reports, "gates": checks, "controls": controls, "artifacts": artifacts, "models": {kind: sha256(args.baseline / f"{kind}.pt") for kind in ("drive", "reverb")}, "limitations": ["wet-only absolute Drive level is not identifiable and belongs to an explicit clean profile", "synthetic forward domain only; no real seal opened"], "data": {"source_read_only": True, "physical_audio_devices_used": False, "rendered_audio_retained": False}, "quality": cycle["quality"], "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest()}
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps({"accepted": accepted, "output": str(args.output)}), flush=True)
    if not accepted: raise SystemExit(2)


if __name__ == "__main__": main()
