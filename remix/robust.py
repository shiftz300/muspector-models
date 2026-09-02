"""Make the Reverb inverse robust to Stage-9 wet-only control errors."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import torch

from .blind import matrix
from .forward_chain import ForwardChainRuntime
from .oracle import evaluate, examples, gates
from .refine import load, save, train
from .stages import CORPUS, sha256


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "cycles/stage10.json"
MODELS = ROOT / "runs/stage/model8"
CONTROLS = ROOT / "runs/stage/model9"
RUN = ROOT / "runs/stage/model10"


def propose(rows: list[dict], estimator) -> list[dict]:
    features, _ = matrix(rows)
    predicted = np.clip(estimator.predict(features), 0.0, 1.0).astype(np.float32)
    return [{**row, "controls": torch.from_numpy(control)} for row, control in zip(rows, predicted, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, default=CYCLE); parser.add_argument("--corpus", type=Path, default=CORPUS); parser.add_argument("--models", type=Path, default=MODELS); parser.add_argument("--controls", type=Path, default=CONTROLS); parser.add_argument("--output", type=Path, default=RUN); parser.add_argument("--train", type=int, default=640); parser.add_argument("--calibrate", type=int, default=160); parser.add_argument("--valid", type=int, default=240); parser.add_argument("--epochs", type=int, default=14); parser.add_argument("--batch", type=int, default=8); parser.add_argument("--rate", type=float, default=8e-5); parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to replace robust run {args.output}")
    cycle_bytes = args.cycle.read_bytes(); cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "stage" or cycle.get("round") != 10 or cycle.get("status") != "planned": raise ValueError("invalid stage10 cycle")
    paths = {"drive": args.models / "drive.pt", "reverb": args.models / "reverb.pt", "drive_control": args.controls / "drive.candidate.joblib", "reverb_control": args.controls / "reverb.candidate.joblib"}
    if any(sha256(path) != cycle["baseline"][name + "_sha256"] for name, path in paths.items()): raise ValueError("stage10 baseline changed")
    args.output.mkdir(parents=True); torch.manual_seed(20261050); np.random.seed(20261050); torch.set_num_threads(args.threads)
    target = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); runtime = ForwardChainRuntime(); estimator = joblib.load(paths["reverb_control"]); frames = 16384
    raw_fit = examples(runtime, args.corpus, "reverb", "train", args.train, frames, 20261150)
    rng = np.random.default_rng(20261151); spread = np.asarray(cycle["training"]["control_noise_std"], dtype=np.float32)
    noisy = [{**row, "controls": torch.from_numpy(np.clip(row["controls"].numpy() + rng.normal(0.0, spread), 0.0, 1.0).astype(np.float32))} for row in raw_fit]
    calibration_truth = examples(runtime, args.corpus, "reverb", "calibrate", args.calibrate, frames, 20261152)
    calibration = propose(calibration_truth, estimator)
    source = paths["reverb"]; candidate, history = train("reverb", load(source).to(target), [*raw_fit, *noisy], calibration, target, args.epochs, args.batch, args.rate)
    valid_truth = examples(runtime, args.corpus, "reverb", "valid", args.valid, frames * 2, 20261153)
    blind_report = evaluate(candidate, propose(valid_truth, estimator), target, 1.0); blind_gates = gates(blind_report)
    oracle_report = evaluate(candidate, valid_truth, target, 1.0); oracle_gates = gates(oracle_report)
    stage9 = json.loads((args.controls / "valid.json").read_text()); drive_report = stage9["reports"]["drive"]; drive_gates = stage9["gates"]["drive"]
    (args.output / "reverb.history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
    reverb_candidate = args.output / "reverb.candidate.pt"; save(candidate.cpu(), source, reverb_candidate)
    accepted = all(drive_gates.values()) and all(blind_gates.values()) and all(oracle_gates.values())
    if accepted:
        shutil.copy2(paths["drive"], args.output / "drive.pt"); reverb_candidate.rename(args.output / "reverb.pt")
        shutil.copy2(paths["drive_control"], args.output / "drive.joblib"); shutil.copy2(paths["reverb_control"], args.output / "reverb.joblib")
    result = {"schema": 1, "status": "accepted-blind-development" if accepted else "rejected-blind", "accepted": accepted, "reports": {"drive": drive_report, "reverb": blind_report, "reverb_oracle": oracle_report}, "gates": {"drive": drive_gates, "reverb": blind_gates, "reverb_oracle": oracle_gates}, "artifacts": {"reverb_candidate": {"path": "reverb.pt" if accepted else "reverb.candidate.pt", "sha256": sha256(args.output / ("reverb.pt" if accepted else "reverb.candidate.pt"))}}, "training": {"trained": ["reverb_inverse"], "frozen": ["drive_inverse", "drive_control", "reverb_control"], "control_noise_std": spread.tolist(), "shared_optimizer": False, "chain_gradient": False}, "limitations": ["synthetic forward domain only; real paired pseudo-label and seal are still unopened", "absolute Drive level belongs to an explicit clean profile"], "data": {"source_read_only": True, "physical_audio_devices_used": False, "rendered_audio_retained": False}, "quality": cycle["quality"], "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest()}
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps({"accepted": accepted, "blind": blind_report, "blind_gates": blind_gates, "oracle": oracle_report, "oracle_gates": oracle_gates, "output": str(args.output)}), flush=True)
    if not accepted: raise SystemExit(2)


if __name__ == "__main__": main()
