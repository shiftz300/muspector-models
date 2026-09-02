"""Refine only the Stage-6 Drive candidate's peak and tail behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .forward_chain import ForwardChainRuntime
from .oracle import evaluate, examples, gates
from .refine import load, save, train
from .stages import CORPUS, sha256


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "cycles/stage7.json"
BASELINE = ROOT / "runs/stage/model6"
RUN = ROOT / "runs/stage/model7"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, default=CYCLE); parser.add_argument("--corpus", type=Path, default=CORPUS); parser.add_argument("--baseline", type=Path, default=BASELINE); parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--train", type=int, default=640); parser.add_argument("--calibrate", type=int, default=128); parser.add_argument("--valid", type=int, default=160); parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--batch", type=int, default=8); parser.add_argument("--rate", type=float, default=1e-5); parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to replace peak run {args.output}")
    cycle_bytes = args.cycle.read_bytes(); cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "stage" or cycle.get("round") != 7 or cycle.get("status") != "planned": raise ValueError("invalid stage7 cycle")
    drive_source = args.baseline / "drive.candidate.pt"; reverb_source = args.baseline / "reverb.candidate.pt"
    if sha256(drive_source) != cycle["baseline"]["drive_sha256"] or sha256(reverb_source) != cycle["baseline"]["reverb_sha256"]: raise ValueError("stage6 candidate changed")
    args.output.mkdir(parents=True); torch.manual_seed(20261040); np.random.seed(20261040); torch.set_num_threads(args.threads)
    target = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); runtime = ForwardChainRuntime(); frames = 8192
    fit = examples(runtime, args.corpus, "drive", "train", args.train, frames, 20261100, "tone")
    calibration = examples(runtime, args.corpus, "drive", "calibrate", args.calibrate, frames, 20261101, "tone")
    drive, history = train("drive", load(drive_source).to(target), fit, calibration, target, args.epochs, args.batch, args.rate)
    valid = examples(runtime, args.corpus, "drive", "valid", args.valid, frames * 2, 20261102, "tone")
    drive_report = evaluate(drive, valid, target); drive_gates = gates(drive_report)
    stage6 = json.loads((args.baseline / "valid.json").read_text()); reverb_report = stage6["reports"]["reverb"]; reverb_gates = stage6["gates"]["reverb"]
    (args.output / "drive.history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
    drive_candidate = args.output / "drive.candidate.pt"; save(drive.cpu(), drive_source, drive_candidate)
    accepted = all(drive_gates.values()) and all(reverb_gates.values())
    if accepted:
        drive_path = args.output / "drive.pt"; drive_candidate.rename(drive_path)
        reverb_path = args.output / "reverb.pt"; shutil.copy2(reverb_source, reverb_path)
    artifacts = {
        "drive": {"path": "drive.pt" if accepted else "drive.candidate.pt", "sha256": sha256(args.output / ("drive.pt" if accepted else "drive.candidate.pt"))},
        "reverb": {"path": "reverb.pt" if accepted else str(reverb_source), "sha256": sha256(args.output / "reverb.pt") if accepted else sha256(reverb_source), "frozen": True},
    }
    result = {"schema": 1, "status": "accepted-oracle" if accepted else "rejected-oracle", "accepted": accepted, "reports": {"drive": drive_report, "reverb": reverb_report}, "gates": {"drive": drive_gates, "reverb": reverb_gates}, "artifacts": artifacts, "training": {"trained": ["drive"], "frozen": ["reverb"], "loss": "aligned-hard-tail-peak", "shared_optimizer": False, "chain_gradient": False}, "data": {"source_read_only": True, "physical_audio_devices_used": False, "rendered_audio_retained": False}, "quality": cycle["quality"], "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest()}
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps({"accepted": accepted, "drive": drive_report, "gates": drive_gates, "output": str(args.output)}), flush=True)
    if not accepted: raise SystemExit(2)


if __name__ == "__main__": main()
