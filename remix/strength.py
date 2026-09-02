"""Calibrate Drive restoration strength after Stage-7 peak-tail refinement."""

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
from .refine import load, rank
from .stages import CORPUS, sha256


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "cycles/stage8.json"
DRIVE = ROOT / "runs/stage/model7/drive.candidate.pt"
REVERB = ROOT / "runs/stage/model6/reverb.candidate.pt"
RUN = ROOT / "runs/stage/model8"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, default=CYCLE); parser.add_argument("--corpus", type=Path, default=CORPUS); parser.add_argument("--drive", type=Path, default=DRIVE); parser.add_argument("--reverb", type=Path, default=REVERB); parser.add_argument("--output", type=Path, default=RUN); parser.add_argument("--calibrate", type=int, default=160); parser.add_argument("--valid", type=int, default=192); parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to replace strength run {args.output}")
    cycle_bytes = args.cycle.read_bytes(); cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "stage" or cycle.get("round") != 8 or cycle.get("status") != "planned": raise ValueError("invalid stage8 cycle")
    if sha256(args.drive) != cycle["baseline"]["drive_sha256"] or sha256(args.reverb) != cycle["baseline"]["reverb_sha256"]: raise ValueError("stage candidate changed")
    args.output.mkdir(parents=True); torch.set_num_threads(args.threads); target = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    runtime = ForwardChainRuntime(); net = load(args.drive).to(target); frames = 8192
    calibration = examples(runtime, args.corpus, "drive", "calibrate", args.calibrate, frames, 20261110, "tone")
    candidates = []
    for strength in np.linspace(0.40, 1.00, 13):
        report = evaluate(net, calibration, target, float(strength)); check = gates(report)
        candidates.append({"strength": float(strength), "rank": rank(report), "report": report, "gates": check})
        print(json.dumps({"strength": float(strength), "rank": rank(report), "gates": check, "report": report}), flush=True)
    selected = max(candidates, key=lambda row: row["rank"]); strength = selected["strength"]
    valid = examples(runtime, args.corpus, "drive", "valid", args.valid, frames * 2, 20261111, "tone")
    drive_report = evaluate(net, valid, target, strength); drive_gates = gates(drive_report)
    stage6 = json.loads((ROOT / "runs/stage/model6/valid.json").read_text()); reverb_report = stage6["reports"]["reverb"]; reverb_gates = stage6["gates"]["reverb"]
    accepted = all(drive_gates.values()) and all(reverb_gates.values())
    if accepted:
        shutil.copy2(args.drive, args.output / "drive.pt"); shutil.copy2(args.reverb, args.output / "reverb.pt")
    result = {"schema": 1, "status": "accepted-oracle" if accepted else "rejected-oracle", "accepted": accepted, "strengths": {"drive": strength, "reverb": 1.0}, "selection": {"split": "calibrate", "candidates": candidates, "selected": strength}, "reports": {"drive": drive_report, "reverb": reverb_report}, "gates": {"drive": drive_gates, "reverb": reverb_gates}, "artifacts": {"drive": {"source": str(args.drive), "sha256": sha256(args.output / "drive.pt") if accepted else sha256(args.drive)}, "reverb": {"source": str(args.reverb), "sha256": sha256(args.output / "reverb.pt") if accepted else sha256(args.reverb)}}, "training": {"weights_changed": False, "calibration_only": True, "shared_optimizer": False, "chain_gradient": False}, "data": {"source_read_only": True, "physical_audio_devices_used": False, "rendered_audio_retained": False}, "quality": cycle["quality"], "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest()}
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if accepted:
        manifest = {"schema": 1, "id": "clean", "version": "0.3.0-oracle", "status": "accepted-oracle", "registry": {"drive": {"path": "drive.pt", "sha256": sha256(args.output / "drive.pt"), "strength": strength}, "reverb": {"path": "reverb.pt", "sha256": sha256(args.output / "reverb.pt"), "strength": 1.0}}, "executor": {"policy": "reverse forward order", "fixed_stage_order": False}, "physical_audio_devices_used": False}
        (args.output / "model.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": accepted, "selected_strength": strength, "drive": drive_report, "gates": drive_gates, "output": str(args.output)}), flush=True)
    if not accepted: raise SystemExit(2)


if __name__ == "__main__": main()
