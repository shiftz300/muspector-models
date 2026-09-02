"""Refine wet-only Reverb controls by inverse-forward replay near a proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import joblib
import numpy as np
import torch

from .blind import matrix
from .forward_chain import ForwardChainRuntime
from .oracle import evaluate, examples, gates
from .refine import load, rank
from .spec import ChainSpec, Reverb
from .stages import CORPUS, sha256


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "cycles/stage11.json"
DRIVE = ROOT / "runs/stage/model8/drive.pt"
REVERB = ROOT / "runs/stage/model10/reverb.candidate.pt"
CONTROLS = ROOT / "runs/stage/model9"
RUN = ROOT / "runs/stage/model11"


def decode(control: np.ndarray) -> Reverb:
    return Reverb(0.2 * 40.0 ** float(control[0]), float(control[1]), float(control[2]) * 0.7)


@torch.inference_mode()
def options(net, runtime: ForwardChainRuntime, wet: np.ndarray, proposal: np.ndarray, target: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = [proposal.copy()]
    for index, amount in enumerate((0.15, 0.15, 0.20)):
        for sign in (-1.0, 1.0):
            candidate = proposal.copy(); candidate[index] = np.clip(candidate[index] + sign * amount, 0.0, 1.0); values.append(candidate)
    controls = np.asarray(values, dtype=np.float32)
    source = torch.from_numpy(wet).to(target).unsqueeze(0).expand(len(controls), -1)
    restored = net(source, torch.from_numpy(controls).to(target), strength=1.0).cpu().numpy()
    energy = max(float(np.mean(np.square(wet, dtype=np.float64))), 1.0e-12)
    replay = []
    for clean, control in zip(restored, controls, strict=True):
        rendered = runtime.render(clean, ChainSpec((decode(control),)))
        replay.append(float(np.mean(np.square(rendered - wet, dtype=np.float64)) / energy))
    distance = np.mean(np.square(controls - proposal[None], dtype=np.float64), axis=1)
    return controls, np.asarray(replay), distance


def choose(rows: list[dict], proposals: np.ndarray, searches: list[tuple[np.ndarray, np.ndarray, np.ndarray]], penalty: float) -> list[dict]:
    result = []
    for row, proposal, (controls, replay, distance) in zip(rows, proposals, searches, strict=True):
        index = int(np.argmin(replay + penalty * distance))
        result.append({**row, "controls": torch.from_numpy(controls[index].copy()), "proposal": proposal.tolist(), "replay_error": float(replay[index])})
    return result


def search(rows: list[dict], estimator, net, runtime: ForwardChainRuntime, target: torch.device) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    features, _ = matrix(rows); proposals = np.clip(estimator.predict(features), 0.0, 1.0).astype(np.float32)
    searches = [options(net, runtime, row["source"].numpy(), proposal, target) for row, proposal in zip(rows, proposals, strict=True)]
    return proposals, searches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, default=CYCLE); parser.add_argument("--corpus", type=Path, default=CORPUS); parser.add_argument("--drive", type=Path, default=DRIVE); parser.add_argument("--reverb", type=Path, default=REVERB); parser.add_argument("--controls", type=Path, default=CONTROLS); parser.add_argument("--output", type=Path, default=RUN); parser.add_argument("--calibrate", type=int, default=64); parser.add_argument("--valid", type=int, default=96); parser.add_argument("--frames", type=int, default=16384); parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to replace replay run {args.output}")
    cycle_bytes = args.cycle.read_bytes(); cycle = json.loads(cycle_bytes); controller = args.controls / "reverb.candidate.joblib"
    if cycle.get("id") != "stage" or cycle.get("round") != 11 or cycle.get("status") != "planned": raise ValueError("invalid stage11 cycle")
    for name, path in (("drive", args.drive), ("reverb", args.reverb), ("reverb_control", controller)):
        if sha256(path) != cycle["baseline"][name + "_sha256"]: raise ValueError(f"{name} baseline changed")
    args.output.mkdir(parents=True); torch.set_num_threads(args.threads); target = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    runtime = ForwardChainRuntime(); net = load(args.reverb).to(target); estimator = joblib.load(controller)
    calibration = examples(runtime, args.corpus, "reverb", "calibrate", args.calibrate, args.frames, 20261160)
    proposals, searches = search(calibration, estimator, net, runtime, target)
    candidates = []
    for penalty in (0.0, 0.01, 0.05, 0.10, 0.25):
        report = evaluate(net, choose(calibration, proposals, searches, penalty), target, 1.0)
        candidates.append({"penalty": penalty, "rank": rank(report), "report": report, "gates": gates(report)})
        print(json.dumps(candidates[-1]), flush=True)
    selected = max(candidates, key=lambda row: row["rank"]); penalty = selected["penalty"]
    valid = examples(runtime, args.corpus, "reverb", "valid", args.valid, args.frames * 2, 20261161)
    proposals, searches = search(valid, estimator, net, runtime, target)
    chosen = choose(valid, proposals, searches, penalty); reverb_report = evaluate(net, chosen, target, 1.0); reverb_gates = gates(reverb_report)
    stage9 = json.loads((args.controls / "valid.json").read_text()); drive_report = stage9["reports"]["drive"]; drive_gates = stage9["gates"]["drive"]
    accepted = all(drive_gates.values()) and all(reverb_gates.values())
    if accepted:
        shutil.copy2(args.drive, args.output / "drive.pt"); shutil.copy2(args.reverb, args.output / "reverb.pt")
        shutil.copy2(args.controls / "drive.candidate.joblib", args.output / "drive.joblib"); shutil.copy2(controller, args.output / "reverb.joblib")
    result = {"schema": 1, "status": "accepted-blind-development" if accepted else "rejected-blind", "accepted": accepted, "search": {"candidates_per_record": 7, "penalty_candidates": candidates, "selected_penalty": penalty, "uses_clean_reference": False}, "reports": {"drive": drive_report, "reverb": reverb_report}, "gates": {"drive": drive_gates, "reverb": reverb_gates}, "artifacts": {"drive_sha256": sha256(args.drive), "reverb_sha256": sha256(args.reverb), "reverb_control_sha256": sha256(controller)}, "limitations": ["synthetic forward domain only; real pseudo-label and seal are unopened", "absolute Drive level belongs to an explicit clean profile"], "data": {"source_read_only": True, "physical_audio_devices_used": False, "rendered_audio_retained": False}, "quality": cycle["quality"], "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest()}
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if accepted:
        model = {"schema": 1, "id": "clean", "version": "0.4.0-blind", "status": "accepted-blind-development", "registry": {"drive": {"inverse": "drive.pt", "control": "drive.joblib", "strength": 0.9}, "reverb": {"inverse": "reverb.pt", "control": "reverb.joblib", "strength": 1.0, "search_candidates": 7, "search_penalty": penalty}}, "executor": {"policy": "reverse forward order", "fixed_stage_order": False}, "physical_audio_devices_used": False}
        (args.output / "model.json").write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": accepted, "penalty": penalty, "reverb": reverb_report, "gates": reverb_gates, "output": str(args.output)}), flush=True)
    if not accepted: raise SystemExit(2)


if __name__ == "__main__": main()
