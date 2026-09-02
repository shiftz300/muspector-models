"""Replay a frozen wet-only gate against an accepted clean restorer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from .asrnn_data import rat_files
from .clean import accepted, evaluate
from .confidence import metrics, rows
from .net import SpectralNet


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/asrnn-physical-effects"))
    parser.add_argument("--cycle", type=Path, default=Path("cycles/clean12.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/clean/model11/clean.pt"))
    parser.add_argument("--gate", type=Path, default=Path("runs/clean/model5/gate.joblib"))
    parser.add_argument("--output", type=Path, default=Path("runs/clean/model11/select.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace wet-only selection report {args.output}")
    cycle_bytes = args.cycle.read_bytes(); cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "clean1" or cycle.get("round") != 12 or cycle.get("status") != "planned":
        raise ValueError("invalid clean12 cycle")
    if digest(args.checkpoint) != cycle["model"]["restorer_sha256"] or digest(args.gate) != cycle["model"]["gate_sha256"]:
        raise ValueError("frozen wet-only artifact hash differs")
    paths = rat_files(args.corpus, "eval")
    matrix, truth = rows(paths, cycle["eligibility"]["minimum_wet_to_dry_rms_ratio"])
    gate = joblib.load(args.gate)
    probability = gate.predict_proba(matrix)[:, 1]
    selected = probability >= cycle["model"]["threshold"]
    confidence = metrics(truth, probability, cycle["model"]["threshold"])
    chosen = [path for path, keep in zip(paths, selected) if keep]
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = SpectralNet(payload["channels"], payload["n_fft"], payload["hop"])
    model.load_state_dict(payload["state_dict"]); model.eval()
    restoration = evaluate(model, chosen, torch.device("cpu"), cycle["model"]["strength"])["metrics"]
    restoration["coverage"] = confidence["coverage"]
    confidence_failures = []
    for name, value in cycle["gates"]["confidence"].items():
        metric = name.removesuffix("_minimum").removesuffix("_maximum")
        if name.endswith("_minimum") and confidence[metric] < value: confidence_failures.append(metric)
        if name.endswith("_maximum") and confidence[metric] > value: confidence_failures.append(metric)
    restoration_passed, restoration_failures = accepted(restoration, cycle["gates"]["restoration"])
    passed = not confidence_failures and restoration_passed
    report = {
        "schema": 1,
        "status": "accepted-wet-only-development" if passed else "rejected",
        "accepted": passed,
        "failures": {"confidence": confidence_failures, "restoration": restoration_failures},
        "confidence": confidence,
        "restoration": restoration,
        "artifacts": {"restorer": digest(args.checkpoint), "gate": digest(args.gate)},
        "strength": cycle["model"]["strength"],
        "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest(),
        "data": {"source_read_only": True, "physical_audio_devices_used": False, "license": "CC-BY-NC-4.0"},
        "quality": {"source_audio_modified": False, "automatic_normalization": False, "automatic_limiting": False, "automatic_dither": False, "lossy_reencoding": False},
        "limitations": ["ASRNN development gate", "confidence gate is not yet a native client artifact", "EGFx OOD confidence remains unsealed"],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
