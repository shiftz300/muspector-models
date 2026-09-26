#!/usr/bin/env python3
"""Measure the non-promotable Clean-oracle ceiling of existing Amp candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .amp_data import AmpPairs
from .amp_model import AmpInverseExpert
from .amp_model2 import AmpDynamicsInverseExpert
from .amp_model3 import AmpFrameDynamicsInverseExpert
from .amp_model4 import AmpStructuredInverseExpert
from .amp_model5 import AmpStageSupervisedInverseExpert
from .rusty_amp_stage_model import RustyAmpStagewiseGrayBoxInverse
from .quality2 import REQUIRED_IMPROVEMENTS, measure, summarize
from .train_amp import SEED, _stratum


def _load(path: Path) -> tuple[str, torch.nn.Module, str]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    architecture = payload.get("architecture") or {}
    name = architecture.get("architecture")
    if name == "control-conditioned-fir-plus-local-tcn":
        model = AmpInverseExpert()
    elif name == "control-fir-plus-envelope-conditioned-multiplicative-local-tcn":
        model = AmpDynamicsInverseExpert(
            architecture.get("hidden_size", 32), architecture.get("depth", 10)
        )
    elif name == "control-fir-plus-bidirectional-frame-dynamics":
        model = AmpFrameDynamicsInverseExpert(
            architecture.get("hidden_size", 48), architecture.get("depth", 2)
        )
    elif name == "structured-output-profile-power-tone-stack-preamp-inverse":
        model = AmpStructuredInverseExpert()
    elif name == "stage-supervised-output-power-tone-preamp-input-inverse":
        model = AmpStageSupervisedInverseExpert()
    elif name == "eight-stage-product-supervised-gray-box-inverse":
        model = RustyAmpStagewiseGrayBoxInverse()
    else:
        raise ValueError(f"unsupported Amp candidate architecture: {name}")
    model.load_state_dict(payload.get("state_dict", {}), strict=True)
    model.eval()
    return str(name), model, hashlib.sha256(path.read_bytes()).hexdigest()


def _score(report: dict) -> tuple:
    required = REQUIRED_IMPROVEMENTS["amp"]
    reductions = [report["metrics"][name]["reduction"] for name in sorted(required)]
    return (
        int(report["passed"]),
        sum(bool(value) for value in report["gates"].values()),
        min(reductions),
        sum(reductions),
    )


def audit(workspace: Path, checkpoints: list[Path], samples: int, target_frames: int) -> dict:
    candidates = [_load(path.resolve()) for path in checkpoints]
    dataset = AmpPairs(workspace.resolve(), "development", samples, target_frames, SEED + 3)
    collections = {name: ([], [], []) for name, _, _ in candidates}
    oracle = ([], [], [])
    oracle_strata = defaultdict(lambda: ([], [], []))
    chosen = Counter()
    any_pass = 0
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            start = int(row["target_start"])
            wet = row["wet"][start:].numpy()
            clean = row["clean"][start:].numpy()
            ranked = []
            for name, model, _ in candidates:
                restored, _, _ = model(row["wet"].unsqueeze(0), row["controls"].unsqueeze(0))
                value = restored[0, start:].numpy().astype(np.float32)
                report = measure("amp", wet, value, clean)
                ranked.append((_score(report), name, value, report))
                for target, source in zip(collections[name], (wet, value, clean), strict=True):
                    target.append(source)
            _, name, value, report = max(ranked, key=lambda item: item[0])
            chosen[name] += 1
            any_pass += int(any(item[3]["passed"] for item in ranked))
            for target, source in zip(oracle, (wet, value, clean), strict=True):
                target.append(source)
            stratum = _stratum(row["control_values"])
            for target, source in zip(oracle_strata[stratum], (wet, value, clean), strict=True):
                target.append(source)
    oracle_report = summarize("amp", *oracle)
    oracle_report["strata"] = {
        name: summarize("amp", *values) for name, values in sorted(oracle_strata.items())
    }
    return {
        "schema": 1,
        "status": "diagnostic-clean-oracle-not-promotable",
        "purpose": "upper-bound existing-candidate complementarity before training a selector",
        "candidates": [
            {"architecture": name, "checkpoint": str(path.resolve()), "sha256": digest,
             "report": summarize("amp", *collections[name])}
            for path, (name, _, digest) in zip(checkpoints, candidates, strict=True)
        ],
        "oracle": {
            "report": oracle_report,
            "any_candidate_pass_fraction": any_pass / len(dataset),
            "chosen_counts": dict(sorted(chosen.items())),
        },
        "provenance": {
            "clean_used_for_oracle_selection": True,
            "valid_for_promotion": False,
            "selector_trained": False,
            "locked_final_audio_opened": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=135)
    parser.add_argument("--target-frames", type=int, default=16384)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    report = audit(args.workspace, args.checkpoints, args.samples, args.target_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "oracle_accepted": report["oracle"]["report"]["accepted"],
        "oracle_pass_fraction": report["oracle"]["report"]["pass_fraction"],
        "any_candidate_pass_fraction": report["oracle"]["any_candidate_pass_fraction"],
        "chosen_counts": report["oracle"]["chosen_counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
