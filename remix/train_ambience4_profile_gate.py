#!/usr/bin/env python3
"""Train a tiny Wet/profile-only safety gate for Product4 Reverb v4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ambience5_profile_direct import AmbiencePairsV4MixedProfileDirect
from .audit_ambience4_profile_direct_gate import (
    OBSERVABLE_FEATURE_NAMES,
    _deploy,
    _rows,
)
from .train_ambience4_profile_direct import SEED


class ProfileSafetyGate(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(OBSERVABLE_FEATURE_NAMES), width),
            nn.GELU(),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


def _threshold(rows: list[dict], scores: np.ndarray) -> float:
    """Freeze the largest zero-false-positive calibration prefix."""
    order = np.argsort(-scores)
    selected = 0
    for rank, index in enumerate(order):
        if not rows[int(index)]["candidate_effective"]:
            break
        selected = rank + 1
    if not selected:
        return float("inf")
    last = float(scores[order[selected - 1]])
    if selected == len(order):
        return last
    return 0.5 * (last + float(scores[order[selected]]))


def _deploy_scores(rows: list[dict], scores: np.ndarray, threshold: float) -> dict:
    tagged = []
    for row, score in zip(rows, scores, strict=True):
        # Reuse the fully audited deployment summarizer with one synthetic key.
        tagged.append({**row, "analytic_mode": "learned", "score": -float(score)})
    thresholds = {
        f"learned:{row['decay_stratum']}": -threshold for row in rows
    }
    return _deploy(tagged, thresholds)


def _rank(report: dict) -> tuple[float, ...]:
    groups = list(report["groups"].values())
    return (
        float(report["accepted"]),
        min((row["effective_eligible_coverage"] for row in groups), default=-1.0),
        report["aggregate"]["effective_eligible_coverage"],
        report["aggregate"]["effective_examples"],
    )


def _fast_rank(rows: list[dict], scores: np.ndarray, threshold: float) -> tuple[float, ...]:
    """Rank checkpoints from cached labels; run expensive audio metrics only once."""
    selected = scores >= threshold
    safe = all(
        not bool(chosen) or row["candidate_effective"]
        for row, chosen in zip(rows, selected, strict=True)
    )
    coverages = []
    for axis in ("source_id", "room_group", "decay_stratum"):
        for value in sorted({row[axis] for row in rows}):
            indices = [index for index, row in enumerate(rows) if row[axis] == value]
            measurable = sum(rows[index]["candidate_result"]["evidence_measurable"] for index in indices)
            effective = sum(
                bool(selected[index]) and rows[index]["candidate_effective"]
                for index in indices
            )
            coverages.append(effective / max(measurable, 1))
    measurable = sum(row["candidate_result"]["evidence_measurable"] for row in rows)
    effective = sum(
        bool(chosen) and row["candidate_effective"]
        for row, chosen in zip(rows, selected, strict=True)
    )
    aggregate = effective / max(measurable, 1)
    accepted = safe and aggregate >= 0.50 and min(coverages, default=0.0) >= 0.50
    return (float(accepted), float(safe), min(coverages, default=0.0), aggregate, effective)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/foundation/product4-reverb-profile-gate-v4-calibration-v1"),
    )
    parser.add_argument("--fit-samples", type=int, default=192)
    parser.add_argument("--calibration-samples", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--width", type=int, default=16)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace profile gate run: {output}")
    if min(args.fit_samples, args.calibration_samples) < 96:
        raise ValueError("profile gate needs at least 96 fit and calibration rows")
    torch.set_num_threads(1)
    workspace = args.workspace.resolve()
    fit_dataset = AmbiencePairsV4MixedProfileDirect(
        workspace, "fit", args.fit_samples, 65_536, SEED + 1
    )
    calibration_dataset = AmbiencePairsV4MixedProfileDirect(
        workspace, "calibration", args.calibration_samples, 65_536, SEED + 2
    )
    fit_rows = _rows(fit_dataset)
    calibration_rows = _rows(calibration_dataset)
    calibration_sources = {row["source_id"] for row in calibration_rows}
    training_rows = [row for row in fit_rows if row["source_id"] in calibration_sources]
    fit_x = np.stack([row["observable_features"] for row in training_rows])
    calibration_x = np.stack([row["observable_features"] for row in calibration_rows])
    fit_y = np.asarray([row["candidate_effective"] for row in training_rows], dtype=np.float32)
    mean = fit_x.mean(axis=0)
    standard_deviation = fit_x.std(axis=0)
    standard_deviation = np.maximum(standard_deviation, 1.0e-5)
    fit_tensor = torch.from_numpy((fit_x - mean) / standard_deviation)
    calibration_tensor = torch.from_numpy((calibration_x - mean) / standard_deviation)
    labels = torch.from_numpy(fit_y)
    positive_weight = torch.tensor(float((1.0 - fit_y).sum() / max(fit_y.sum(), 1.0)))
    candidates = []
    checkpoints = {
        min(args.epochs, epoch) for epoch in (25, 50, 100, 200, 400, args.epochs)
    }
    for seed_offset in range(4):
        torch.manual_seed(SEED + seed_offset)
        model = ProfileSafetyGate(args.width)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-3)
        history = []
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(fit_tensor)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=positive_weight
            )
            loss.backward()
            optimizer.step()
            if epoch in {0, args.epochs - 1}:
                history.append({"epoch": epoch + 1, "fit_bce": float(loss.detach())})
            if epoch + 1 in checkpoints:
                model.eval()
                with torch.inference_mode():
                    training_scores = model(fit_tensor).numpy()
                    calibration_scores = model(calibration_tensor).numpy()
                threshold = _threshold(calibration_rows, calibration_scores)
                state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
                candidates.append((
                    _fast_rank(calibration_rows, calibration_scores, threshold),
                    seed_offset, epoch + 1, state, threshold,
                    training_scores.copy(), calibration_scores.copy(), list(history),
                ))
                model.train()
    rank, seed_offset, selected_epoch, state, threshold, training_scores, calibration_scores, history = max(
        candidates, key=lambda row: (row[0], -row[1], -row[2])
    )
    fit_report = _deploy_scores(training_rows, training_scores, threshold)
    calibration_report = _deploy_scores(calibration_rows, calibration_scores, threshold)
    model = ProfileSafetyGate(args.width)
    model.load_state_dict(state)
    output.mkdir(parents=True)
    checkpoint = output / "gate.pt"
    torch.save({
        "schema": 1,
        "feature_names": OBSERVABLE_FEATURE_NAMES,
        "feature_mean": torch.from_numpy(mean),
        "feature_standard_deviation": torch.from_numpy(standard_deviation),
        "logit_threshold": threshold,
        "width": args.width,
        "state_dict": model.state_dict(),
    }, checkpoint)
    report = {
        "schema": 1,
        "status": (
            "calibration-gate-passed-development-sealed"
            if calibration_report["accepted"] else "calibration-gate-rejected"
        ),
        "accepted": False,
        "calibration_gate_passed": calibration_report["accepted"],
        "deployable_gate": True,
        "model": {
            "architecture": "observable-profile-safety-mlp",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "width": args.width,
            "checkpoint": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "logit_threshold": threshold,
            "feature_names": OBSERVABLE_FEATURE_NAMES,
        },
        "training": {
            "accelerator": "cpu",
            "seed": SEED + seed_offset,
            "seed_candidates": 4,
            "epochs": args.epochs,
            "selected_epoch": selected_epoch,
            "fit_samples": args.fit_samples,
            "fit_samples_used_for_gate": len(training_rows),
            "fit_source_filter": sorted(calibration_sources),
            "calibration_samples": args.calibration_samples,
            "selection_partition": "calibration",
            "selection_rule": "zero false positives then maximize per-group and aggregate effective coverage",
            "selected_rank": list(rank),
            "history": history,
        },
        "fit": fit_report,
        "calibration": calibration_report,
        "development_opened": False,
        "listening_rows_used": False,
        "fresh_rochester_opened": False,
        "locked_final_accessed": False,
        "inference_inputs": "Wet, analytic output, current RIR/profile and current controls only",
        "clean_input_at_runtime": False,
        "chain_order_input": False,
        "graph_order_input": False,
        "neighbor_effect_input": False,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "selected_seed": report["training"]["seed"],
        "threshold": threshold,
        "fit": fit_report["aggregate"],
        "calibration": calibration_report["aggregate"],
        "failed_groups": [
            name for name, row in calibration_report["groups"].items()
            if not row["accepted"]
        ],
        "metrics": str(output / "metrics.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
