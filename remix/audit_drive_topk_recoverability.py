#!/usr/bin/env python3
"""Frozen Graybox Drive Top-K Clean recoverability audit.

This does not estimate controls, train a selector, or promote a model.  It asks
three separate questions on the frozen repository-owned synthetic audit:

1. Can the accepted inverse recover Clean when given the true controls?
2. Does Wet-only inverse-forward replay put any acceptable Clean in Top-K?
3. Is replay Top-1 already an acceptable selector?

Truth and Clean are used only after ranking to score candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .audit_drive_control_identifiability import (
    CHUNK,
    CROP,
    TOP_K,
    candidate_grid,
    clean_example,
    normalized_controls,
    render_batch,
    truth_controls,
)
from .foundation_expert_runtime import FoundationExpertRuntime
from .quality2 import measure


EXAMPLES = 12
COVERAGE_K = (1, 3, 5, 12, 24, 48, 324)
THRESHOLDS = {
    "true_control_pass_fraction_minimum": 0.80,
    "all_grid_oracle_pass_fraction_minimum": 0.90,
    "top_k_oracle_pass_fraction_minimum": 0.80,
    "replay_top1_pass_fraction_minimum": 0.80,
}


def _fraction(rows: list[dict], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows]))


def summarize_rows(rows: list[dict]) -> dict:
    """Apply the frozen route decision without inspecting audio or controls."""
    if not rows:
        raise ValueError("recoverability audit requires at least one example")
    metrics = {
        "examples": len(rows),
        "true_control_pass_fraction": _fraction(rows, "true_control_passed"),
        "all_grid_oracle_pass_fraction": _fraction(rows, "all_grid_has_pass"),
        "top_k_oracle_pass_fraction": _fraction(rows, "top_k_has_pass"),
        "replay_top1_pass_fraction": _fraction(rows, "top1_passed"),
        "median_first_passing_replay_rank": None,
    }
    ranks = [row["first_passing_replay_rank"] for row in rows]
    finite_ranks = [rank for rank in ranks if rank is not None]
    if finite_ranks:
        metrics["median_first_passing_replay_rank"] = float(np.median(finite_ranks))
    gates = {
        "true_control_capacity": metrics["true_control_pass_fraction"]
        >= THRESHOLDS["true_control_pass_fraction_minimum"],
        "all_grid_oracle_capacity": metrics["all_grid_oracle_pass_fraction"]
        >= THRESHOLDS["all_grid_oracle_pass_fraction_minimum"],
        "top_k_recoverability": metrics["top_k_oracle_pass_fraction"]
        >= THRESHOLDS["top_k_oracle_pass_fraction_minimum"],
        "replay_top1_selector": metrics["replay_top1_pass_fraction"]
        >= THRESHOLDS["replay_top1_pass_fraction_minimum"],
    }
    capacity_viable = gates["true_control_capacity"] and gates["all_grid_oracle_capacity"]
    top_k_viable = capacity_viable and gates["top_k_recoverability"]
    selector_viable = top_k_viable and gates["replay_top1_selector"]
    if selector_viable:
        status = "development-graybox-topk-and-selector-pass-not-promoted"
    elif top_k_viable:
        status = "development-graybox-topk-recoverable-selector-needed"
    elif capacity_viable:
        status = "rejected-replay-ranking"
    else:
        status = "rejected-graybox-capacity"
    return {
        "status": status,
        "capacity_viable": capacity_viable,
        "top_k_viable": top_k_viable,
        "selector_viable": selector_viable,
        "metrics": metrics,
        "gates": gates,
    }


def _clean_nmse(restored: np.ndarray, clean: np.ndarray) -> np.ndarray:
    selection = slice(CROP, -CROP)
    denominator = float(np.mean(np.square(clean[selection]))) + 1.0e-12
    return np.mean(
        np.square(restored[:, selection] - clean[None, selection]), axis=1
    ) / denominator


def _quality_summary(row: dict) -> dict:
    return {
        "passed": bool(row["passed"]),
        "metric_reductions": {
            name: float(metric["reduction"])
            for name, metric in row["metrics"].items()
        },
        "failed_gates": sorted(name for name, passed in row["gates"].items() if not passed),
        "correction_to_wet_distance": float(row["correction_to_wet_distance"]),
        "restored_peak": float(row["restored_peak"]),
    }


def audit(checkpoint: Path, device: str = "mps") -> dict:
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the frozen Drive Top-K audit")
    runtime = FoundationExpertRuntime("nonlinear", checkpoint)
    assert runtime.model is not None
    runtime.model = runtime.model.to(torch.device(device))
    candidates = candidate_grid()
    normalized = normalized_controls(candidates)
    rows = []
    coverage_hits = {value: 0 for value in COVERAGE_K}
    for example_index in range(EXAMPLES):
        clean = clean_example(example_index)
        truth = truth_controls(example_index)
        truth_index = candidates.index(truth)
        wet = render_batch(clean, [truth])[0]
        restored_parts = []
        for offset in range(0, len(candidates), CHUNK):
            count = min(CHUNK, len(candidates) - offset)
            wet_tensor = torch.from_numpy(wet).to(device).unsqueeze(0).expand(count, -1)
            control_tensor = torch.from_numpy(normalized[offset : offset + count]).to(device)
            with torch.inference_mode():
                restored, _, _ = runtime.model(wet_tensor, control_tensor)
            restored_parts.append(restored.cpu().numpy())
        restored = np.concatenate(restored_parts)
        replay = render_batch(restored, candidates)
        selection = slice(CROP, -CROP)
        replay_denominator = float(np.mean(np.square(wet[selection]))) + 1.0e-12
        replay_scores = np.mean(
            np.square(replay[:, selection] - wet[None, selection]), axis=1
        ) / replay_denominator
        ranking = np.argsort(replay_scores, kind="stable")
        clean_scores = _clean_nmse(restored, clean)
        quality = [
            measure("nonlinear", wet[selection], candidate[selection], clean[selection])
            for candidate in restored
        ]
        passes = np.asarray([row["passed"] for row in quality], dtype=bool)
        ranked_passes = passes[ranking]
        for value in COVERAGE_K:
            coverage_hits[value] += int(bool(np.any(ranked_passes[:value])))
        passing_ranks = np.flatnonzero(ranked_passes)
        first_passing_rank = None if not len(passing_ranks) else int(passing_ranks[0]) + 1
        chosen = ranking[:TOP_K]
        chosen_passing = chosen[passes[chosen]]
        all_passing = np.flatnonzero(passes)
        best_top_k = (
            None
            if not len(chosen_passing)
            else int(chosen_passing[np.argmin(clean_scores[chosen_passing])])
        )
        best_all = (
            None
            if not len(all_passing)
            else int(all_passing[np.argmin(clean_scores[all_passing])])
        )
        top1 = int(ranking[0])
        row = {
            "example": example_index,
            "truth": truth,
            "truth_rank": int(np.flatnonzero(ranking == truth_index)[0]) + 1,
            "true_control_passed": bool(passes[truth_index]),
            "true_control_clean_nmse": float(clean_scores[truth_index]),
            "true_control_quality": _quality_summary(quality[truth_index]),
            "top1": candidates[top1],
            "top1_passed": bool(passes[top1]),
            "top1_replay_nmse": float(replay_scores[top1]),
            "top1_clean_nmse": float(clean_scores[top1]),
            "top1_quality": _quality_summary(quality[top1]),
            "top_k_has_pass": bool(len(chosen_passing)),
            "top_k_pass_count": int(len(chosen_passing)),
            "best_top_k": None if best_top_k is None else {
                "controls": candidates[best_top_k],
                "replay_rank": int(np.flatnonzero(ranking == best_top_k)[0]) + 1,
                "replay_nmse": float(replay_scores[best_top_k]),
                "clean_nmse": float(clean_scores[best_top_k]),
                "quality": _quality_summary(quality[best_top_k]),
            },
            "all_grid_has_pass": bool(len(all_passing)),
            "all_grid_pass_count": int(len(all_passing)),
            "best_all_grid": None if best_all is None else {
                "controls": candidates[best_all],
                "replay_rank": int(np.flatnonzero(ranking == best_all)[0]) + 1,
                "replay_nmse": float(replay_scores[best_all]),
                "clean_nmse": float(clean_scores[best_all]),
                "quality": _quality_summary(quality[best_all]),
            },
            "first_passing_replay_rank": first_passing_rank,
        }
        rows.append(row)
    summary = summarize_rows(rows)
    return {
        "schema": 1,
        "status": summary["status"],
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "compute": device,
        "candidate_count": len(candidates),
        "top_k": TOP_K,
        "coverage_k": list(COVERAGE_K),
        "coverage_curve": {
            str(value): coverage_hits[value] / EXAMPLES for value in COVERAGE_K
        },
        "thresholds": THRESHOLDS,
        "summary": summary,
        "examples": rows,
        "scope": {
            "wet_only_replay_ranking": True,
            "truth_and_clean_used_for_scoring_only": True,
            "candidate_controls_are_expert_inputs": True,
            "synthetic_repository_owned_drive": True,
            "selector_trained": False,
            "training_performed": False,
            "graph_order_input": False,
            "neighbouring_effect_input": False,
            "physical_drive": False,
            "amp_included": False,
            "reverb_included": False,
            "locked_final_opened": False,
            "source_audio_modified": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("runs/foundation/product3-safe/nonlinear/model.pt"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product11-drive-topk-recoverability/audit.json"),
    )
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    report = audit(args.checkpoint.resolve(), args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "coverage_curve": report["coverage_curve"],
        "summary": report["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
