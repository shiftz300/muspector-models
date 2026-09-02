#!/usr/bin/env python3
"""Freeze the CS-3 zero-centered recurrent fine-tuning experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(workspace: Path) -> dict:
    run = workspace / "remix/runs/asrnn-cs3-centered-phase6"
    training_path = run / "training.json"
    if not training_path.is_file():
        raise ValueError(f"missing Phase-6 training evidence: {training_path}")
    training = json.loads(training_path.read_text())
    official_path = run / "metrics.json"
    official = json.loads(official_path.read_text()) if official_path.is_file() else None
    admitted = bool(training["admitted_for_official_eval"])
    if admitted != (official is not None):
        raise ValueError(
            "official evaluation must exist exactly when calibration admitted the model"
        )
    accepted = bool(official and official["accepted"])
    initial = training["history"][0]["calibrate"]
    selected = training["calibration"]
    status = (
        "accepted-internal-noncommercial-pilot"
        if accepted
        else "rejected-on-official-eval"
        if official is not None
        else "rejected-on-calibrate-official-eval-unopened"
    )
    failed_gates = []
    gate_limits = {
        "global_esr": (selected["global_esr"], "<=", 0.05),
        "mean_per_file_esr": (selected["mean_per_file_esr"], "<=", 0.10),
        "p95_per_file_esr": (selected["p95_per_file_esr"], "<=", 0.25),
        "absolute_peak_error_p95": (
            selected["absolute_peak_error_p95"],
            "<=",
            0.02,
        ),
        "peak_ratio_median_lower": (selected["peak_ratio_median"], ">=", 0.75),
        "peak_ratio_median_upper": (selected["peak_ratio_median"], "<=", 1.25),
        "peak_ratio_p95": (selected["peak_ratio_p95"], "<=", 1.35),
    }
    for name, (value, relation, limit) in gate_limits.items():
        passed = value <= limit if relation == "<=" else value >= limit
        if not passed:
            failed_gates.append(
                {"metric": name, "observed": value, "required": f"{relation} {limit}"}
            )
    failed_official_gates = []
    if official is not None:
        metrics = official["official_eval"]
        official_limits = {
            "global_esr": (metrics["global_esr"], "<=", 0.05),
            "global_esr_relative_improvement": (
                metrics["global_esr_relative_improvement"],
                ">=",
                0.50,
            ),
            "global_mae_relative_improvement": (
                metrics["global_mae_relative_improvement"],
                ">=",
                0.50,
            ),
            "preemphasis_esr_relative_improvement": (
                metrics["preemphasis_esr_relative_improvement"],
                ">=",
                0.50,
            ),
            "spectral_loss_relative_improvement": (
                metrics["spectral_loss_relative_improvement"],
                ">=",
                0.30,
            ),
            "mean_per_file_esr": (metrics["mean_per_file_esr"], "<=", 0.10),
            "median_per_file_esr": (metrics["median_per_file_esr"], "<=", 0.05),
            "p95_per_file_esr": (metrics["p95_per_file_esr"], "<=", 0.25),
            "peak_ratio_median_lower": (metrics["peak_ratio_median"], ">=", 0.75),
            "peak_ratio_median_upper": (metrics["peak_ratio_median"], "<=", 1.25),
            "peak_ratio_p95": (metrics["peak_ratio_p95"], "<=", 1.35),
            "absolute_peak_error_p95": (
                metrics["absolute_peak_error_p95"],
                "<=",
                0.02,
            ),
            "quiet_prediction_peak_maximum": (
                metrics["quiet_prediction_peak_maximum"],
                "<=",
                1.0e-3,
            ),
            "static_silence_max_absolute_output": (
                metrics["static_silence_max_absolute_output"],
                "<=",
                0.0,
            ),
            "dynamic_control_silence_max_absolute_output": (
                metrics["dynamic_control_silence_max_absolute_output"],
                "<=",
                0.0,
            ),
            "stream_max_absolute_error": (
                metrics["stream_max_absolute_error"],
                "<=",
                2.0e-6,
            ),
        }
        for name, (value, relation, limit) in official_limits.items():
            passed = value <= limit if relation == "<=" else value >= limit
            if not passed:
                failed_official_gates.append(
                    {
                        "metric": name,
                        "observed": value,
                        "required": f"{relation} {limit}",
                    }
                )
    return {
        "schema": 1,
        "phase": "phase-6-cs3-zero-centered-recurrent-finetune",
        "status": status,
        "phase_6_evaluation_complete": True,
        "admitted_for_official_eval": admitted,
        "official_eval_opened": official is not None,
        "official_eval_accepted": accepted,
        "device": "cs3",
        "architecture": {
            "base": "stable 4x8 conditioned LSTM initialized from verified CS-3 weights",
            "training": "full recurrent/output fine-tuning with peak-aware loss",
            "runtime": "shared-weight F(dry, controls) - F(zero, controls)",
            "candidate_recurrent_infinity_norm_maximum": 0.995,
            "zero_input_safe_by_construction": True,
        },
        "data": {
            "fit_files": training["fit_files"],
            "fit_peak_windows": training["fit_windows"],
            "calibrate_files": training["calibrate_files"],
            "split_rule": "take-id modulo 5; official eval excluded from selection",
        },
        "initial_calibration": initial,
        "selected_calibration": selected,
        "peak_error_p95_improvement": (
            initial["absolute_peak_error_p95"]
            - selected["absolute_peak_error_p95"]
        ),
        "failed_calibration_gates": failed_gates,
        "failed_official_gates": failed_official_gates,
        "runtime_validation": training["runtime"],
        "official_evaluation": official["official_eval"] if official else None,
        "checkpoint_retained": accepted,
        "quality_contract": training["quality_policy"],
        "ui_integration_allowed": accepted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("remix/runs/asrnn-cs3-phase6-summary.json"),
    )
    args = parser.parse_args()
    report = summarize(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
