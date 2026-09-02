#!/usr/bin/env python3
"""Freeze the peak-aware CS-3 residual-adapter experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(workspace: Path) -> dict:
    training_path = workspace / "remix/runs/asrnn-cs3-peak-adapter-phase5/training.json"
    if not training_path.is_file():
        raise ValueError(f"missing Phase-5 training evidence: {training_path}")
    training = json.loads(training_path.read_text())
    initial = training["history"][0]["calibrate"]
    selected = training["calibration"]
    return {
        "schema": 1,
        "phase": "phase-5-cs3-peak-aware-residual-adapter",
        "status": "rejected-on-calibrate-official-eval-unopened",
        "phase_5_evaluation_complete": True,
        "admitted_for_official_eval": False,
        "official_eval_opened": False,
        "device": "cs3",
        "architecture": {
            "base": "frozen stable 4x8 conditioned LSTM",
            "adapter": "12-state bias-free causal GRU residual",
            "features": "dry/base/difference plus audio-multiplied controls",
            "zero_input_safe_by_construction": True,
        },
        "data": {
            "fit_files": training["fit_files"],
            "fit_peak_windows": training["fit_windows"],
            "calibrate_files": training["calibrate_files"],
            "split_rule": "take-id modulo 5; no official-eval selection",
        },
        "initial_calibration": initial,
        "selected_calibration": selected,
        "peak_error_p95_improvement": (
            initial["absolute_peak_error_p95"]
            - selected["absolute_peak_error_p95"]
        ),
        "failed_gate": "absolute_peak_error_p95 <= 0.02",
        "runtime": training["runtime"],
        "decision": (
            "reject post-stable residual correction; it preserves silence and streaming "
            "but does not materially repair cross-performance peaks. Future work must "
            "train the recurrent forward model itself with peak-aware objectives and an "
            "independent locked-final capture pack."
        ),
        "checkpoint_retained": False,
        "quality_contract": training["quality_policy"],
        "ui_integration_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("remix/runs/asrnn-cs3-phase5-summary.json"),
    )
    args = parser.parse_args()
    report = summarize(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
