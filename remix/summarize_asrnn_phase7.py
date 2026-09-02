#!/usr/bin/env python3
"""Freeze Phase-7 CS-3 robustness and unpaired-guitar evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def summarize(workspace: Path) -> dict:
    run = workspace / "remix/runs/asrnn-cs3-phase7"
    training = _read(run / "training.json")
    groups = _read(workspace / "remix/runs/asrnn-cs3-phase7-group-audit.json")
    diagnosis = _read(workspace / "remix/runs/asrnn-cs3-phase7-base-peak-diagnosis.json")
    source = _read(workspace / "remix/runs/guitarset-source-audit.json")
    challenge = _read(run / "metrics.json")
    ood = _read(run / "guitarset-ood.json")
    if training is None or groups is None or diagnosis is None or not source or not source["passed"]:
        raise ValueError("Phase-7 training, split audit, and peak diagnosis are required")
    admitted = bool(training["admitted_for_development_challenge"])
    if ood is None or (admitted and challenge is None):
        raise ValueError("Phase-7 requires OOD evidence and any admitted challenge evaluation")
    if not admitted and challenge is not None:
        raise ValueError("rejected calibration must not be reported as challenge admission")
    if ood["checkpoint_sha256"] != training["checkpoint_sha256"]:
        raise ValueError("OOD and calibration evidence refer to different checkpoints")
    if ood["archive_md5"] != source["archive_md5"]:
        raise ValueError("OOD and source audit refer to different archives")
    if bool(ood["candidate_diagnostic_only"]) == admitted:
        raise ValueError("OOD diagnostic-only label disagrees with calibration admission")
    accepted = bool(
        admitted and challenge["accepted"] and ood["passed"]
    )
    checkpoint = run / "phase7-centered-stable-effect.pt"
    if checkpoint.is_file() != accepted:
        raise ValueError("Phase-7 checkpoint retention differs from final admission")
    selected = training["calibration"]
    initial = training["history"][0]["calibration"]
    return {
        "schema": 1,
        "phase": "phase-7-cross-guitar-transient-robustness",
        "phase_7_evaluation_complete": True,
        "status": (
            "accepted-internal-development-pilot"
            if accepted
            else "rejected-on-development-challenge-or-ood"
            if admitted
            else "rejected-on-coverage-calibration"
        ),
        "accepted": accepted,
        "calibration_admitted": admitted,
        "candidate_development_challenge_evaluated": challenge is not None,
        "development_challenge_opened_for_base_diagnosis": True,
        "development_challenge_accepted": bool(challenge and challenge["accepted"]),
        "guitarset_candidate_ood_evaluated": ood is not None,
        "guitarset_candidate_ood_passed": bool(ood and ood["passed"]),
        "architecture": {
            "runtime": "shared-weight zero-centered stable 4x8 conditioned LSTM",
            "training": "full three-second CS-3 clips with causal truncated backpropagation",
            "loss": "waveform/preemphasis/envelope/peak/two-resolution-STFT plus top-25-percent CVaR",
            "regularization": "stable-base anchor and recurrent infinity-norm projection <=0.995",
            "selection": "coverage calibration and worst-Attack peak gate",
        },
        "data": {
            "fit_files": training["fit_files"],
            "calibration_files": training["calibration_files"],
            "train_unique_dry_hashes": groups.get("train_unique_dry_hashes"),
            "train_clip_frames": groups.get("train_clip_frames"),
            "fit_calibration_exact_dry_duplicates": groups.get(
                "fit_calibration_exact_dry_duplicates"
            ),
            "cross_split_exact_dry_duplicates": groups["cross_split_exact_dry_duplicates"],
            "fit_calibration_near_performance_pairs": groups["fit_calibration_near_performance_pairs"],
            "cross_split_near_performance_pairs": groups["cross_split_near_performance_pairs"],
            "train_near_performance_pairs": groups["train_near_performance_pairs"],
            "near_performance_rule": groups["near_performance_rule"],
            "selection": groups["calibration_selection"],
            "take_number_is_performance_identity": False,
            "calibration_partition_changed_since_phase6": True,
            "phase6_calibration_metrics_directly_comparable": False,
        },
        "initial_calibration": initial,
        "selected_calibration": selected,
        "calibration_peak_p95_improvement": (
            initial["absolute_peak_error_p95"] - selected["absolute_peak_error_p95"]
        ),
        "calibration_worst_attack_peak_p95_improvement": (
            initial["worst_attack_absolute_peak_error_p95"]
            - selected["worst_attack_absolute_peak_error_p95"]
        ),
        "baseline_challenge_diagnosis": {
            "examples": diagnosis["examples"],
            "over_peak_gate_files": diagnosis["absolute_peak_error_over_002_count"],
            "absolute_peak_error_p95": diagnosis["absolute_peak_error_p95"],
            "correlations": diagnosis["peak_error_correlations"],
        },
        "development_challenge": challenge["official_eval"] if challenge else None,
        "guitarset_ood": ood["metrics"] if ood else None,
        "external_audio": {
            "dataset": "GuitarSet v1.1.0 mono pickup mix",
            "archive": "data/downloads/guitarset-audio-mono-pickup-mix.zip",
            "bytes": source["archive_bytes"],
            "md5": source["archive_md5"],
            "files": source["files"],
            "duration_hours": source["duration_hours"],
            "source_full_scale_hit_samples": source.get("full_scale_hit_samples"),
            "license": "CC-BY-4.0",
            "used_for_training": False,
            "has_cs3_wet_truth": False,
            "can_support_pedal_fidelity_claim": False,
        },
        "runtime": training["runtime"],
        "training_configuration": training.get("configuration"),
        "compute_device": training.get("compute_device"),
        "candidate_checkpoint_sha256": training["checkpoint_sha256"],
        "base_checkpoint_sha256": ood["base_checkpoint_sha256"],
        "checkpoint_retained": checkpoint.is_file(),
        "quality_contract": training["quality_policy"],
        "new_locked_final_audio_opened": False,
        "ui_integration_allowed": False,
        "release_limitations": [
            "ASRNN model remains CC-BY-NC-4.0 and development-only",
            "official ASRNN eval is a development challenge, not locked-final",
            "unpaired public guitar audio cannot replace paired physical-effect captures",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("remix/runs/asrnn-cs3-phase7-summary.json")
    )
    args = parser.parse_args()
    report = summarize(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
