#!/usr/bin/env python3
"""Verify every frozen hybrid Delay artifact before accepting it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/delay-forward-pilot"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    checkpoint = args.run / "delay-forward-candidate.pt"
    parameters = args.run / "delay-forward-parameters.json"
    training = _load(args.run / "metrics.json")
    extended = _load(args.run / "extended-validation.json")
    challenge = _load(args.run / "challenge-validation.json")
    closed_loop = _load(args.run / "closed-loop-validation.json")
    closed_loop_challenge = _load(args.run / "closed-loop-challenge.json")
    contract = _load(args.run / "runtime-contract.json")
    checkpoint_hash = _sha256(checkpoint)
    parameter_hash = _sha256(parameters)
    inverse_bundle = Path(closed_loop["inverse_bundle"])
    inverse_hash = _sha256(inverse_bundle)
    gates = {
        "training_candidate_promotable": bool(training["promotable"]),
        "training_checkpoint_hash_matches": training["checkpoint_sha256"] == checkpoint_hash,
        "training_kept_challenge_closed": not training["policy"]["challenge_renderer_opened"],
        "training_kept_locked_test_closed": not training["policy"]["locked_tele_test_opened"],
        "training_used_no_audio_devices": not training["policy"]["physical_audio_devices_used"],
        "time_is_exact_physical_delay": bool(training["policy"]["time_is_exact_physical_delay"]),
        "feedback_is_exact": bool(training["policy"]["feedback_is_exact"]),
        "mix_is_exact": bool(training["policy"]["mix_is_exact"]),
        "extended_validation_passed": bool(extended["passed"]),
        "extended_timing_is_sample_exact": extended["invariants"]["worst_timing_error_samples"] == 0,
        "extended_bypass_is_exact": extended["invariants"]["bypass_max_absolute_error"] == 0.0,
        "extended_silence_is_exact": extended["invariants"]["silence_max_absolute_output"] == 0.0,
        "extended_used_no_audio_devices": not extended["physical_audio_devices_used"],
        "challenge_passed": bool(challenge["passed"]),
        "challenge_checkpoint_immutable": bool(challenge["checkpoint_immutable_during_audit"]),
        "challenge_checkpoint_hash_matches": (
            challenge["checkpoint_sha256_before_challenge"]
            == challenge["checkpoint_sha256_after_challenge"]
            == checkpoint_hash
        ),
        "challenge_not_used_for_training": not challenge["challenge_result_used_for_training"],
        "challenge_kept_locked_test_closed": not challenge["locked_tele_test_opened"],
        "challenge_used_no_audio_devices": not challenge["physical_audio_devices_used"],
        "closed_loop_validation_passed": bool(closed_loop["passed"]),
        "closed_loop_model_artifacts_immutable": bool(
            closed_loop["model_artifacts_immutable_during_audit"]
        ),
        "closed_loop_forward_hash_matches": (
            closed_loop["forward_checkpoint_sha256_before"]
            == closed_loop["forward_checkpoint_sha256_after"]
            == checkpoint_hash
        ),
        "closed_loop_inverse_hash_matches": (
            closed_loop["inverse_bundle_sha256_before"]
            == closed_loop["inverse_bundle_sha256_after"]
            == inverse_hash
        ),
        "closed_loop_sources_read_only": bool(closed_loop["source_files_read_only"]),
        "closed_loop_resampled_analysis_copies_only": bool(
            closed_loop["analysis_copy_resampling_only"]
        ),
        "closed_loop_used_no_runtime_normalization": not closed_loop[
            "runtime_automatic_normalization"
        ],
        "closed_loop_kept_challenge_closed": not closed_loop[
            "challenge_renderer_opened"
        ],
        "closed_loop_kept_locked_test_closed": not closed_loop[
            "locked_tele_test_opened"
        ],
        "closed_loop_used_no_audio_devices": not closed_loop[
            "physical_audio_devices_used"
        ],
        "closed_loop_challenge_passed": bool(closed_loop_challenge["passed"]),
        "closed_loop_challenge_model_artifacts_immutable": bool(
            closed_loop_challenge["model_artifacts_immutable_during_audit"]
        ),
        "closed_loop_challenge_forward_hash_matches": (
            closed_loop_challenge["forward_checkpoint_sha256_before"]
            == closed_loop_challenge["forward_checkpoint_sha256_after"]
            == checkpoint_hash
        ),
        "closed_loop_challenge_inverse_hash_matches": (
            closed_loop_challenge["inverse_bundle_sha256_before"]
            == closed_loop_challenge["inverse_bundle_sha256_after"]
            == inverse_hash
        ),
        "closed_loop_challenge_was_opened": bool(
            closed_loop_challenge["challenge_renderer_opened"]
        ),
        "closed_loop_challenge_not_used_for_training": not closed_loop_challenge[
            "challenge_result_used_for_training"
        ],
        "closed_loop_challenge_sources_read_only": bool(
            closed_loop_challenge["source_files_read_only"]
        ),
        "closed_loop_challenge_resampled_analysis_copies_only": bool(
            closed_loop_challenge["analysis_copy_resampling_only"]
        ),
        "closed_loop_challenge_used_no_runtime_normalization": not closed_loop_challenge[
            "runtime_automatic_normalization"
        ],
        "closed_loop_challenge_kept_locked_test_closed": not closed_loop_challenge[
            "locked_tele_test_opened"
        ],
        "closed_loop_challenge_used_no_audio_devices": not closed_loop_challenge[
            "physical_audio_devices_used"
        ],
        "runtime_contract_passed": contract["status"] == "passed",
        "runtime_checkpoint_hash_matches": contract["checkpoint_sha256"] == checkpoint_hash,
        "runtime_parameter_hash_matches": contract["parameters_sha256"] == parameter_hash,
        "runtime_bypass_is_exact": contract["bypass_max_absolute_error"] == 0.0,
        "runtime_silence_is_exact": contract["silence_max_absolute_output"] == 0.0,
        "runtime_streaming_parity_passed": all(
            values["max_absolute_error"] <= 2.0e-6 for values in contract["parity"].values()
        ),
        "runtime_used_no_audio_devices": not contract["audio_quality_policy"][
            "physical_audio_devices_used"
        ],
    }
    accepted = all(gates.values())
    report = {
        "schema": 1,
        "status": "accepted-research-candidate" if accepted else "rejected",
        "accepted": accepted,
        "model_family": "delay-forward-hybrid",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "parameters": str(parameters),
        "parameters_sha256": parameter_hash,
        "gates": gates,
        "scope": {
            "real_hardware_fidelity_claim_allowed": False,
            "ready_for_native_runtime_prototyping": accepted,
            "ready_for_ui_integration": False,
            "locked_tele_test_opened": False,
            "physical_audio_devices_used": False,
        },
    }
    (args.run / "acceptance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
