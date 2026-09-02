#!/usr/bin/env python3
"""Verify the complete frozen Reverb forward research phase."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/reverb-forward-phase1"


def _load(name: str) -> dict:
    return json.loads((RUN / name).read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    fit = _load("metrics.json")
    validation = _load("validation.json")
    challenge = _load("challenge-validation.json")
    runtime = _load("runtime-contract.json")
    closed_loop = _load("closed-loop-validation.json")
    closed_loop_challenge = _load("closed-loop-challenge.json")
    profile = Path(fit["profile"])
    profile_hash = _sha256(profile)
    inverse_bundle = Path(closed_loop["inverse_bundle"])
    inverse_hash = _sha256(inverse_bundle)
    gates = {
        "fit_profile_hash_matches": fit["profile_sha256"] == profile_hash,
        "fit_mix_is_exact": bool(fit["mix_is_exact"]),
        "fit_kept_challenge_closed": not fit["policy"]["challenge_renderer_opened"],
        "fit_kept_locked_test_closed": not fit["policy"]["locked_tele_test_opened"],
        "fit_used_no_audio_devices": not fit["policy"]["physical_audio_devices_used"],
        "validation_passed": bool(validation["passed"]),
        "validation_kept_challenge_closed": not validation["challenge_renderer_opened"],
        "validation_bypass_is_exact": validation["invariants"]["bypass_max_absolute_error"] == 0.0,
        "validation_silence_is_exact": validation["invariants"]["silence_max_absolute_output"] == 0.0,
        "validation_used_no_audio_devices": not validation["physical_audio_devices_used"],
        "challenge_passed": bool(challenge["passed"]),
        "challenge_profile_was_fit": bool(challenge["challenge_profile_fit_from_calibration_grid"]),
        "challenge_not_used_for_architecture_selection": not challenge[
            "challenge_result_used_for_architecture_selection"
        ],
        "challenge_kept_locked_test_closed": not challenge["locked_tele_test_opened"],
        "challenge_used_no_audio_devices": not challenge["physical_audio_devices_used"],
        "runtime_passed": bool(runtime["passed"]),
        "runtime_profile_hash_matches": runtime["profile_sha256"] == profile_hash,
        "runtime_streaming_parity_passed": all(
            values["max_absolute_error"] <= 2.0e-6 for values in runtime["parity"].values()
        ),
        "runtime_bypass_is_exact": runtime["bypass_max_absolute_error"] == 0.0,
        "runtime_silence_is_exact": runtime["silence_max_absolute_output"] == 0.0,
        "runtime_used_no_audio_devices": not runtime["audio_quality_policy"][
            "physical_audio_devices_used"
        ],
        "closed_loop_passed": bool(closed_loop["passed"]),
        "closed_loop_inverse_hash_matches": (
            closed_loop["inverse_bundle_sha256_before"]
            == closed_loop["inverse_bundle_sha256_after"]
            == inverse_hash
        ),
        "closed_loop_model_artifacts_immutable": bool(
            closed_loop["model_artifacts_immutable_during_audit"]
        ),
        "closed_loop_kept_challenge_closed": not closed_loop["challenge_renderer_opened"],
        "closed_loop_kept_locked_test_closed": not closed_loop["locked_tele_test_opened"],
        "closed_loop_used_no_audio_devices": not closed_loop["physical_audio_devices_used"],
        "closed_loop_challenge_passed": bool(closed_loop_challenge["passed"]),
        "closed_loop_challenge_profile_was_fit": bool(
            closed_loop_challenge["challenge_profile_fit_from_calibration_grid"]
        ),
        "closed_loop_challenge_not_used_for_architecture_selection": not closed_loop_challenge[
            "challenge_result_used_for_architecture_selection"
        ],
        "closed_loop_challenge_inverse_hash_matches": (
            closed_loop_challenge["inverse_bundle_sha256_before"]
            == closed_loop_challenge["inverse_bundle_sha256_after"]
            == inverse_hash
        ),
        "closed_loop_challenge_kept_locked_test_closed": not closed_loop_challenge[
            "locked_tele_test_opened"
        ],
        "closed_loop_challenge_used_no_audio_devices": not closed_loop_challenge[
            "physical_audio_devices_used"
        ],
    }
    accepted = all(gates.values())
    report = {
        "schema": 1,
        "status": "accepted-research-candidate" if accepted else "rejected",
        "accepted": accepted,
        "model_family": "reverb-forward-capture-profile",
        "profile": str(profile),
        "profile_sha256": profile_hash,
        "gates": gates,
        "scope": {
            "ready_for_native_runtime_prototyping": accepted,
            "ready_for_ui_integration": False,
            "real_hardware_fidelity_claim_allowed": False,
            "requires_aligned_device_calibration": True,
            "locked_tele_test_opened": False,
            "physical_audio_devices_used": False,
        },
    }
    (RUN / "acceptance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
