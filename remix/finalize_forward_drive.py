#!/usr/bin/env python3
"""Verify every frozen Drive candidate artifact before accepting it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/drive-forward-pilot"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / "acceptance.json"

    checkpoint = args.run / "drive-forward-candidate.pt"
    onnx = args.run / "drive-forward-candidate.onnx"
    training = _load(args.run / "metrics.json")
    extended = _load(args.run / "extended-validation.json")
    amplitude = _load(args.run / "amplitude-validation.json")
    challenge = _load(args.run / "challenge-validation.json")
    contract = _load(args.run / "runtime-contract.json")
    checkpoint_hash = _sha256(checkpoint)
    onnx_hash = _sha256(onnx)

    gates = {
        "training_candidate_promotable": bool(training["promotable_to_bundle"]),
        "training_checkpoint_hash_matches": training["checkpoint_sha256"] == checkpoint_hash,
        "training_excluded_challenge": not training["data_policy"]["challenge_renderer_opened"],
        "training_excluded_locked_test": not training["data_policy"]["locked_tele_test_opened"],
        "training_used_no_real_hardware": not training["data_policy"]["real_hardware_capture"],
        "extended_validation_passed": bool(extended["passed"]),
        "extended_used_no_audio_devices": not extended["physical_audio_devices_used"],
        "amplitude_validation_passed": bool(amplitude["passed"]),
        "amplitude_used_no_audio_devices": not amplitude["physical_audio_devices_used"],
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
        "runtime_contract_passed": contract["status"] == "passed",
        "runtime_parity_passed": bool(contract["parity"]["passed"]),
        "runtime_checkpoint_hash_matches": contract["checkpoint_sha256"] == checkpoint_hash,
        "runtime_onnx_hash_matches": contract["onnx_sha256"] == onnx_hash,
        "runtime_silence_is_exact_zero": (
            contract["parity"]["silence_max_absolute_output"] == 0.0
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
        "model_family": "drive-forward-causal-lstm",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "onnx": str(onnx),
        "onnx_sha256": onnx_hash,
        "gates": gates,
        "scope": {
            "synthetic_renderer_domains": ["reference", "alternate", "stress", "challenge"],
            "external_evaluation_domain": "pedalboard",
            "real_hardware_fidelity_claim_allowed": False,
            "ready_for_runtime_prototyping": accepted,
            "ready_for_ui_integration": False,
            "locked_tele_test_opened": False,
            "physical_audio_devices_used": False,
        },
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
