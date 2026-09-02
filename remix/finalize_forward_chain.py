#!/usr/bin/env python3
"""Accept the frozen full-chain Remix phase only when every gate is intact."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/full-chain-phase1"


def _load(name: str) -> dict:
    return json.loads((RUN / name).read_text())


def main() -> None:
    architecture = _load("architecture-freeze.json")
    oracle = _load("oracle-validation.json")
    runtime = _load("runtime-contract.json")
    closed = _load("closed-loop-validation.json")
    challenge = _load("challenge-validation.json")
    metrics = _load("metrics.json")
    gates = {
        "architecture_frozen_before_challenge_metrics": architecture["status"] == "frozen-before-challenge-metrics",
        "challenge_metrics_unobserved_before_freeze": not architecture["challenge_metrics_observed_before_freeze"],
        "oracle_all_domains_passed": oracle["status"] == "passed",
        "oracle_covers_all_15_ordered_topologies": bool(oracle["all_15_nonempty_ordered_topologies"]),
        "runtime_passed": bool(runtime["passed"]),
        "runtime_bypass_bit_exact": runtime["bypass_max_absolute_error"] == 0.0,
        "runtime_silence_bit_exact": runtime["silence_max_absolute_output"] == 0.0,
        "runtime_streaming_parity": all(value["max_absolute_error"] <= 2.0e-6 for value in runtime["parity"].values()),
        "closed_loop_passed": bool(closed["aggregate"]["passed"]),
        "challenge_passed": challenge["status"] == "passed",
        "challenge_scored_after_freeze": bool(challenge["architecture_and_gates_frozen_before_metrics_observed"]),
        "challenge_not_used_for_selection": not challenge["result_used_for_architecture_selection"],
        "artifacts_immutable": bool(metrics["model_artifacts_immutable_during_audit"]),
        "sources_read_only": bool(metrics["source_files_read_only"]),
        "analysis_copy_resampling_only": bool(metrics["analysis_copy_resampling_only"]),
        "no_automatic_normalization": not metrics["automatic_normalization"],
        "no_automatic_limiting": not metrics["automatic_limiting"],
        "no_lossy_reencoding": not metrics["lossy_reencoding"],
        "locked_tele_test_closed": not metrics["locked_tele_test_opened"],
        "no_physical_audio_devices": not metrics["physical_audio_devices_used"],
    }
    accepted = all(gates.values())
    report = {
        "schema": 1,
        "status": "accepted-research-candidate" if accepted else "rejected",
        "accepted": accepted,
        "model_family": "ordered-remix-forward-chain",
        "artifact_hashes": metrics["artifacts_after"],
        "gates": gates,
        "scope": {
            "ready_for_native_runtime_prototyping": accepted,
            "ready_for_ui_integration": False,
            "real_hardware_fidelity_claim_allowed": False,
            "requires_aligned_reverb_device_calibration": True,
            "locked_tele_test_opened": False,
            "physical_audio_devices_used": False,
        },
    }
    (RUN / "acceptance.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
