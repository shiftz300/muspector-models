#!/usr/bin/env python3
"""Accept the frozen Delay inverse Phase 2 candidate on parameter and audio gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/delay-inverse-phase2"


def _load(name: str) -> dict:
    return json.loads((RUN / name).read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(report: dict, metric: str) -> float:
    values = report["validation"].values()
    return sum(domain[metric] for domain in values) / len(report["validation"])


def main() -> None:
    training = _load("metrics.json")
    baseline = _load("baseline-closed-loop-long.json")
    candidate = _load("candidate-closed-loop-long.json")
    challenge = _load("challenge-closed-loop.json")
    checkpoint = Path(training["candidate_checkpoint"])
    checkpoint_hash = _sha256(checkpoint)
    candidate_esr_ratio = _mean(candidate, "closed_loop_esr") / _mean(
        baseline, "closed_loop_esr"
    )
    candidate_mae_ratio = _mean(candidate, "closed_loop_mae") / _mean(
        baseline, "closed_loop_mae"
    )
    per_domain_esr_ratios = {
        domain: metrics["closed_loop_esr"]
        / baseline["validation"][domain]["closed_loop_esr"]
        for domain, metrics in candidate["validation"].items()
    }
    per_domain_mae_ratios = {
        domain: metrics["closed_loop_mae"]
        / baseline["validation"][domain]["closed_loop_mae"]
        for domain, metrics in candidate["validation"].items()
    }
    gates = {
        "training_candidate_promotable": bool(training["promotable"]),
        "training_checkpoint_hash_matches": (
            training["candidate_checkpoint_sha256"] == checkpoint_hash
        ),
        "training_mean_single_parameter_error_improved": (
            training["mean_single_validation_ratio"] <= 0.92
        ),
        "training_mean_mixed_parameter_error_improved": (
            training["mean_mixed_validation_ratio"] <= 1.0
        ),
        "training_kept_challenge_closed": not training["policy"][
            "challenge_renderer_opened"
        ],
        "training_kept_locked_test_closed": not training["policy"][
            "locked_tele_test_opened"
        ],
        "training_used_no_audio_devices": not training["policy"][
            "physical_audio_devices_used"
        ],
        "baseline_long_validation_passed": bool(baseline["passed"]),
        "candidate_long_validation_passed": bool(candidate["passed"]),
        "candidate_model_artifacts_immutable": bool(
            candidate["model_artifacts_immutable_during_audit"]
        ),
        "candidate_checkpoint_hash_matches": (
            candidate["delay_checkpoint_override_sha256_before"]
            == candidate["delay_checkpoint_override_sha256_after"]
            == checkpoint_hash
        ),
        "candidate_mean_closed_loop_esr_improved": candidate_esr_ratio < 1.0,
        "candidate_mean_closed_loop_mae_improved": candidate_mae_ratio < 1.0,
        "candidate_no_domain_esr_regression": max(per_domain_esr_ratios.values()) <= 1.0,
        "candidate_no_domain_mae_regression": max(per_domain_mae_ratios.values()) <= 1.0,
        "candidate_sources_read_only": bool(candidate["source_files_read_only"]),
        "candidate_used_no_runtime_normalization": not candidate[
            "runtime_automatic_normalization"
        ],
        "candidate_kept_locked_test_closed": not candidate["locked_tele_test_opened"],
        "candidate_used_no_audio_devices": not candidate["physical_audio_devices_used"],
        "challenge_passed": bool(challenge["passed"]),
        "challenge_was_opened_after_freeze": bool(challenge["challenge_renderer_opened"]),
        "challenge_not_used_for_training": not challenge["challenge_result_used_for_training"],
        "challenge_checkpoint_hash_matches": (
            challenge["delay_checkpoint_override_sha256_before"]
            == challenge["delay_checkpoint_override_sha256_after"]
            == checkpoint_hash
        ),
        "challenge_model_artifacts_immutable": bool(
            challenge["model_artifacts_immutable_during_audit"]
        ),
        "challenge_kept_locked_test_closed": not challenge["locked_tele_test_opened"],
        "challenge_used_no_audio_devices": not challenge["physical_audio_devices_used"],
    }
    promotion_path = RUN / "promotion.json"
    if promotion_path.is_file():
        promotion = _load("promotion.json")
        canonical = Path(promotion["promoted_checkpoint"])
        rollback = Path(promotion["rollback_checkpoint"])
        bundle_manifest = json.loads(
            (ROOT / "remix/runs/order-control-paired-public/bundle-manifest.json").read_text()
        )
        canonical_metrics = json.loads(
            (ROOT / "remix/runs/order-control-paired-public/metrics.json").read_text()
        )
        runtime_smoke = json.loads(
            (ROOT / "remix/runs/order-control-paired-public/runtime-smoke.json").read_text()
        )
        bundle = ROOT / "remix/runs/order-control-paired-public/remixer-bundle.pt"
        gates.update(
            {
                "promotion_completed": promotion["status"] == "promoted",
                "promoted_checkpoint_hash_matches": (
                    _sha256(canonical) == checkpoint_hash
                ),
                "rollback_checkpoint_hash_matches": (
                    _sha256(rollback) == promotion["source_checkpoint_sha256"]
                ),
                "canonical_metrics_hash_matches": (
                    canonical_metrics["delay_checkpoint_sha256"] == checkpoint_hash
                ),
                "bundle_delay_hash_matches": (
                    bundle_manifest["sources"]["delay"]["sha256"] == checkpoint_hash
                ),
                "bundle_artifact_hash_matches": (
                    bundle_manifest["artifact_sha256"] == _sha256(bundle)
                ),
                "runtime_smoke_bundle_hash_matches": (
                    runtime_smoke["bundle_sha256"] == _sha256(bundle)
                ),
            }
        )
    accepted = all(gates.values())
    report = {
        "schema": 1,
        "status": "accepted-research-candidate" if accepted else "rejected",
        "accepted": accepted,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "gates": gates,
        "comparison": {
            "mean_closed_loop_esr_ratio": candidate_esr_ratio,
            "mean_closed_loop_mae_ratio": candidate_mae_ratio,
            "per_domain_closed_loop_esr_ratio": per_domain_esr_ratios,
            "per_domain_closed_loop_mae_ratio": per_domain_mae_ratios,
        },
        "scope": {
            "ready_for_bundle_promotion": accepted,
            "ready_for_ui_integration": False,
            "real_hardware_fidelity_claim_allowed": False,
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
