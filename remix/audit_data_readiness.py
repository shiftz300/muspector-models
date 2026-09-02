#!/usr/bin/env python3
"""Produce the Phase-1 Capture Pack readiness evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .capture_kit import drive_capture_plan
from .capture_pack import audit_capture_manifest, audit_drive_plan, validate_source_registry


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def audit(workspace: Path, output: Path) -> dict:
    registry_path = workspace / "remix" / "data_sources.json"
    registry = validate_source_registry(registry_path, workspace=workspace)
    plan = drive_capture_plan(device_id="unassigned-drive", player_id="split-local-source")
    plan_audit = audit_drive_plan(plan)

    manifest_root = workspace / "data" / "manifests"
    manifests = sorted(manifest_root.glob("*.json")) if manifest_root.is_dir() else []
    manifest_audits = []
    for manifest in manifests:
        try:
            manifest_audits.append(
                {"manifest": str(manifest), "audit": audit_capture_manifest(manifest)}
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            manifest_audits.append(
                {"manifest": str(manifest), "audit": {"passed": False, "error": str(error)}}
            )

    real_packs_ready = bool(manifest_audits) and all(
        entry["audit"].get("passed", False) for entry in manifest_audits
    )
    pipeline_path = output / "capture-adapter-pipeline-smoke.json"
    pipeline = json.loads(pipeline_path.read_text()) if pipeline_path.is_file() else None
    asrnn_audit_path = workspace / "remix/runs/asrnn-rat-adapter-pilot/data-audit.json"
    asrnn_audit = (
        json.loads(asrnn_audit_path.read_text()) if asrnn_audit_path.is_file() else None
    )
    asrnn_metrics_path = workspace / "remix/runs/asrnn-rat-stable-pilot/metrics.json"
    asrnn_metrics = (
        json.loads(asrnn_metrics_path.read_text()) if asrnn_metrics_path.is_file() else None
    )
    inverse_summary_path = workspace / "remix/runs/asrnn-rat-inverse-summary.json"
    inverse_summary = (
        json.loads(inverse_summary_path.read_text())
        if inverse_summary_path.is_file()
        else None
    )
    phase4_summary_path = workspace / "remix/runs/asrnn-multi-device-phase4-summary.json"
    phase4_summary = (
        json.loads(phase4_summary_path.read_text())
        if phase4_summary_path.is_file()
        else None
    )
    phase5_summary_path = workspace / "remix/runs/asrnn-cs3-phase5-summary.json"
    phase5_summary = (
        json.loads(phase5_summary_path.read_text())
        if phase5_summary_path.is_file()
        else None
    )
    phase6_summary_path = workspace / "remix/runs/asrnn-cs3-phase6-summary.json"
    phase6_summary = (
        json.loads(phase6_summary_path.read_text())
        if phase6_summary_path.is_file()
        else None
    )
    phase7_summary_path = workspace / "remix/runs/asrnn-cs3-phase7-summary.json"
    phase7_summary = (
        json.loads(phase7_summary_path.read_text())
        if phase7_summary_path.is_file()
        else None
    )
    cs3_card_path = workspace / "remix/runs/asrnn-cs3-ensemble-pilot/model-card.json"
    cs3_card = json.loads(cs3_card_path.read_text()) if cs3_card_path.is_file() else None
    cs3_checkpoint = workspace / "remix/runs/asrnn-cs3-ensemble-pilot/cs3-stable.pt"
    cs3_graph = workspace / "remix/runs/asrnn-cs3-ensemble-pilot/cs3-stable.onnx"
    cs3_artifacts_match = bool(
        cs3_card and cs3_checkpoint.is_file()
        and hashlib.sha256(cs3_checkpoint.read_bytes()).hexdigest() == cs3_card.get("checkpoint_sha256")
        and (not cs3_card.get("onnx_sha256") or (
            cs3_graph.is_file() and hashlib.sha256(cs3_graph.read_bytes()).hexdigest() == cs3_card["onnx_sha256"]
        ))
    )
    acceptance = {
        "schema": 1,
        "phase": "capture-pack-data-readiness",
        "phase_1_complete": registry["passed"] and plan_audit["passed"],
        "phase_2_offline_pipeline_ready": bool(
            pipeline is not None and pipeline.get("pipeline_passed", False)
        ),
        "phase_2_real_device_training_ready": real_packs_ready,
        "phase_2_real_hardware_public_pilot": {
            "data_ready": bool(asrnn_audit and asrnn_audit.get("passed", False)),
            "training_completed": asrnn_metrics is not None,
            "accepted": bool(asrnn_metrics and asrnn_metrics.get("accepted", False)),
            "scope": "internal non-commercial development only",
            "metrics": str(asrnn_metrics_path) if asrnn_metrics else None,
        },
        "phase_3_inverse_control_recovery": {
            "evaluation_completed": bool(
                inverse_summary
                and inverse_summary.get("phase_3_evaluation_complete", False)
            ),
            "exact_all_controls": bool(
                inverse_summary
                and inverse_summary.get("phase_3_exact_all_controls", False)
            ),
            "precise_controls": (
                inverse_summary.get("precise_controls", []) if inverse_summary else []
            ),
            "unresolved_controls": (
                inverse_summary.get("unresolved_controls", []) if inverse_summary else []
            ),
            "scope": "internal non-commercial development only",
            "summary": str(inverse_summary_path) if inverse_summary else None,
        },
        "phase_4_multi_device_forward": {
            "evaluation_completed": bool(
                phase4_summary
                and phase4_summary.get("phase_4_evaluation_complete", False)
            ),
            "runtime_architecture_completed": bool(
                phase4_summary
                and phase4_summary.get("phase_4_runtime_architecture_complete", False)
            ),
            "accepted_devices": (
                phase4_summary.get("accepted_devices", []) if phase4_summary else []
            ),
            "rejected_devices": (
                phase4_summary.get("rejected_devices", []) if phase4_summary else []
            ),
            "summary": str(phase4_summary_path) if phase4_summary else None,
        },
        "phase_5_peak_aware_adapter": {
            "evaluation_completed": bool(
                phase5_summary
                and phase5_summary.get("phase_5_evaluation_complete", False)
            ),
            "admitted_for_official_eval": bool(
                phase5_summary
                and phase5_summary.get("admitted_for_official_eval", False)
            ),
            "official_eval_opened": bool(
                phase5_summary and phase5_summary.get("official_eval_opened", False)
            ),
            "checkpoint_retained": bool(
                phase5_summary and phase5_summary.get("checkpoint_retained", False)
            ),
            "summary": str(phase5_summary_path) if phase5_summary else None,
        },
        "phase_6_peak_aware_recurrent_finetune": {
            "evaluation_completed": bool(
                phase6_summary
                and phase6_summary.get("phase_6_evaluation_complete", False)
            ),
            "admitted_for_official_eval": bool(
                phase6_summary
                and phase6_summary.get("admitted_for_official_eval", False)
            ),
            "official_eval_opened": bool(
                phase6_summary and phase6_summary.get("official_eval_opened", False)
            ),
            "official_eval_accepted": bool(
                phase6_summary and phase6_summary.get("official_eval_accepted", False)
            ),
            "checkpoint_retained": bool(
                phase6_summary and phase6_summary.get("checkpoint_retained", False)
            ),
            "summary": str(phase6_summary_path) if phase6_summary else None,
        },
        "phase_7_cross_guitar_robustness": {
            "evaluation_completed": bool(
                phase7_summary
                and phase7_summary.get("phase_7_evaluation_complete", False)
            ),
            "accepted": bool(phase7_summary and phase7_summary.get("accepted", False)),
            "calibration_admitted": bool(
                phase7_summary and phase7_summary.get("calibration_admitted", False)
            ),
            "guitarset_candidate_ood_evaluated": bool(
                phase7_summary
                and phase7_summary.get("guitarset_candidate_ood_evaluated", False)
            ),
            "guitarset_candidate_ood_passed": bool(
                phase7_summary
                and phase7_summary.get("guitarset_candidate_ood_passed", False)
            ),
            "checkpoint_retained": bool(
                phase7_summary and phase7_summary.get("checkpoint_retained", False)
            ),
            "summary": str(phase7_summary_path) if phase7_summary else None,
        },
        "source_registry_passed": registry["passed"],
        "phase_8_cs3_capacity_upgrade": {
            "accepted": bool(cs3_card and cs3_card.get("accepted", False) and cs3_artifacts_match),
            "artifact_hashes_match": cs3_artifacts_match,
            "model_card": str(cs3_card_path) if cs3_card else None,
            "architecture": cs3_card.get("architecture") if cs3_card else None,
            "scope": "internal non-commercial development only; not independent final evidence",
            "ui_integration_allowed": False,
            "new_locked_final_audio_opened": False,
        },
        "capture_plan_passed": plan_audit["passed"],
        "real_capture_manifests_found": len(manifest_audits),
        "locked_final_policy": {
            "development_audio_opened": False,
            "metadata_only_until_one_shot_final": True,
        },
        "download_decision": {
            "download_now": False,
            "reason": (
                "The licensed ASRNN real-hardware dataset is downloaded and audited for an "
                "internal ProCo RAT pilot. The remaining product-fidelity evidence is a "
                "session-disjoint paired Capture Pack, which unpaired public guitar cannot replace. pOD-set "
                "remains metadata-only because its record page does not state usable rights."
            ),
            "downloaded": "asrnn-physical-effects (CC-BY-NC-4.0)",
            "unpaired_ood_downloaded": "GuitarSet mono pickup mix (CC-BY-4.0)",
            "unpaired_ood_can_measure_pedal_fidelity": False,
            "candidate_not_downloaded": "pod-set (45.1 GB)",
        },
        "quality_contract": {
            "source_audio_mutation": "forbidden",
            "analysis_copy_resampling_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "exact_post_latency_alignment": True,
        },
        "audio_devices_accessed": 0,
        "next_gate": (
            "Retain the accepted internal non-commercial pilots only; no UI or product "
            "fidelity claim is admitted. Phase 7 and GuitarSet OOD results must be read "
            "separately: unpaired runtime stability is not paired-effect accuracy. "
            "Prioritize separately licensed, session-disjoint Dry/Wet files with audible "
            "Tone sweeps and varied dynamics; receive them as offline files, without "
            "accessing physical audio devices. Any further architecture experiments "
            "remain development-only; reserve new locked-final audio until all gates freeze."
        ),
    }
    _write(output / "source-audit.json", registry)
    _write(output / "capture-plan-audit.json", plan_audit)
    _write(output / "manifest-audits.json", {"schema": 1, "packs": manifest_audits})
    _write(output / "acceptance.json", acceptance)
    return acceptance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("remix/runs/data-readiness-phase1"),
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.workspace.resolve(), args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
