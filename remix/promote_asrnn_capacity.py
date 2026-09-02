#!/usr/bin/env python3
"""Promote only a hash-matched capacity candidate passing every development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .evaluate_asrnn_effect import _quality_gate
from .evaluate_phase7_guitarset_ood import EXPECTED_MD5
from .stable_effect import load_stable_effect


def validate_evidence(screen: dict, challenge: dict, ood: dict, runtime: dict, digest: str) -> dict:
    matching = [row for row in screen["results"] if row["checkpoint_sha256"] == digest]
    if (
        not screen.get("screen_complete", False)
        or screen.get("expected_candidates", 0) != len(screen["results"])
        or len(matching) != 1
    ):
        raise ValueError("capacity pool or candidate provenance is incomplete")
    selected = matching[0]
    if any(item["checkpoint_sha256"] != digest for item in (selected, challenge, ood, runtime)):
        raise ValueError("capacity evidence refers to different checkpoint bytes")
    metrics = challenge["official_eval"]
    worst_attack = max(row["absolute_peak_error_p95"] for row in metrics["by_first_control"].values())
    checks = {
        "calibration": bool(selected["calibration"]["passes_selection_gate"]
                            and selected["calibration"]["examples"] == 89
                            and selected["calibration"]["absolute_peak_error_p95"] <= 0.02
                            and selected["calibration"]["worst_attack_absolute_peak_error_p95"] <= 0.03),
        "development_coverage": metrics["examples"] == 352 and len(metrics["by_first_control"]) == 11,
        "development_quality": bool(challenge["accepted"] and _quality_gate(metrics)),
        "development_worst_attack_peak": worst_attack <= 0.03,
        "guitarset_runtime": bool(ood["passed"] and len(ood["checks"]) == 10 and all(ood["checks"].values())),
        "guitarset_coverage": ood["metrics"]["examples"] == 180 and ood["metrics"]["source_members"] == 60,
        "guitarset_source": ood["archive_md5"] == EXPECTED_MD5,
        "same_frozen_reference": ood["base_source_checkpoint_sha256"] == "b6f2875d6ba594e429c2a3bdc6adfd9af7c2bededc0fdef5c868b680558569ce",
        "dynamic_control_and_quiet_safety": bool(runtime["additional_safety_passed"]
                                                  and runtime["dynamic_control_signal_stream_max_error"] <= 2e-6
                                                  and runtime["quiet_input_peak"] <= 1e-3
                                                  and runtime["dynamic_control_silence_peak"] == 0.0),
        "cpu_faster_than_realtime": bool(runtime.get("cpu_realtime_passed", runtime.get("pytorch_cpu_realtime_passed", False))
                                         and runtime["compute_device"] == "cpu"
                                         and runtime["full_realtime_factor"] < 1.0
                                         and runtime["streaming_realtime_factor"] < 1.0),
        "contractive_recurrence": max(selected["import_runtime"]["candidate_recurrent_infinity_norms"]) <= 0.995001,
    }
    if runtime.get("backend") == "onnxruntime":
        checks["same_onnx_graph_evaluated"] = challenge.get("onnx_sha256") == ood.get("onnx_sha256") == runtime["onnx_sha256"]
        checks["float32_export_parity"] = bool(runtime["torch_onnx_max_absolute_error"] <= 2e-6
                                                and runtime["float_precision"] == "float32"
                                                and not runtime["quantization"])
    if not all(checks.values()):
        raise ValueError(f"capacity candidate rejected: {[name for name, passed in checks.items() if not passed]}")
    return checks


def promote(run: Path, output: Path, screen_path: Path | None = None, checkpoint_path: Path | None = None) -> dict:
    if output.exists():
        raise ValueError(f"pilot output already exists: {output}")
    screen = json.loads((screen_path or run / "screen.json").read_text())
    runtime_name = "onnx-runtime.json" if (run / "onnx-runtime.json").is_file() else "runtime.json"
    challenge, ood, runtime = (
        json.loads((run / name).read_text())
        for name in ("challenge.json", "guitarset-ood.json", runtime_name)
    )
    source = checkpoint_path or run / "selected.pt"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    checks = validate_evidence(screen, challenge, ood, runtime, digest)
    model, _ = load_stable_effect(source)
    members = model.members if hasattr(model, "members") else [model]
    norms = [float(layer.weight_hh_l0.detach()[2 * member.hidden_size:3 * member.hidden_size].abs().sum(1).max())
             for member in members for layer in member.rnn_layers]
    if max(norms) > 0.995001:
        raise ValueError("actual checkpoint recurrence fails the frozen stability bound")
    graph = Path(runtime["onnx_path"]) if runtime.get("backend") == "onnxruntime" else None
    if graph and hashlib.sha256(graph.read_bytes()).hexdigest() != runtime["onnx_sha256"]:
        raise ValueError("ONNX graph bytes differ from runtime evidence")
    reference_path = run / "challenge-pytorch.json"
    reference = json.loads(reference_path.read_text()) if graph else challenge
    if (
        reference["checkpoint_sha256"] != digest
        or reference["official_eval"]["examples"] != 352
        or not _quality_gate(reference["official_eval"])
        or max(row["absolute_peak_error_p95"] for row in reference["official_eval"]["by_first_control"].values()) > 0.03
    ):
        raise ValueError("full PyTorch reference challenge does not pass for this checkpoint")
    checks["pytorch_reference_full_challenge"] = True
    selected = next(row for row in screen["results"] if row["checkpoint_sha256"] == digest)
    output.mkdir(parents=True)
    checkpoint = output / "cs3-stable.pt"
    shutil.move(str(source), checkpoint)
    if graph:
        shutil.move(str(graph), output / "cs3-stable.onnx")
    report = {
        "schema": 1,
        "phase": "phase-8-capacity-upgrade",
        "evaluation_complete": True,
        "accepted": True,
        "status": "accepted-internal-noncommercial-development-pilot",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "checkpoint_retained": True,
        "architecture": selected.get("architecture", "stable-conditioned-lstm-4x64"),
        "parameters": selected["parameters"],
        "sample_rate": 48000,
        "control_names": ["attack"],
        "source_model": selected["member"],
        "source_sha256": selected["source_sha256"],
        "source_archive_sha256": screen["archive_sha256"],
        "group_audit_sha256": screen["group_audit_sha256"],
        "source_components": selected.get("sources", []),
        "candidate_recurrent_infinity_norms": norms,
        "evidence": {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in (
                ("calibration", screen_path or run / "screen.json"),
                ("challenge", run / "challenge.json"),
                ("guitarset_ood", run / "guitarset-ood.json"),
                ("runtime", run / runtime_name),
                ("pytorch_reference", reference_path if graph else run / "challenge.json"),
            )
        },
        "local_gradient_updates": 0,
        "method": "fixed convex ensemble of public pretrained causal models" if selected.get("sources") else "public pretrained architecture selection and independent runtime import",
        "selection_method": "calibration admission followed by development challenge and OOD elimination",
        "initial_calibration_choice": screen["selected"]["member"],
        "calibration_is_independent_final_holdout": False,
        "calibration_caveat": screen["calibration_caveat"],
        "calibration": selected["calibration"],
        "development_challenge": challenge["official_eval"],
        "pytorch_reference_challenge": reference["official_eval"],
        "guitarset_ood": ood["metrics"],
        "runtime": {**runtime, "measured_onnx_path": runtime.get("onnx_path"),
                    "onnx_path": str(output / "cs3-stable.onnx") if graph else None},
        "onnx_path": str(output / "cs3-stable.onnx") if graph else None,
        "onnx_sha256": runtime.get("onnx_sha256"),
        "checks": checks,
        "dataset_license": "CC-BY-NC-4.0",
        "quality_contract": challenge["quality_policy"],
        "new_locked_final_audio_opened": False,
        "ui_integration_allowed": False,
        "commercial_release_allowed": False,
        "limitations": [
            "development quality gates passed; no new independent locked-final evidence",
            "public checkpoint, not locally trained from scratch",
            "one CS-3 Attack control at the dataset's other fixed settings",
            "unpaired GuitarSet measures runtime safety, not pedal fidelity",
            "offline CPU timing is not a real audio-driver latency guarantee",
        ],
    }
    challenge["checkpoint"] = str(checkpoint)
    (output / "metrics.json").write_text(json.dumps(challenge, indent=2) + "\n")
    (output / "model-card.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("remix/runs/asrnn-cs3-capacity-screen"))
    parser.add_argument("--output", type=Path, default=Path("remix/runs/asrnn-cs3-ensemble-pilot"))
    parser.add_argument("--screen", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    report = promote(args.run, args.output, args.screen, args.checkpoint)
    print(json.dumps({"status": report["status"], "checkpoint": report["checkpoint"], "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
