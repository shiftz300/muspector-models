#!/usr/bin/env python3
"""Fail-closed DFZ forward promotion from complete, source-bound CPU evidence.

Legacy training summaries are selection evidence, not admission reports. Create
the calibration report with ``evaluate_asrnn_effect --split dfz-calibration``.
This program never evaluates audio or touches an audio device; promotion only
checks evidence, model/graph bytes, and the source-file digests already recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

import onnx
import torch

from .asrnn_effects import effect_files
from .evaluate_asrnn_effect import _quality_gate, manifest_sha256
from .stable_charge_residual import ChargeBank
from .stable_effect import load_stable_effect


CONTROL_VALUES = (0, 50, 100)
ALL_TAKES = set(range(1, 33))
CALIBRATION_TAKES = {5, 10, 15, 20, 25, 30}
FIT_TAKES = ALL_TAKES - CALIBRATION_TAKES
CONTROL_GROUPS = {f"{a},{b}" for a, b in itertools.product(CONTROL_VALUES, repeat=2)}
SCORED_FRAMES_PER_FILE = 144_000 - 1_024


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def valid_digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def finite_numbers(value) -> bool:
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_numbers(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return isinstance(value, str)


def validate_records(records: list[dict], split: str, takes: set[int]) -> dict[str, dict]:
    expected = {f"{a},{b},{take}.wav" for a, b in itertools.product(CONTROL_VALUES, repeat=2) for take in takes}
    require(isinstance(records, list) and len(records) == len(expected), "DFZ source record count differs")
    indexed = {}
    for row in records:
        path = PurePosixPath(row["relative_path"])
        require(not path.is_absolute() and ".." not in path.parts and len(path.parts) >= 3,
                "DFZ source path must remain relative to the corpus")
        require(path.parts[-3:-1] == ("dfz", split), "DFZ source split/path mismatch")
        require(path.name in expected and path.name not in indexed, "DFZ source filename coverage differs")
        a, b, _ = map(int, path.stem.split(","))
        require(row["controls"] == [a / 100.0, b / 100.0], "DFZ source control semantics differ")
        require(row["frames"] == 144_000 and row["sample_rate"] == 48_000 and row["channels"] == 2,
                "DFZ source geometry differs from three-second 48kHz Dry/Wet")
        require(all(valid_digest(row.get(key)) for key in ("file_sha256", "dry_sha256", "pair_sha256")),
                "DFZ source content hashes are missing or malformed")
        peak = row["target_peak_after_warmup"]
        require(isinstance(peak, (int, float)) and math.isfinite(peak) and peak >= 0,
                "DFZ source target peak is invalid")
        indexed[path.name] = row
    require(set(indexed) == expected, "DFZ source files do not cover the frozen partition")
    return indexed


def validate_data_manifest(report: dict, partition: str) -> dict[str, dict]:
    manifest = report.get("data_manifest")
    require(isinstance(manifest, dict), "complete data manifest is required; legacy summaries cannot be promoted")
    split, takes = ("train", CALIBRATION_TAKES) if partition == "calibration" else ("eval", ALL_TAKES)
    require(manifest.get("schema") == 1 and manifest.get("device") == "dfz"
            and manifest.get("official_split") == split and manifest.get("partition") == partition,
            "DFZ data manifest identity differs")
    require(manifest.get("full_causal_prefix_processed") is True
            and manifest.get("warmup_frames_excluded_from_scoring") == 1_024,
            "DFZ evaluation did not use the frozen full-prefix scoring protocol")
    records = manifest["records"]
    require(manifest.get("sha256") == manifest_sha256(records), "DFZ data manifest hash mismatch")
    return validate_records(records, split, takes)


def validate_metrics(report: dict, partition: str, records: dict[str, dict]) -> None:
    field = "calibration" if partition == "calibration" else "official_eval"
    metrics = report[field]
    require(finite_numbers(metrics), "DFZ metrics contain non-finite or malformed values")
    require(report.get("accepted") is True and _quality_gate(metrics), f"DFZ {partition} audio quality gate failed")
    count, per_tuple = (54, 6) if partition == "calibration" else (288, 32)
    require(metrics["examples"] == count and metrics.get("scored_frames") == count * SCORED_FRAMES_PER_FILE,
            f"DFZ {partition} example/frame coverage differs")
    first = metrics.get("by_first_control", {})
    tuples = metrics.get("by_control_tuple", {})
    require(set(first) == {"0", "50", "100"} and set(tuples) == CONTROL_GROUPS,
            "DFZ Blend/Filter group coverage differs")
    require(all(row["files"] == count // 3 for row in first.values())
            and all(row["files"] == per_tuple for row in tuples.values()),
            "DFZ per-group example counts differ")
    # Retain the stricter capacity-selection worst-first-control protection.
    require(max(row["absolute_peak_error_p95"] for row in first.values()) <= 0.03,
            "DFZ worst Blend peak P95 exceeds 0.03")
    quiet = sum(row["target_peak_after_warmup"] < 1.0e-3 for row in records.values())
    require(metrics.get("quiet_examples") == quiet and metrics.get("audible_examples") == count - quiet,
            "DFZ quiet/audible coverage is not disclosed accurately")
    require(metrics.get("quiet_dataset_gate_vacuous") is (quiet == 0), "DFZ empty quiet set must be explicitly disclosed")
    if quiet == 0:
        require(metrics["quiet_prediction_peak_maximum"] == 0.0, "DFZ empty quiet-set sentinel differs")


def validate_evidence(calibration: dict, pytorch: dict, deployed: dict, runtime: dict,
                      checkpoint_digest: str, graph_digest: str) -> dict[str, bool]:
    """Pure validation entry point; never runs models or accesses a dataset."""
    require(valid_digest(checkpoint_digest) and valid_digest(graph_digest), "invalid artifact digests")
    for report in (calibration, pytorch, deployed, runtime):
        require(report.get("checkpoint_sha256") == checkpoint_digest, "DFZ evidence checkpoint SHA mismatch")
    for report in (calibration, pytorch, deployed):
        require(report.get("device") == "dfz" and report.get("control_names") == ["blend", "filter"],
                "DFZ effect identity or control semantics differ")
        require(report.get("compute_device") == "cpu", "DFZ admission requires CPU evaluation")
        policy = report.get("quality_policy", {})
        require(policy.get("source_files_read_only") is True and all(policy.get(key) is False for key in
                ("automatic_normalization", "automatic_limiting", "lossy_reencoding", "physical_audio_devices_used")),
                "DFZ audio-quality policy differs")
    require(calibration.get("runtime") == pytorch.get("runtime") == "stable"
            and calibration.get("onnx_sha256") is None and pytorch.get("onnx_sha256") is None,
            "DFZ PyTorch reference evidence must use the unchanged stable CPU model")
    require(deployed.get("runtime") == "onnxruntime-cpu" and deployed.get("onnx_sha256") == graph_digest,
            "DFZ deployed evidence must evaluate the exact ONNX graph")

    cal_records = validate_data_manifest(calibration, "calibration")
    torch_records = validate_data_manifest(pytorch, "development")
    onnx_records = validate_data_manifest(deployed, "development")
    require(pytorch["data_manifest"]["sha256"] == deployed["data_manifest"]["sha256"]
            and torch_records == onnx_records, "DFZ PyTorch and ONNX used different source recordings")
    for report, partition, records in ((calibration, "calibration", cal_records),
                                        (pytorch, "development", torch_records),
                                        (deployed, "development", onnx_records)):
        validate_metrics(report, partition, records)

    train = calibration.get("train_partition_manifest", {})
    require(train.get("schema") == 1 and train.get("device") == "dfz"
            and train.get("official_split") == "train"
            and train.get("selection") == "take_id_modulo_5_equals_0_for_calibration"
            and train.get("fit_audio_used_for_model_evaluation") is False
            and train.get("independent_session_holdout") is False,
            "DFZ complete fit/calibration partition provenance is required")
    partitions = train.get("partitions", {})
    require(set(partitions) == {"fit", "calibration"} and train.get("sha256") == manifest_sha256(partitions),
            "DFZ training partition manifest hash mismatch")
    fit_records = validate_records(partitions["fit"], "train", FIT_TAKES)
    pooled_cal = validate_records(partitions["calibration"], "train", CALIBRATION_TAKES)
    require(pooled_cal == cal_records and not (set(fit_records) & set(cal_records)),
            "DFZ fit/calibration partition differs from the evaluated calibration set")
    train_dry = {row["dry_sha256"] for row in (*fit_records.values(), *cal_records.values())}
    development_dry = {row["dry_sha256"] for row in torch_records.values()}
    require(not (train_dry & development_dry), "DFZ official train/eval contain exact Dry overlap")

    require(runtime.get("backend") == "onnxruntime" and runtime.get("compute_device") == "cpu"
            and runtime.get("threads") == 1 and runtime.get("onnx_sha256") == graph_digest,
            "DFZ runtime must be the same single-thread CPU ONNX graph")
    require(runtime.get("float_precision") == "float32" and runtime.get("quantization") is False,
            "DFZ requires unquantized float32 export")
    numeric = ("full_realtime_factor", "streaming_realtime_factor", "torch_onnx_max_absolute_error",
               "dynamic_control_signal_stream_max_error", "dynamic_control_silence_peak", "quiet_input_peak")
    require(all(isinstance(runtime.get(key), (int, float)) and math.isfinite(runtime[key]) and runtime[key] >= 0
                for key in numeric), "DFZ runtime measurements are invalid")
    require(runtime.get("additional_safety_passed") is True
            and runtime["torch_onnx_max_absolute_error"] <= 2e-6
            and runtime["dynamic_control_signal_stream_max_error"] <= 2e-6
            and runtime["dynamic_control_silence_peak"] == 0.0 and runtime["quiet_input_peak"] <= 1e-3,
            "DFZ ONNX parity/quiet/dynamic-stream safety gate failed")
    require(runtime.get("cpu_realtime_passed") is True and runtime.get("streaming_block_frames") == 1_024
            and 0 < runtime["full_realtime_factor"] < 1 and 0 < runtime["streaming_realtime_factor"] < 1,
            "DFZ whole/1024-frame CPU real-time gate failed")
    require(all(runtime.get(key) is False for key in
                ("physical_audio_devices_used", "ui_integration_allowed", "automatic_normalization", "automatic_limiting")),
            "DFZ runtime safety policy differs")
    return dict.fromkeys(("cpu_calibration_full54", "fit234_cal54_development288_coverage",
                          "nine_control_tuple_coverage", "source_manifests_match",
                          "official_train_eval_dry_disjoint", "pytorch_full288_quality",
                          "onnx_full288_quality", "worst_blend_peak_p95_le_003",
                          "same_checkpoint_and_graph", "quiet_coverage_disclosed",
                          "independent_quiet_and_dynamic_stream_probes", "float32_onnx_parity",
                          "single_thread_cpu_realtime"), True)


def verify_source_files(calibration: dict, development: dict, corpus: Path) -> None:
    """Verify exact source bytes without decoding or playing audio."""
    train = calibration["train_partition_manifest"]["partitions"]
    records = train["fit"] + train["calibration"] + development["data_manifest"]["records"]
    expected_paths = {path.resolve().relative_to(corpus.resolve()).as_posix()
                      for split in ("train", "eval") for path in effect_files(corpus, "dfz", split)}
    require({row["relative_path"] for row in records} == expected_paths, "actual DFZ corpus file coverage changed")
    for row in records:
        require(file_sha256(corpus / row["relative_path"]) == row["file_sha256"],
                f"DFZ source bytes changed: {row['relative_path']}")


def validate_charge_banks(model: torch.nn.Module) -> list[dict]:
    """Check the actual fixed EMA modules, separately from LSTM candidates.

    For zero initial state, p=tanh(100*Dry)^2 is in [0,1]. A nonnegative
    diagonal alpha in (0,1), complementary input weights and zero biases
    preserve that interval; ReLU is identity on these reachable states.
    This statement concerns only the charge state, not the composite model.
    """
    audits, recognized_cells = [], set()
    for name, bank in model.named_modules():
        if not isinstance(bank, ChargeBank):
            continue
        require(type(bank) is ChargeBank, "DFZ charge bank subclass behavior has not been audited")
        cell = bank.cell
        require(type(cell) is torch.nn.RNN and cell.nonlinearity == "relu",
                "DFZ charge bank must use the audited ReLU RNN")
        require(cell.input_size == 1 and cell.hidden_size == bank.width and cell.num_layers == 1
                and cell.batch_first and not cell.bidirectional and cell.dropout == 0,
                "DFZ charge bank recurrent geometry differs")
        require(all(not parameter.requires_grad and parameter.dtype == torch.float32
                    and torch.isfinite(parameter).all().item() for parameter in cell.parameters()),
                "DFZ charge bank must remain fixed finite float32")
        recurrent, incoming = cell.weight_hh_l0.detach(), cell.weight_ih_l0.detach()
        require(tuple(recurrent.shape) == (bank.width, bank.width)
                and tuple(incoming.shape) == (bank.width, 1) and cell.bias
                and tuple(cell.bias_ih_l0.shape) == tuple(cell.bias_hh_l0.shape) == (bank.width,),
                "DFZ charge bank parameter geometry differs")
        alpha = recurrent.diag()
        require(torch.equal(recurrent, torch.diag(alpha))
                and bool(((alpha > 0) & (alpha < 1)).all()),
                "DFZ charge bank requires diagonal alpha strictly between zero and one")
        require(bool((incoming >= 0).all()) and torch.equal(incoming[:, 0], 1 - alpha),
                "DFZ charge bank input weights must be nonnegative and exactly complementary")
        require(cell.bias and bool((cell.bias_ih_l0.detach() == 0).all())
                and bool((cell.bias_hh_l0.detach() == 0).all()),
                "DFZ charge bank biases must remain exactly zero")
        recognized_cells.add(id(cell))
        audits.append({
            "module": name, "kind": "fixed-nonnegative-diagonal-relu-ema", "width": bank.width,
            "alpha": alpha.tolist(), "input_weights": incoming[:, 0].tolist(),
            "state_transition_infinity_norm": float(alpha.max()),
            "relu_verified": True, "parameters_frozen": True, "zero_biases_verified": True,
            "nonnegative_complementary_input_verified": True,
            "zero_initial_state_reachable_interval": [0.0, 1.0],
            "bound_scope": "charge state only under its bounded source-input transform; not a whole-model proof",
        })
    require(all(id(cell) in recognized_cells for cell in model.modules() if isinstance(cell, torch.nn.RNN)),
            "DFZ contains an RNN outside the audited fixed ChargeBank")
    return audits


def validate_artifacts(checkpoint: Path, graph: Path, digest: str) -> tuple[torch.nn.Module, dict, list[float], list[dict]]:
    model, payload = load_stable_effect(checkpoint)
    require(payload.get("device") == "dfz" and payload.get("sample_rate") == 48_000
            and model.control_count == 2, "actual DFZ checkpoint identity differs")
    require(all(parameter.dtype == torch.float32 and torch.isfinite(parameter).all().item()
                for parameter in model.parameters()), "actual DFZ parameters are not finite float32")
    recurrent = [layer for layer in model.modules() if isinstance(layer, torch.nn.LSTM)]
    require(bool(recurrent), "DFZ stable-core checkpoint has no recurrent layers")
    norms = [float(layer.weight_hh_l0.detach()[2 * layer.hidden_size:3 * layer.hidden_size].abs().sum(1).max())
             for layer in recurrent]
    require(max(norms) <= 0.995001, "actual DFZ candidate recurrent matrix exceeds the frozen bound")
    charge_audits = validate_charge_banks(model)
    # A graph-only digest cannot authenticate weights stored beside the graph.
    # Current exports are self-contained; reject external tensors rather than
    # allowing an unbound sidecar to change after the parity measurements.
    network = onnx.load(graph, load_external_data=False)
    require(all(tensor.data_location != onnx.TensorProto.EXTERNAL and not tensor.external_data
                for tensor in network.graph.initializer), "DFZ ONNX external weight data is not source-bound")
    onnx.checker.check_model(network, full_check=True)
    metadata = {row.key: row.value for row in network.metadata_props}
    require(metadata.get("checkpoint_sha256") == digest and metadata.get("sample_rate") == "48000"
            and metadata.get("control_count") == "2" and metadata.get("precision") == "float32",
            "actual ONNX graph metadata differs from the DFZ checkpoint")
    require(not any("Quantize" in node.op_type or node.op_type.startswith("QLinear") for node in network.graph.node),
            "actual ONNX graph contains quantization")
    for tensor in network.graph.initializer:
        kind = onnx.TensorProto.DataType.Name(tensor.data_type)
        require(not ("FLOAT" in kind or kind == "DOUBLE") or kind == "FLOAT", "actual ONNX weights are not float32")
    return model, payload, norms, charge_audits


def promote(args) -> dict:
    require(not args.output.exists(), "DFZ promotion output already exists")
    paths = {"calibration": args.calibration, "pytorch": args.pytorch, "onnx": args.development, "runtime": args.runtime}
    reports = {key: json.loads(path.read_text()) for key, path in paths.items()}
    digest, graph_digest = file_sha256(args.checkpoint), file_sha256(args.onnx)
    checks = validate_evidence(reports["calibration"], reports["pytorch"], reports["onnx"], reports["runtime"], digest, graph_digest)
    verify_source_files(reports["calibration"], reports["onnx"], args.corpus)
    model, payload, norms, charge_audits = validate_artifacts(args.checkpoint, args.onnx, digest)
    checks.update(actual_source_bytes_match=True, actual_float32_artifacts_match=True,
                  actual_contractive_candidate_matrices=True)
    if charge_audits:
        checks["actual_fixed_charge_banks_verified"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dfz-admission-", dir=args.output.parent) as temporary:
        staging = Path(temporary)
        shutil.copy2(args.checkpoint, staging / "dfz-stable.pt")
        shutil.copy2(args.onnx, staging / "dfz-stable.onnx")
        require(file_sha256(staging / "dfz-stable.pt") == digest and file_sha256(staging / "dfz-stable.onnx") == graph_digest,
                "DFZ artifacts changed while packaging")
        evidence = {}
        for name, path in paths.items():
            destination = staging / f"{name}-evidence.json"
            shutil.copy2(path, destination)
            require(json.loads(destination.read_text()) == reports[name], "DFZ evidence changed while packaging")
            evidence[name] = {"path": str(args.output / destination.name), "sha256": file_sha256(destination)}
        card = {
            "schema": 1, "phase": "phase-9-dfz-forward", "device": "dfz", "accepted": True,
            "evaluation_complete": True, "status": "accepted-internal-noncommercial-development-pilot",
            "checkpoint": str(args.output / "dfz-stable.pt"), "checkpoint_sha256": digest,
            "onnx_path": str(args.output / "dfz-stable.onnx"), "onnx_sha256": graph_digest,
            "architecture": payload["architecture"], "sample_rate": 48_000, "control_names": ["blend", "filter"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "candidate_recurrent_infinity_norms": norms, "checks": checks, "evidence": evidence,
            "charge_bank_audits": charge_audits,
            "charge_bank_recurrent_infinity_norms": [audit["state_transition_infinity_norm"] for audit in charge_audits],
            "formal_stability_proof_verified": False,
            "stability_evidence": "contractive LSTM candidate matrices, separately audited fixed charge banks if present, and empirical zero/quiet/runtime checks; not a whole-model stability proof",
            "calibration": reports["calibration"]["calibration"],
            "development_challenge": reports["onnx"]["official_eval"],
            "pytorch_reference_challenge": reports["pytorch"]["official_eval"],
            "runtime": {**reports["runtime"], "onnx_path": str(args.output / "dfz-stable.onnx")},
            "data_manifest_sha256": reports["onnx"]["data_manifest"]["sha256"],
            "train_partition_manifest_sha256": reports["calibration"]["train_partition_manifest"]["sha256"],
            "calibration_is_independent_final_holdout": False, "new_locked_final_audio_opened": False,
            "physical_audio_devices_used": False, "source_audio_modified": False,
            "ui_integration_allowed": False, "commercial_release_allowed": False, "dataset_license": "CC-BY-NC-4.0",
            "quality_contract": reports["onnx"]["quality_policy"],
            "development_control_grid": [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
            "unseen_control_interpolation_evaluated": False,
            "limitations": ["official eval is development evidence, not independent locked-final validation",
                            "public pretrained core may have seen official train; take partition is selection only",
                            "only the nine Blend/Filter combinations have paired forward validation",
                            "an empty quiet corpus subset is not measured quiet fidelity; independent low-level probes are required",
                            "offline single-thread timing is not a real audio-driver latency guarantee",
                            "candidate recurrent matrix norms alone do not prove full gated or composite-model stability",
                            "unpaired GuitarSet OOD evidence is not claimed by this DFZ admission"],
        }
        (staging / "model-card.json").write_text(json.dumps(card, indent=2, allow_nan=False) + "\n")
        packaged_metrics = {**reports["onnx"], "checkpoint": card["checkpoint"]}
        (staging / "metrics.json").write_text(json.dumps(packaged_metrics, indent=2, allow_nan=False) + "\n")
        require(not args.output.exists(), "DFZ promotion destination appeared during validation")
        os.rename(staging, args.output)
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "onnx", "calibration", "pytorch", "development", "runtime", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/asrnn-physical-effects"))
    args = parser.parse_args()
    card = promote(args)
    print(json.dumps({"accepted": card["accepted"], "checkpoint": card["checkpoint"], "checks": card["checks"]}))


if __name__ == "__main__":
    main()
