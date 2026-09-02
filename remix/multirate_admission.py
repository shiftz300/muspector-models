"""Strict evidence/structure validation for the separate finite-history schema.

Does not change the recurrent-model importer, create a model card, or promote
anything. No LSTM norm is fabricated for a convolutional architecture. All
original audio, data coverage, parity and CPU timing thresholds remain intact.
"""
from __future__ import annotations

import math
from pathlib import Path, PurePosixPath

import torch

from .centered_multirate_fuzz import CenteredModulatedAudioGCN, CenteredMultirateFuzz
from .evaluate_multirate_fuzz import ARCHITECTURES, CENTERED_ARCHITECTURE, validate_calibration_entry
from .multirate_fuzz import FiniteController, ModulatedAudioGCN, MultirateFuzz
from .promote_dfz_forward import (finite_numbers, require, valid_digest, validate_data_manifest, validate_metrics)
from .precheck_multirate_fuzz import sha256
from .widen_multirate_fuzz import WIDE_GEOMETRY


def validate_structure(model, payload):
    centered = payload.get("architecture") == CENTERED_ARCHITECTURE
    require(type(model) is (CenteredMultirateFuzz if centered else MultirateFuzz)
            and type(model.audio) is (CenteredModulatedAudioGCN if centered else ModulatedAudioGCN)
            and type(model.controller) is FiniteController,
            "only the reviewed finite-history classes are supported")
    require(payload.get("geometry") == WIDE_GEOMETRY and payload.get("architecture") in ARCHITECTURES
            and payload.get("sample_rate") == 48000 and payload.get("device") == "dfz"
            and payload.get("experimental_schema") == 1, "finite-history schema identity differs")
    require(model.audio.width == 32 and model.controller.width == 16 and model.fast_count == model.slow_count == 10
            and model.control_count == 2 and model.sample_rate == 48000, "finite-history geometry differs")
    require(model.audio.dilations == model.controller.dilations == [2 ** index for index in range(10)], "dilation clock differs")
    require(not any(isinstance(layer, torch.nn.RNNBase) for layer in model.modules()), "unexpected learned recurrence")
    require(all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in model.state_dict().values()),
            "finite float32 model weights required")
    expected_state = [width * 2 ** index for width in (32, 16) for index in range(10)]
    require(model.state_widths == expected_state, "finite-history state layout differs")
    for branch, width in ((model.audio, 32), (model.controller, 16)):
        require(type(branch.input) is torch.nn.Conv1d and branch.input.bias is None and branch.input.kernel_size == (1,)
                and branch.input.stride == (1,) and branch.input.padding == (0,) and branch.input.groups == 1
                and branch.input.in_channels == (1 if branch is model.audio else 8) and branch.input.out_channels == width
                and branch.input.dilation == (1,),
                "input projection geometry differs")
        require(type(branch.output) is torch.nn.Conv1d and branch.output.bias is None and branch.output.kernel_size == (1,)
                and branch.output.stride == (1,) and branch.output.padding == (0,) and branch.output.groups == 1
                and branch.output.in_channels == width and branch.output.out_channels == (1 if branch is model.audio else width)
                and branch.output.dilation == (1,),
                "output projection geometry differs")
        require(len(branch.convolutions) == len(branch.projections) == 10, "finite block count differs")
        for index, (convolution, projection) in enumerate(zip(branch.convolutions, branch.projections)):
            require(type(convolution) is torch.nn.Conv1d and convolution.bias is None and convolution.kernel_size == (3,)
                    and convolution.dilation == (2 ** index,) and convolution.stride == (1,)
                    and convolution.padding == (0,) and convolution.groups == 1
                    and convolution.in_channels == width and convolution.out_channels == 2 * width,
                    "reviewed causal gated convolution differs")
            require(type(projection) is torch.nn.Conv1d and projection.bias is None and projection.kernel_size == (1,)
                    and projection.stride == (1,) and projection.padding == (0,) and projection.groups == 1
                    and projection.dilation == (1,)
                    and projection.in_channels == projection.out_channels == width, "zero-preserving residual projection differs")
    require(len(model.audio.current_controls) == len(model.audio.slow_controls) == 10
            and len(model.controller.gate_bias) == 10, "modulation branch count differs")
    for current, slow in zip(model.audio.current_controls, model.audio.slow_controls):
        require(type(current) is torch.nn.Linear and current.in_features == 2 and current.out_features == 64
                and current.bias is not None and type(slow) is torch.nn.Linear and slow.in_features == 16
                and slow.out_features == 64 and slow.bias is None, "control modulation geometry differs")
    require(all(bias.shape == (16,) for bias in model.controller.gate_bias), "slow gate geometry differs")
    if centered:
        require(len(model.audio.activation_current) == len(model.audio.activation_slow) == 10,
                "conditional threshold branch count differs")
        for current, slow in zip(model.audio.activation_current, model.audio.activation_slow):
            require(type(current) is torch.nn.Linear and current.in_features == 2 and current.out_features == 32
                    and current.bias is not None and type(slow) is torch.nn.Linear and slow.in_features == 16
                    and slow.out_features == 32 and slow.bias is None, "conditional threshold geometry differs")

    def row_norm(weight):
        return float(weight.detach().double().reshape(weight.shape[0], -1).abs().sum(1).max())

    # Centered tanh is the difference of two values in[-1,1], so its fast
    # update bound is3, not the plain model's1.5. Neither is an output clamp.
    update_bound = 3. if centered else 1.5
    value_bound = row_norm(model.audio.input.weight) * 21.4
    fast_bounds = [value_bound]
    for projection in model.audio.projections:
        value_bound += update_bound * row_norm(projection.weight) / math.sqrt(10)
        fast_bounds.append(value_bound)
    output_bound = value_bound * row_norm(model.audio.output.weight)
    slow_bound = row_norm(model.controller.input.weight) * max(21.4 ** 2, 21.4, 1.)
    for projection in model.controller.projections:
        slow_bound += row_norm(projection.weight) / math.sqrt(10)
    slow_bound *= row_norm(model.controller.output.weight)
    require(all(math.isfinite(value) and 0 <= value < 1e30 for value in (output_bound, slow_bound, *fast_bounds)),
            "conservative finite-history bound is not safely representable")
    state = model.initial_state(torch.zeros(1, 1))
    return {"architecture": "finite-causal-centered-gated-convolutions" if centered else "finite-causal-gated-convolutions",
            "fast_update_bound": update_bound, "learned_recurrent_matrices": False,
            "audio_receptive_field_frames": model.audio.receptive_field,
            "controller_receptive_field_blocks": model.controller.receptive_field, "controller_block_frames": 64,
            "zero_audio_tail_expiry_frames": model.audio.receptive_field - 1,
            "state_tensor_bytes_mono": sum(value.numel() * value.element_size() for value in state),
            "conservative_output_bound_for_abs_dry_le1": output_bound, "conservative_slow_held_bound_for_abs_dry_le1": slow_bound,
            "bound_scope": "real-arithmetic BIBO bound for reviewed structure, zero initial state and abs(Dry)<=1; not a fidelity metric, output clamp or proof of arbitrary-input floating-point safety"}


def source_hashes_unchanged(report):
    hashes = report.get("source_sha256")
    require(isinstance(hashes, dict) and bool(hashes), "missing evaluator/training source hashes")
    for name, digest in hashes.items():
        require(isinstance(name, str) and PurePosixPath(name).name == name and name.endswith(".py") and valid_digest(digest),
                "invalid source-code evidence reference")
        require(sha256(Path(__file__).parent / name) == digest, "evidence source code changed")


def validate_evidence(calibration, pytorch, deployed, runtime, checkpoint_digest, graph_digest, calibration_digest):
    require(all(valid_digest(value) for value in (checkpoint_digest, graph_digest, calibration_digest)), "invalid evidence digests")
    partitions = validate_calibration_entry(calibration, checkpoint_digest)
    architecture = calibration["architecture"]
    cal_records = validate_data_manifest(calibration, "calibration")
    for report, backend in ((pytorch, "multirate-pytorch-cpu"), (deployed, "onnxruntime-cpu")):
        require(report.get("schema") == 1 and report.get("architecture") == architecture
                and report.get("geometry") == WIDE_GEOMETRY and report.get("device") == "dfz"
                and report.get("control_names") == ["blend", "filter"] and report.get("compute_device") == "cpu"
                and report.get("runtime") == backend and report.get("checkpoint_sha256") == checkpoint_digest,
                "development candidate/backend identity differs")
        require(report.get("calibration_entry_sha256") == calibration_digest and report.get("admitted") is False
                and report.get("source_and_audio_reverified") is True, "development lacks verified calibration authorization")
        policy = report.get("quality_policy", {})
        require(policy.get("source_files_read_only") is True and all(policy.get(key) is False for key in
                ("automatic_normalization", "automatic_limiting", "lossy_reencoding", "physical_audio_devices_used")),
                "development audio-quality policy differs")
        records = validate_data_manifest(report, "development")
        validate_metrics(report, "development", records)
    require(calibration.get("geometry") == WIDE_GEOMETRY, "calibration model geometry differs")
    require(pytorch.get("onnx_sha256") is None and deployed.get("onnx_sha256") == graph_digest, "deployed graph identity differs")
    require(pytorch["data_manifest"] == deployed["data_manifest"], "backend development sources differ")
    train_dry = {row["dry_sha256"] for values in partitions.values() for row in values}
    require(not train_dry.intersection(row["dry_sha256"] for row in deployed["data_manifest"]["records"]),
            "official train/development Dry overlap")
    require(all(report["data_manifest"]["corpus_root"] == calibration["data_manifest"]["corpus_root"]
                for report in (pytorch, deployed)), "corpus root differs")
    training_digest = calibration.get("training_report_sha256")
    require(valid_digest(training_digest) and all(report.get("training_report_sha256") == training_digest for report in (pytorch, deployed))
            and runtime.get("parent_report_sha256") == training_digest, "training evidence differs")
    purpose = "trained-centered-core-audit" if architecture == CENTERED_ARCHITECTURE else "trained-wide-core-audit"
    require(runtime.get("purpose") == purpose and runtime.get("runtime_passed") is True
            and runtime.get("calibration_passed") is True and runtime.get("checkpoint_sha256") == checkpoint_digest
            and runtime.get("graph_sha256") == graph_digest and runtime.get("geometry") == WIDE_GEOMETRY
            and runtime.get("source_and_audio_reverified") is True, "complete trained runtime audit required")
    if architecture == CENTERED_ARCHITECTURE:
        require(runtime.get("architecture") == architecture, "centered runtime architecture differs")
    require(all(runtime.get(key) is False for key in ("admitted", "official_eval_opened", "physical_audio_devices_used",
                                                     "source_audio_modified", "automatic_gain_or_normalization")), "runtime safety policy differs")
    cases, probes = runtime.get("real_audio_cases"), runtime.get("dynamic_probes")
    require(isinstance(cases, list) and len(cases) == 54 and {row.get("name") for row in cases} == set(cal_records),
            "runtime real-audio calibration coverage differs")
    require(isinstance(probes, list) and len(probes) == 3
            and {(row.get("batch"), row.get("frames")) for row in probes} == {(2, 6173), (1, 144000), (1, 1440000)},
            "runtime dynamic/long-recording coverage differs")
    require(finite_numbers(cases) and finite_numbers(probes), "nonfinite runtime cases")
    require(all(0 <= row[key] <= 2e-6 for row in cases for key in ("parity_max", "onnx_stream_max")),
            "real-audio float32 parity failed")
    require(all(0 <= row[key] <= 2e-6 for row in probes for key in ("parity_max", "onnx_stream_max", "cpu_stream_max"))
            and all(row.get("phase_exact") is True and row.get("inputs_unchanged") is True for row in probes),
            "dynamic long/stream/integrity checks failed")
    safety, timing = runtime.get("safety", {}), runtime.get("single_thread_rtf_including_validation", {})
    require(set(safety) == set(timing) == {"cpu", "onnx"} and finite_numbers(safety) and finite_numbers(timing),
            "complete finite CPU/ONNX safety and timing required")
    require(all(row.get("zero_peak") == row.get("expired_tail_peak") == row.get("future_prefix_error") == 0.
                and 0 <= row.get("quiet_peak", float("inf")) <= 1e-3 for row in safety.values()), "runtime silence/quiet/causality failed")
    require(all(set(row) == {"full", "stream1024"} and all(isinstance(values, list) and len(values) == 3
                and all(0 < value < 1 for value in values) for values in row.values()) for row in timing.values()),
            "single-thread full/1024 real-time check failed")
    return dict.fromkeys(("complete_cpu_calibration54", "official_development288_both_backends", "nine_control_tuples",
                          "unchanged_audio_quality_thresholds", "worst_blend_le003", "source_manifest_match", "exact_dry_disjoint",
                          "trained_single_float32_graph", "real54_and_long_dynamic_parity_le2e6", "exact_zero_finite_tail_causal",
                          "quiet_input_le001", "single_thread_cpu_realtime", "same_training_checkpoint_and_graph"), True)
