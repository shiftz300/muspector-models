"""Synthetic, audio-device-free tests for strict DFZ forward admission."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
import torch

from .evaluate_asrnn_effect import (_quality_gate, assert_artifacts_unchanged, evaluate,
                                    manifest_sha256, source_record)
from .promote_dfz_forward import (
    ALL_TAKES,
    CALIBRATION_TAKES,
    CONTROL_GROUPS,
    CONTROL_VALUES,
    FIT_TAKES,
    SCORED_FRAMES_PER_FILE,
    validate_artifacts,
    validate_charge_banks,
    validate_evidence,
    validate_records,
    verify_source_files,
)
from .stable_charge_residual import ChargeBank


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def records(split: str, takes: set[int]) -> list[dict]:
    rows = []
    for a, b in itertools.product(CONTROL_VALUES, repeat=2):
        for take in sorted(takes):
            relative = f"datasets/dfz/{split}/{a},{b},{take}.wav"
            rows.append({"relative_path": relative, "file_sha256": digest(relative),
                         "dry_sha256": digest("dry:" + relative), "pair_sha256": digest("pair:" + relative),
                         "frames": 144_000, "sample_rate": 48_000, "channels": 2,
                         "controls": [a / 100.0, b / 100.0], "target_peak_after_warmup": 0.25})
    return sorted(rows, key=lambda row: row["relative_path"])


def measurements(count: int) -> dict:
    return {
        "examples": count, "audible_examples": count, "quiet_examples": 0,
        "quiet_dataset_gate_vacuous": True, "scored_frames": count * SCORED_FRAMES_PER_FILE,
        "global_esr": 0.005, "global_esr_relative_improvement": 0.99,
        "global_mae_relative_improvement": 0.90, "preemphasis_esr_relative_improvement": 0.90,
        "spectral_loss_relative_improvement": 0.80, "mean_per_file_esr": 0.01,
        "median_per_file_esr": 0.005, "p95_per_file_esr": 0.02,
        "peak_ratio_median": 1.0, "peak_ratio_p95": 1.01,
        "absolute_peak_error_p95": 0.015, "quiet_prediction_peak_maximum": 0.0,
        "static_silence_max_absolute_output": 0.0, "dynamic_control_silence_max_absolute_output": 0.0,
        "stream_max_absolute_error": 1e-7,
        "by_first_control": {str(value): {"files": count // 3, "mean_esr": 0.01, "absolute_peak_error_p95": 0.02}
                             for value in CONTROL_VALUES},
        "by_control_tuple": {key: {"files": count // 9, "mean_esr": 0.01, "absolute_peak_error_p95": 0.02}
                             for key in CONTROL_GROUPS},
    }


def fixtures() -> tuple[dict, dict, dict, dict, str, str]:
    checkpoint, graph = digest("checkpoint"), digest("graph")
    train = {"fit": records("train", FIT_TAKES), "calibration": records("train", CALIBRATION_TAKES)}
    report = {
        "schema": 1, "device": "dfz", "accepted": True, "checkpoint_sha256": checkpoint,
        "onnx_sha256": None, "runtime": "stable", "compute_device": "cpu", "control_names": ["blend", "filter"],
        "quality_policy": {"source_files_read_only": True, "automatic_normalization": False,
                           "automatic_limiting": False, "lossy_reencoding": False, "physical_audio_devices_used": False},
    }
    calibration = deepcopy(report)
    calibration["calibration"] = measurements(54)
    calibration["data_manifest"] = {
        "schema": 1, "device": "dfz", "official_split": "train", "partition": "calibration",
        "records": train["calibration"], "sha256": manifest_sha256(train["calibration"]),
        "warmup_frames_excluded_from_scoring": 1_024, "full_causal_prefix_processed": True,
    }
    calibration["train_partition_manifest"] = {
        "schema": 1, "device": "dfz", "official_split": "train",
        "selection": "take_id_modulo_5_equals_0_for_calibration", "partitions": train,
        "sha256": manifest_sha256(train), "fit_audio_used_for_model_evaluation": False,
        "independent_session_holdout": False,
    }
    pytorch = deepcopy(report)
    development = records("eval", ALL_TAKES)
    pytorch["official_eval"] = measurements(288)
    pytorch["data_manifest"] = {
        "schema": 1, "device": "dfz", "official_split": "eval", "partition": "development",
        "records": development, "sha256": manifest_sha256(development),
        "warmup_frames_excluded_from_scoring": 1_024, "full_causal_prefix_processed": True,
    }
    deployed = deepcopy(pytorch)
    deployed.update(runtime="onnxruntime-cpu", onnx_sha256=graph)
    runtime = {
        "checkpoint_sha256": checkpoint, "onnx_sha256": graph, "backend": "onnxruntime",
        "compute_device": "cpu", "threads": 1, "float_precision": "float32", "quantization": False,
        "full_realtime_factor": 0.4, "streaming_realtime_factor": 0.45, "streaming_block_frames": 1_024,
        "torch_onnx_max_absolute_error": 1e-7, "dynamic_control_signal_stream_max_error": 1e-7,
        "dynamic_control_silence_peak": 0.0, "quiet_input_peak": 0.0001,
        "additional_safety_passed": True, "cpu_realtime_passed": True, "physical_audio_devices_used": False,
        "ui_integration_allowed": False, "automatic_normalization": False, "automatic_limiting": False,
    }
    return calibration, pytorch, deployed, runtime, checkpoint, graph


class DFZAdmissionTests(unittest.TestCase):
    def test_fixed_charge_bank_has_its_own_strict_ema_bound(self):
        audits = validate_charge_banks(ChargeBank())
        self.assertEqual(len(audits), 1)
        self.assertGreater(audits[0]["state_transition_infinity_norm"], 0.995)
        self.assertLess(audits[0]["state_transition_infinity_norm"], 1.0)
        self.assertTrue(audits[0]["nonnegative_complementary_input_verified"])
        self.assertEqual(audits[0]["zero_initial_state_reachable_interval"], [0.0, 1.0])

    def test_charge_bank_rejects_unreviewed_structure_and_trainable_weights(self):
        for name, value in (("nonlinearity", "tanh"), ("input_size", 2), ("num_layers", 2),
                            ("bidirectional", True), ("batch_first", False), ("dropout", 0.1)):
            with self.subTest(name=name):
                bank = ChargeBank()
                setattr(bank.cell, name, value)
                with self.assertRaisesRegex(ValueError, "charge bank"):
                    validate_charge_banks(bank)
        bank = ChargeBank()
        bank.cell.weight_hh_l0.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "fixed finite"):
            validate_charge_banks(bank)
        bank = ChargeBank()
        bank.cell.weight_ih_l0 = torch.nn.Parameter(torch.zeros(bank.width, 2), requires_grad=False)
        with self.assertRaisesRegex(ValueError, "parameter geometry"):
            validate_charge_banks(bank)
        with self.assertRaisesRegex(ValueError, "outside the audited"):
            validate_charge_banks(torch.nn.RNN(1, 5, nonlinearity="relu"))

    def test_charge_bank_rejects_corrupted_coefficients_and_biases(self):
        changes = (("weight_hh_l0", (0, 1), 1e-8), ("weight_hh_l0", (0, 0), 0.0),
                   ("weight_hh_l0", (0, 0), 1.0), ("weight_hh_l0", (0, 0), -0.1),
                   ("weight_ih_l0", (0, 0), -1e-8), ("weight_ih_l0", (0, 0), 0.1),
                   ("bias_ih_l0", (0,), 1e-8), ("bias_hh_l0", (0,), 1e-8),
                   ("weight_hh_l0", (0, 0), float("nan")))
        for name, index, value in changes:
            with self.subTest(name=name, index=index, value=value):
                bank = ChargeBank()
                with torch.no_grad():
                    getattr(bank.cell, name)[index] = value
                with self.assertRaisesRegex(ValueError, "charge bank"):
                    validate_charge_banks(bank)

    def test_charge_bank_does_not_weaken_original_lstm_candidate_gate(self):
        model = torch.nn.Module()
        model.control_count = 2
        model.base = torch.nn.LSTM(3, 2, batch_first=True)
        model.bank = ChargeBank()
        with torch.no_grad():
            model.base.weight_hh_l0.zero_()
            model.base.weight_hh_l0[4:6].copy_(torch.eye(2) * 0.995)
        graph = onnx.helper.make_model(onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["dry"], ["rendered"])], "synthetic-charge-admission",
            [onnx.helper.make_tensor_value_info("dry", onnx.TensorProto.FLOAT, [1, 1])],
            [onnx.helper.make_tensor_value_info("rendered", onnx.TensorProto.FLOAT, [1, 1])]))
        for key, value in {"checkpoint_sha256": digest("candidate"), "sample_rate": "48000",
                           "control_count": "2", "precision": "float32"}.items():
            entry = graph.metadata_props.add()
            entry.key, entry.value = key, value
        payload = {"device": "dfz", "sample_rate": 48_000}
        with patch("remix.promote_dfz_forward.load_stable_effect", return_value=(model, payload)), \
             patch("remix.promote_dfz_forward.onnx.load", return_value=graph):
            _, _, lstm_norms, charge_audits = validate_artifacts(Path("synthetic.pt"), Path("synthetic.onnx"), digest("candidate"))
            self.assertLessEqual(max(lstm_norms), 0.995001)
            self.assertGreater(charge_audits[0]["state_transition_infinity_norm"], 0.995001)
            with torch.no_grad():
                model.base.weight_hh_l0[4, 0] = 0.996
            with self.assertRaisesRegex(ValueError, "candidate recurrent matrix"):
                validate_artifacts(Path("synthetic.pt"), Path("synthetic.onnx"), digest("candidate"))

    def test_complete_cpu_evidence_is_accepted(self):
        checks = validate_evidence(*fixtures())
        self.assertTrue(all(checks.values()))
        self.assertIn("independent_quiet_and_dynamic_stream_probes", checks)

    def test_old_calibration_summary_cannot_be_promoted(self):
        args = list(fixtures())
        del args[0]["data_manifest"]
        with self.assertRaisesRegex(ValueError, "manifest"):
            validate_evidence(*args)

    def test_frozen_quality_gate_has_not_been_weakened(self):
        values = measurements(288)
        self.assertTrue(_quality_gate(values))
        for key, value in (("absolute_peak_error_p95", 0.02000001), ("global_esr", 0.05000001),
                           ("spectral_loss_relative_improvement", 0.2999999),
                           ("static_silence_max_absolute_output", 1e-20),
                           ("dynamic_control_silence_max_absolute_output", 1e-20),
                           ("stream_max_absolute_error", 2.00001e-6),
                           ("quiet_prediction_peak_maximum", 0.00100001)):
            with self.subTest(key=key):
                changed = deepcopy(values)
                changed[key] = value
                self.assertFalse(_quality_gate(changed))

    def test_each_quality_report_must_pass_independently(self):
        for index, field in ((0, "calibration"), (1, "official_eval"), (2, "official_eval")):
            with self.subTest(index=index):
                args = list(fixtures())
                args[index][field]["absolute_peak_error_p95"] = 0.02000001
                with self.assertRaisesRegex(ValueError, "quality"):
                    validate_evidence(*args)

    def test_worst_blend_gate_is_not_hidden_by_global_average(self):
        args = list(fixtures())
        args[2]["official_eval"]["by_first_control"]["0"]["absolute_peak_error_p95"] = 0.0300001
        with self.assertRaisesRegex(ValueError, "worst Blend"):
            validate_evidence(*args)

    def test_checkpoint_and_graph_provenance_must_match(self):
        for index in range(4):
            with self.subTest(checkpoint_report=index):
                args = list(fixtures())
                args[index]["checkpoint_sha256"] = digest("other")
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    validate_evidence(*args)
        for index in (2, 3):
            with self.subTest(graph_report=index):
                args = list(fixtures())
                args[index]["onnx_sha256"] = digest("other graph")
                with self.assertRaises(ValueError):
                    validate_evidence(*args)

    def test_cpu_onnx_safety_and_real_time_are_mandatory(self):
        changes = (("backend", "pytorch"), ("compute_device", "mps"), ("threads", 2),
                   ("float_precision", "float16"), ("quantization", True),
                   ("full_realtime_factor", 1.0), ("streaming_realtime_factor", 1.0),
                   ("streaming_block_frames", 2_048), ("torch_onnx_max_absolute_error", 2.01e-6),
                   ("dynamic_control_signal_stream_max_error", 2.01e-6),
                   ("dynamic_control_silence_peak", 1e-20), ("quiet_input_peak", 0.00101),
                   ("full_realtime_factor", float("nan")), ("cpu_realtime_passed", False),
                   ("physical_audio_devices_used", True))
        for key, value in changes:
            with self.subTest(key=key, value=value):
                args = list(fixtures())
                args[3][key] = value
                with self.assertRaises(ValueError):
                    validate_evidence(*args)

    def test_pytorch_reference_may_not_use_gpu_or_onnx(self):
        for index in (0, 1):
            for key, value in (("compute_device", "mps"), ("runtime", "onnxruntime-cpu")):
                with self.subTest(index=index, key=key):
                    args = list(fixtures())
                    args[index][key] = value
                    with self.assertRaises(ValueError):
                        validate_evidence(*args)

    def test_complete_nine_tuple_and_three_blend_counts_are_required(self):
        for field, key in (("by_control_tuple", "0,0"), ("by_first_control", "0")):
            args = list(fixtures())
            args[2]["official_eval"][field][key]["files"] -= 1
            with self.assertRaisesRegex(ValueError, "counts"):
                validate_evidence(*args)
        for key, value in (("examples", 287), ("scored_frames", 288 * SCORED_FRAMES_PER_FILE - 1)):
            args = list(fixtures())
            args[2]["official_eval"][key] = value
            with self.assertRaisesRegex(ValueError, "coverage"):
                validate_evidence(*args)

    def test_same_filenames_cannot_disguise_the_wrong_split(self):
        train = records("train", ALL_TAKES)
        development = records("eval", ALL_TAKES)
        self.assertEqual([Path(row["relative_path"]).name for row in train],
                         [Path(row["relative_path"]).name for row in development])
        self.assertNotEqual(manifest_sha256(train), manifest_sha256(development))
        with self.assertRaisesRegex(ValueError, "split/path"):
            validate_records(train, "eval", ALL_TAKES)

    def test_manifest_tampering_and_source_mismatch_are_rejected(self):
        args = list(fixtures())
        args[1]["data_manifest"]["records"][0]["file_sha256"] = digest("changed")
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            validate_evidence(*args)
        args[1]["data_manifest"]["sha256"] = manifest_sha256(args[1]["data_manifest"]["records"])
        with self.assertRaisesRegex(ValueError, "different source"):
            validate_evidence(*args)

    def test_fit_calibration_and_development_dry_overlap_is_rejected(self):
        args = list(fixtures())
        overlap = args[0]["train_partition_manifest"]["partitions"]["fit"][0]["dry_sha256"]
        for index in (1, 2):
            args[index]["data_manifest"]["records"][0]["dry_sha256"] = overlap
            args[index]["data_manifest"]["sha256"] = manifest_sha256(args[index]["data_manifest"]["records"])
        with self.assertRaisesRegex(ValueError, "Dry overlap"):
            validate_evidence(*args)

    def test_fit_partition_cannot_be_truncated_or_changed(self):
        args = list(fixtures())
        train = args[0]["train_partition_manifest"]
        train["partitions"]["fit"].pop()
        train["sha256"] = manifest_sha256(train["partitions"])
        with self.assertRaisesRegex(ValueError, "record count"):
            validate_evidence(*args)

    def test_quiet_empty_set_is_disclosed_not_claimed_as_measured(self):
        for field, replacement in (("quiet_examples", None), ("audible_examples", 287),
                                    ("quiet_dataset_gate_vacuous", False)):
            args = list(fixtures())
            if replacement is None:
                del args[2]["official_eval"][field]
            else:
                args[2]["official_eval"][field] = replacement
            with self.assertRaisesRegex(ValueError, "quiet|empty quiet"):
                validate_evidence(*args)
        args = list(fixtures())
        args[3]["quiet_input_peak"] = 0.002
        with self.assertRaisesRegex(ValueError, "quiet"):
            validate_evidence(*args)

    def test_geometry_prefix_and_control_semantics_cannot_change(self):
        for key, value in (("frames", 48_000), ("sample_rate", 44_100), ("channels", 1),
                           ("controls", [0.5, 0.5]), ("relative_path", "../dfz/eval/0,0,1.wav")):
            with self.subTest(key=key):
                rows = records("eval", ALL_TAKES)
                rows[0][key] = value
                with self.assertRaises(ValueError):
                    validate_records(rows, "eval", ALL_TAKES)
        args = list(fixtures())
        args[2]["data_manifest"]["full_causal_prefix_processed"] = False
        with self.assertRaisesRegex(ValueError, "full-prefix"):
            validate_evidence(*args)

    def test_nonfinite_metrics_are_rejected(self):
        args = list(fixtures())
        args[2]["official_eval"]["by_control_tuple"]["0,0"]["mean_esr"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_evidence(*args)

    def test_source_record_hashes_do_not_modify_audio_arrays(self):
        with tempfile.TemporaryDirectory(prefix="dfz-manifest-test-") as temporary:
            corpus = Path(temporary)
            path = corpus / "datasets/dfz/train/0,0,5.wav"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"synthetic-source-bytes-not-played")
            dry = np.linspace(-0.1, 0.1, 2_048, dtype=np.float32)
            wet = dry * np.float32(0.5)
            dry.setflags(write=False)
            wet.setflags(write=False)
            before = dry.tobytes(), wet.tobytes()
            row = source_record(path, corpus, dry, wet, np.array([0.0, 0.0], dtype=np.float32))
            self.assertEqual(row["file_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(row["dry_sha256"], hashlib.sha256(before[0]).hexdigest())
            self.assertEqual(row["pair_sha256"], hashlib.sha256(before[0] + before[1]).hexdigest())
            self.assertEqual((dry.tobytes(), wet.tobytes()), before)

    def test_on_disk_source_drift_is_rejected_without_decoding_audio(self):
        args = fixtures()
        with tempfile.TemporaryDirectory(prefix="dfz-source-drift-test-") as temporary:
            corpus = Path(temporary)
            all_rows = (args[0]["train_partition_manifest"]["partitions"]["fit"]
                        + args[0]["train_partition_manifest"]["partitions"]["calibration"]
                        + args[1]["data_manifest"]["records"])
            all_paths = [corpus / row["relative_path"] for row in all_rows]
            with patch("remix.promote_dfz_forward.effect_files", side_effect=lambda root, kind, split:
                       [path for path in all_paths if path.parent.name == split]), \
                 patch("remix.promote_dfz_forward.file_sha256", return_value=digest("changed bytes")):
                with self.assertRaisesRegex(ValueError, "source bytes changed"):
                    verify_source_files(args[0], args[1], corpus)

    def test_concurrent_artifact_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="dfz-artifact-drift-test-") as temporary:
            checkpoint, graph = Path(temporary) / "model.pt", Path(temporary) / "model.onnx"
            checkpoint.write_bytes(b"checkpoint")
            graph.write_bytes(b"graph")
            assert_artifacts_unchanged(checkpoint, digest("checkpoint"), graph, digest("graph"))
            graph.write_bytes(b"replacement graph")
            with self.assertRaisesRegex(ValueError, "graph changed"):
                assert_artifacts_unchanged(checkpoint, digest("checkpoint"), graph, digest("graph"))
            checkpoint.write_bytes(b"replacement checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                assert_artifacts_unchanged(checkpoint, digest("checkpoint"))

    def test_evaluator_records_actual_synthetic_groups_and_empty_quiet_set(self):
        class Identity(torch.nn.Module):
            control_count = 2

            def forward(self, dry, controls, state=None):
                return dry, state

        with tempfile.TemporaryDirectory(prefix="dfz-evaluator-test-") as temporary:
            corpus = Path(temporary)
            checkpoint = corpus / "model.pt"
            checkpoint.write_bytes(b"synthetic-checkpoint")
            paths = [corpus / f"datasets/dfz/eval/{name}.wav" for name in ("0,0,1", "50,100,1")]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(path.name.encode())
            dry = (np.sin(np.arange(5_120, dtype=np.float32) * 0.1) * 0.02).astype(np.float32)

            def read_synthetic(path, device_key):
                a, b, _ = map(int, path.stem.split(","))
                return dry, dry.copy(), np.array([a / 100, b / 100], dtype=np.float32)

            with patch("remix.evaluate_asrnn_effect.load_stable_effect", return_value=(Identity(), {"device": "dfz"})), \
                 patch("remix.evaluate_asrnn_effect.effect_files", return_value=paths), \
                 patch("remix.evaluate_asrnn_effect.read_effect_pair", side_effect=read_synthetic):
                report = evaluate(checkpoint, corpus, "dfz", compute="cpu")
            metrics = report["official_eval"]
            self.assertTrue(report["accepted"])
            self.assertEqual(metrics["quiet_examples"], 0)
            self.assertTrue(metrics["quiet_dataset_gate_vacuous"])
            self.assertEqual(metrics["audible_examples"], 2)
            self.assertEqual(metrics["scored_frames"], 2 * 4_096)
            self.assertEqual(set(metrics["by_control_tuple"]), {"0,0", "50,100"})
            self.assertTrue(all(value["files"] == 1 for value in metrics["by_control_tuple"].values()))
            self.assertEqual(report["data_manifest"]["sha256"], manifest_sha256(report["data_manifest"]["records"]))
            self.assertEqual(report["checkpoint_sha256"], digest("synthetic-checkpoint"))


if __name__ == "__main__":
    unittest.main()
