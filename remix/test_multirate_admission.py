from copy import deepcopy
import unittest

import torch

from .multirate_admission import validate_evidence, validate_structure
from .multirate_fuzz import MultirateFuzz
from .test_dfz_admission import digest, fixtures
from .widen_multirate_fuzz import WIDE_GEOMETRY


def complete_evidence():
    cal, pytorch, deployed, _, checkpoint, graph = fixtures()
    entry, training = digest("calibration-entry"), digest("trained-wide-report")
    for report in (cal, pytorch, deployed):
        report.update(architecture="causal-multirate-gcn", geometry=WIDE_GEOMETRY.copy(), admitted=False,
                      source_and_audio_reverified=True, training_report_sha256=training)
        report["data_manifest"]["corpus_root"] = "/synthetic-corpus-not-opened"
    cal["runtime"] = pytorch["runtime"] = "multirate-pytorch-cpu"
    pytorch["calibration_entry_sha256"] = deployed["calibration_entry_sha256"] = entry
    runtime = {"purpose": "trained-wide-core-audit", "runtime_passed": True, "calibration_passed": True,
               "checkpoint_sha256": checkpoint, "graph_sha256": graph, "parent_report_sha256": training,
               "geometry": WIDE_GEOMETRY.copy(), "source_and_audio_reverified": True,
               "admitted": False, "official_eval_opened": False, "physical_audio_devices_used": False,
               "source_audio_modified": False, "automatic_gain_or_normalization": False,
               "real_audio_cases": [{"name": row["relative_path"].split("/")[-1], "parity_max": 1e-6, "onnx_stream_max": 0.}
                                    for row in cal["data_manifest"]["records"]],
               "dynamic_probes": [{"batch": batch, "frames": frames, "parity_max": 1e-6, "onnx_stream_max": 0.,
                                   "cpu_stream_max": 1e-7, "phase_exact": True, "inputs_unchanged": True}
                                  for batch, frames in ((2, 6173), (1, 144000), (1, 1440000))],
               "safety": {backend: {"zero_peak": 0., "expired_tail_peak": 0., "future_prefix_error": 0., "quiet_peak": 1e-4}
                          for backend in ("cpu", "onnx")},
               "single_thread_rtf_including_validation": {backend: {"full": [.2] * 3, "stream1024": [.2] * 3}
                                                          for backend in ("cpu", "onnx")}}
    return cal, pytorch, deployed, runtime, checkpoint, graph, entry


class MultirateAdmissionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_complete_evidence(self):
        evidence = complete_evidence()
        before = deepcopy(evidence)
        self.assertTrue(all(validate_evidence(*evidence).values()))
        self.assertEqual(evidence, before)

    def test_each_waveform_gate_is_unchanged(self):
        for index, field in ((0, "calibration"), (1, "official_eval"), (2, "official_eval")):
            for name, value in (("absolute_peak_error_p95", .02000001), ("global_esr", .05000001),
                                ("spectral_loss_relative_improvement", .2999999), ("stream_max_absolute_error", 2.00001e-6)):
                evidence = list(complete_evidence())
                evidence[index][field][name] = value
                with self.assertRaises(ValueError):
                    validate_evidence(*evidence)

    def test_failed_or_initialization_only_runtime_never_admitted(self):
        for key, value in (("purpose", "function-preserving-widen-preflight"), ("runtime_passed", False),
                           ("calibration_passed", False), ("source_and_audio_reverified", False)):
            evidence = list(complete_evidence())
            evidence[3][key] = value
            with self.assertRaises(ValueError):
                validate_evidence(*evidence)

    def test_runtime_empty_missing_nonfinite_or_regressed(self):
        changes = [lambda r: r.update(safety={}), lambda r: r.update(single_thread_rtf_including_validation={}),
                   lambda r: r["real_audio_cases"].pop(), lambda r: r["dynamic_probes"].pop(),
                   lambda r: r["real_audio_cases"][0].update(parity_max=2.00001e-6),
                   lambda r: r["real_audio_cases"][0].update(parity_max=float("nan")),
                   lambda r: r["safety"]["onnx"].update(quiet_peak=.00101),
                   lambda r: r["safety"]["cpu"].update(expired_tail_peak=1e-12),
                   lambda r: r["single_thread_rtf_including_validation"]["onnx"].update(stream1024=[.1, .1, 1.])]
        for change in changes:
            evidence = list(complete_evidence())
            change(evidence[3])
            with self.assertRaises(ValueError):
                validate_evidence(*evidence)

    def test_source_checkpoint_and_group_binding(self):
        for index in (0, 1, 2, 3):
            evidence = list(complete_evidence())
            evidence[index]["checkpoint_sha256"] = digest("wrong")
            with self.assertRaises(ValueError):
                validate_evidence(*evidence)
        evidence = list(complete_evidence())
        evidence[2]["official_eval"]["by_first_control"]["0"]["absolute_peak_error_p95"] = .030001
        with self.assertRaises(ValueError):
            validate_evidence(*evidence)

    def payload(self):
        return {"geometry": WIDE_GEOMETRY.copy(), "architecture": "causal-multirate-gcn",
                "experimental_schema": 1, "device": "dfz", "sample_rate": 48000}

    def test_finite_structure_has_its_own_proof_not_lstm_norm(self):
        model = MultirateFuzz(**WIDE_GEOMETRY)
        proof = validate_structure(model, self.payload())
        self.assertFalse(proof["learned_recurrent_matrices"])
        self.assertEqual(proof["audio_receptive_field_frames"], 2047)
        self.assertEqual(proof["zero_audio_tail_expiry_frames"], 2046)
        self.assertEqual(proof["state_tensor_bytes_mono"], 393160)
        self.assertIn("not a fidelity metric", proof["bound_scope"])

    def test_unreviewed_bias_recurrence_or_geometry_rejected(self):
        for change in (lambda m: setattr(m.audio.input, "bias", torch.nn.Parameter(torch.zeros(32))),
                       lambda m: setattr(m, "unexpected", torch.nn.LSTM(1, 1)),
                       lambda m: setattr(m.audio.convolutions[0], "stride", (2,)),
                       lambda m: setattr(m.audio.output, "in_channels", 31),
                       lambda m: m.state_widths.__setitem__(0, 1)):
            model = MultirateFuzz(**WIDE_GEOMETRY)
            change(model)
            with self.assertRaises(ValueError):
                validate_structure(model, self.payload())

    def test_nonfinite_weight_rejected(self):
        model = MultirateFuzz(**WIDE_GEOMETRY)
        with torch.no_grad():
            model.audio.output.weight[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            validate_structure(model, self.payload())


if __name__ == "__main__":
    unittest.main()
