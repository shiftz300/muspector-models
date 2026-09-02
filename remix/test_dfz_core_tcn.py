"""Offline CPU tests for output-conditioned history and prediction integrity."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from .core_conditioned_tcn import CoreConditionedResidual, CoreConditionedTCN
from .dfz_frozen_predictions import tensor_sha256, verify_payload
from .train_dfz_core_tcn import selection_key
from .train_dfz_long_tcn import context_windows


class CausalBase(torch.nn.Module):
    control_count = 2
    export_state_widths = [3]

    def __init__(self):
        super().__init__()
        self.recurrent = torch.nn.RNN(1, 3, batch_first=True, bias=False)
        self.output = torch.nn.Linear(3, 1, bias=False)

    def forward(self, dry, controls, state=None):
        hidden, next_state = self.recurrent(dry[..., None], None if state is None else state[0][0])
        return self.output(hidden).squeeze(-1), ((next_state, torch.zeros_like(next_state)),)


class CoreTCNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(986)

    def test_production_zero_output_preserves_core_and_freezes_only_core(self):
        tcn = CoreConditionedTCN(16, 12)
        self.assertEqual(tcn.receptive_field, 8_191)
        model = CoreConditionedResidual(CausalBase(), tcn)
        self.assertFalse(any(value.requires_grad for value in model.base.parameters()))
        self.assertTrue(all(value.requires_grad for value in model.tcn.parameters()))
        dry, controls = torch.randn(2, 1_231) * .03, torch.rand(2, 2)
        with torch.inference_mode():
            expected, _ = model.base(dry, controls)
            output, _ = model(dry, controls)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)

    def test_complete_core_prediction_history_matches_runtime_after_weight_change(self):
        tcn = CoreConditionedTCN(3, 12)
        with torch.no_grad():
            tcn.output.weight.normal_(std=.1)
        model = CoreConditionedResidual(CausalBase(), tcn).eval()
        dry, controls = torch.randn(3, 13_000) * .03, torch.tensor([[0., 0.], [.5, 1.], [1., .5]])
        starts, scored = [0, 133, 9_177], 512
        original_dry = dry.clone()
        with torch.inference_mode():
            original, _ = model.base(dry, controls)
            complete, _ = model(dry, controls)
            x_window = context_windows(list(dry), starts, scored)
            p_window = context_windows(list(original), starts, scored)
            rendered = p_window[:, 8_190:] + tcn(x_window, p_window, controls)[0][:, 8_190:]
        expected = torch.stack([row[start:start + scored] for row, start in zip(complete, starts)])
        torch.testing.assert_close(rendered, expected, atol=2e-6, rtol=0)
        torch.testing.assert_close(dry, original_dry, atol=0, rtol=0)

    def test_actual_core_and_tcn_stream_with_dynamic_controls_and_exact_zero(self):
        tcn = CoreConditionedTCN(3, 12)
        with torch.no_grad():
            tcn.output.weight.normal_(std=.1)
        model = CoreConditionedResidual(CausalBase(), tcn).eval()
        dry, controls = torch.randn(2, 10_337) * .03, torch.rand(2, 10_337, 2)
        with torch.inference_mode():
            whole, _ = model(dry, controls)
            state, chunks = None, []
            for start, stop in ((0, 17), (17, 513), (513, 5_121), (5_121, 10_337)):
                value, state = model(dry[:, start:stop], controls[:, start:stop], state)
                chunks.append(value)
            zero, _ = model(torch.zeros_like(dry), controls)
        torch.testing.assert_close(whole, torch.cat(chunks, 1), atol=2e-6, rtol=0)
        self.assertEqual(int(torch.count_nonzero(zero)), 0)
        self.assertEqual(len(state), 13)

    def test_prediction_cache_rejects_changed_tensor_signature_or_extra_audio(self):
        predictions = torch.randn(2, 16)
        signature = {"partitions": {"fit": [{"path": "fit.wav", "sha256": "aaa"}],
                                     "calibration": [{"path": "cal.wav", "sha256": "bbb"}]},
                     "frames_per_recording": 16, "source_sha256": "source"}
        payload = {"schema": 1, "signature": signature, "predictions": predictions,
                   "prediction_sha256": tensor_sha256(predictions)}
        self.assertIs(verify_payload(payload, signature), predictions)
        changed_tensor = {**payload, "predictions": predictions.clone()}
        changed_tensor["predictions"][0, 0] += 1
        changed_signature = copy.deepcopy(signature)
        changed_signature["source_sha256"] = "changed-source"
        for candidate, expected in ((changed_tensor, signature), (payload, changed_signature),
                                    ({**payload, "wet": torch.zeros(2, 16)}, signature),
                                    ({**payload, "predictions": predictions.double()}, signature)):
            with self.assertRaises(ValueError):
                verify_payload(candidate, expected)

    def test_prediction_only_cache_roundtrip_supports_mmap_and_exclusive_creation(self):
        predictions = torch.randn(2, 16)
        signature = {"partitions": {"fit": ["fit"], "calibration": ["cal"]},
                     "frames_per_recording": 16}
        payload = {"schema": 1, "signature": signature, "predictions": predictions,
                   "prediction_sha256": tensor_sha256(predictions)}
        with tempfile.TemporaryDirectory(prefix="dfz-prediction-test-") as temporary:
            path = Path(temporary) / "predictions.pt"
            with path.open("xb") as handle:
                torch.save(payload, handle)
            restored = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            torch.testing.assert_close(verify_payload(restored, signature), predictions, atol=0, rtol=0)
            with self.assertRaises(FileExistsError):
                path.open("xb")

    def test_complete_gate_pass_beats_lower_peak_with_failing_control_group(self):
        passing = {"passes_selection_gate": True, "absolute_peak_error_p95": .019,
                   "global_esr": .01, "worst_attack_absolute_peak_error_p95": .028}
        failing = {"passes_selection_gate": False, "absolute_peak_error_p95": .018,
                   "global_esr": .01, "worst_attack_absolute_peak_error_p95": .035}
        self.assertLess(selection_key(passing), selection_key(failing))
        equally_passing = {**passing, "worst_attack_absolute_peak_error_p95": .029}
        self.assertLess(selection_key(passing), selection_key(equally_passing))


if __name__ == "__main__":
    unittest.main()
