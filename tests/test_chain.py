import unittest
from unittest.mock import Mock, patch

import numpy as np

from remix.chain import ChainRuntime


class ChainTests(unittest.TestCase):
    def runtime(self, *, accepted=True, active=("drive",)):
        runtime = object.__new__(ChainRuntime)
        runtime.family = Mock()
        runtime.family.infer.return_value = {
            "scores": {"drive": 0.9, "delay": 0.1, "reverb": 0.1}
        }
        runtime.gate = Mock()
        runtime.gate.infer.return_value = {
            "accepted": accepted,
            "active": list(active),
            "confidence": 0.999,
            "threshold": 0.998,
        }
        runtime.knobs = Mock()
        runtime.knobs.infer.return_value = {
            "active_effects": list(("drive", "delay", "reverb")),
            "normalized_controls": [0.5] * 9,
            "bundle_sha256": "a" * 64,
        }
        runtime.order = Mock()
        runtime.order.infer.return_value = {
            "active_effects": list(active),
            "normalized_controls": [0.5] * 9,
            "bundle_sha256": "a" * 64,
        }
        runtime.knobs.controls_from_report.return_value = {"controls": {}}
        return runtime

    @patch("remix.chain.vector", return_value=np.zeros(39, dtype=np.float32))
    def test_abstention_stops_before_order_and_controls(self, _vector):
        runtime = self.runtime(accepted=False)
        audio = np.ones(8, dtype=np.float32)
        report = runtime.infer(audio, audio)
        self.assertEqual(report["decision"], "abstain")
        runtime.order.infer.assert_not_called()
        runtime.knobs.controls_from_report.assert_not_called()

    @patch("remix.chain.vector", return_value=np.zeros(39, dtype=np.float32))
    def test_accepted_family_set_reuses_order_report_for_controls(self, _vector):
        runtime = self.runtime()
        audio = np.ones(8, dtype=np.float32)
        report = runtime.infer(audio, audio)
        self.assertEqual(report["decision"], "accepted")
        runtime.order.infer.assert_called_once()
        runtime.knobs.controls_from_report.assert_called_once_with(report["order"])

    @patch("remix.chain.vector", return_value=np.zeros(39, dtype=np.float32))
    def test_source_arrays_remain_bit_identical(self, _vector):
        runtime = self.runtime()
        dry = np.linspace(-0.5, 0.5, 8, dtype=np.float32)
        wet = dry.copy()
        dry_before, wet_before = dry.copy(), wet.copy()
        runtime.infer(dry, wet)
        np.testing.assert_array_equal(dry, dry_before)
        np.testing.assert_array_equal(wet, wet_before)


if __name__ == "__main__":
    unittest.main()
