import unittest
from unittest.mock import Mock

import numpy as np

from remix.adapter import DeviceRuntime, analysis


class AdapterTests(unittest.TestCase):
    def test_analysis_resamples_and_pads_without_mutating_source(self):
        source = np.linspace(-0.5, 0.5, 48_000, dtype=np.float32)
        before = source.copy()
        result = analysis(source, 48_000, 44_100, 220_500)
        self.assertEqual(result.shape, (220_500,))
        np.testing.assert_array_equal(source, before)

    def runtime(self, chain_report):
        chain = Mock()
        chain.infer.return_value = chain_report
        adapter = Mock()
        adapter.manifest = {"device": "rat"}
        adapter.infer_controls.return_value = {
            "decision": "accepted", "reason": None, "controls": {"distortion": {"normalized": 0.5}}
        }
        return DeviceRuntime(chain, "rat", adapter), adapter

    def test_family_mismatch_stops_before_device_adapter(self):
        runtime, adapter = self.runtime({"decision": "bypass", "active": []})
        audio = np.ones(48_000, dtype=np.float32)
        report = runtime.infer(audio, audio, 48_000)
        self.assertEqual(report["decision"], "abstain")
        self.assertEqual(report["reason"], "family-mismatch")
        adapter.infer_controls.assert_not_called()

    def test_accepted_drive_routes_to_rat_adapter(self):
        runtime, adapter = self.runtime({"decision": "accepted", "active": ["drive"]})
        audio = np.ones(48_000, dtype=np.float32)
        report = runtime.infer(audio, audio, 48_000)
        self.assertEqual(report["decision"], "accepted")
        adapter.infer_controls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
