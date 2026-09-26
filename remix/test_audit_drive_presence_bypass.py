from __future__ import annotations

import unittest

import numpy as np

from .audit_drive_presence_bypass import _aggregate, _bypass_metrics


class DrivePresenceBypassAuditTests(unittest.TestCase):
    def test_exact_stage_is_safe(self) -> None:
        time = np.arange(48_000, dtype=np.float32) / 48_000.0
        source = (0.1 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)
        result = _bypass_metrics(source, source.copy(), 48_000)
        self.assertTrue(result["safe_to_bypass"])

    def test_level_shift_is_not_hidden_by_scale_invariant_residual(self) -> None:
        source = np.linspace(-0.2, 0.2, 48_000, dtype=np.float32)
        result = _bypass_metrics(source, source * 1.5, 48_000)
        self.assertLess(result["metrics"]["scale_invariant_residual_db"], -20.0)
        self.assertFalse(result["gates"]["gain_shift"])
        self.assertFalse(result["safe_to_bypass"])

    def test_aggregate_requires_coverage_and_safe_fraction(self) -> None:
        row = {"safe_to_bypass": True, "metrics": {
            "scale_invariant_residual_db": -40.0,
            "gain_shift_db": 0.0,
            "spectral_error": 0.0,
            "high_band_error": 0.0,
            "transient_error": 0.0,
            "dynamics_error": 0.0,
        }}
        self.assertFalse(_aggregate([row] * 9)["negative_detection_may_bypass"])
        aggregate = _aggregate([row] * 10)
        self.assertTrue(aggregate["negative_detection_may_bypass"])
        self.assertEqual(aggregate["metric_worst"]["scale_invariant_residual_db"], -40.0)
        self.assertEqual(aggregate["metric_worst"]["absolute_gain_shift_db"], 0.0)


if __name__ == "__main__":
    unittest.main()
