import unittest

import numpy as np

from .restoration_quality import _improvement, measure, summarize


class RestorationQualityTests(unittest.TestCase):
    @staticmethod
    def signals():
        rng = np.random.default_rng(17)
        clean = np.zeros(48_000, dtype=np.float64)
        for start in range(2_000, 44_000, 4_000):
            time = np.arange(900) / 48_000
            phrase = (np.sin(2 * np.pi * 220 * time) + 0.45 * np.sin(2 * np.pi * 4_800 * time))
            clean[start : start + 900] += 0.35 * phrase * np.exp(-time * 38)
        clean += rng.normal(0, 2.0e-4, len(clean))
        wet = np.tanh(clean * 5.0) * 0.45
        return clean.astype(np.float32), wet.astype(np.float32)

    def test_original_clean_target_passes_all_gates(self):
        clean, wet = self.signals()
        report = measure(wet, clean, clean)
        self.assertTrue(report["passed"])
        self.assertGreater(report["spectral_improvement"], 0.99)

    def test_identity_does_not_count_as_restoration(self):
        clean, wet = self.signals()
        self.assertFalse(measure(wet, wet, clean)["passed"])

    def test_blurred_and_reclipped_outputs_fail(self):
        clean, wet = self.signals()
        blurred = np.convolve(clean, np.ones(41) / 41, mode="same").astype(np.float32)
        clipped = np.clip(clean * 8.0, -1.0, 1.0).astype(np.float32)
        blurred_report = measure(wet, blurred, clean)
        clipped_report = measure(wet, clipped, clean)
        self.assertFalse(blurred_report["passed"])
        self.assertFalse(blurred_report["gates"]["high_band_detail"])
        self.assertFalse(clipped_report["passed"])
        self.assertFalse(clipped_report["gates"]["no_new_clipping"])

    def test_summary_needs_broad_per_example_success(self):
        clean, wet = self.signals()
        report = summarize([wet, wet], [clean, wet], [clean, clean])
        self.assertEqual(report["pass_fraction"], 0.5)

    def test_improvement_is_bounded_when_wet_error_is_nearly_zero(self):
        self.assertEqual(_improvement(0.0, 0.0), 0.0)
        self.assertAlmostEqual(_improvement(1.0e-12, 1.0), -1.0)
        self.assertAlmostEqual(_improvement(1.0, 1.0e-12), 1.0)
        self.assertLessEqual(abs(_improvement(1.0e-12, 2.0e-12)), 1.0)


if __name__ == "__main__":
    unittest.main()
