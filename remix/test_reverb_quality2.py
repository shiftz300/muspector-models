import unittest

import numpy as np
from scipy.signal import fftconvolve

from .reverb_quality2 import measure, summarize


class ReverbQuality2Tests(unittest.TestCase):
    def _release_fixture(self):
        rng = np.random.default_rng(71)
        clean = np.zeros(48_000, dtype=np.float32)
        clean[:12_000] = rng.normal(0.0, 0.05, 12_000)
        impulse = np.zeros(16_000, dtype=np.float32)
        impulse[0] = 0.7
        frames = np.arange(1, len(impulse))
        impulse[1:] = 0.01 * np.exp(-frames / 3_000.0) * np.sin(frames / 9.0)
        wet = fftconvolve(clean, impulse, mode="full")[: len(clean)].astype(np.float32)
        return wet, clean

    def test_measurable_clean_restoration_is_effective(self):
        wet, clean = self._release_fixture()
        row = measure(wet, clean, clean)
        self.assertTrue(row["evidence_measurable"], row)
        self.assertEqual(row["decision"], "effective-restoration")
        self.assertTrue(row["passed"])

    def test_unmeasurable_input_requires_exact_bypass(self):
        frames = np.arange(24_000)
        clean = (0.03 * np.sin(2.0 * np.pi * frames / 97.0)).astype(np.float32)
        bypass = measure(clean, clean.copy(), clean)
        altered = measure(clean, clean * np.float32(0.999), clean)
        self.assertEqual(bypass["decision"], "hard-bypass")
        self.assertTrue(bypass["passed"])
        self.assertEqual(altered["decision"], "unsafe-or-ineffective-processing")
        self.assertFalse(altered["passed"])

    def test_summary_rejects_one_unsafe_unmeasurable_example(self):
        wet, clean = self._release_fixture()
        effective = measure(wet, clean, clean)
        continuous = np.full(24_000, 0.02, dtype=np.float32)
        unsafe = measure(continuous, continuous * np.float32(0.99), continuous)
        report = summarize([effective, unsafe])
        self.assertFalse(report["accepted"])
        self.assertFalse(report["gates"]["every_example_safe"])
        self.assertFalse(report["gates"]["unmeasurable_hard_bypass"])


if __name__ == "__main__":
    unittest.main()
