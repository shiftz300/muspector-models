import unittest

import numpy as np

from .reverb_quality import measure, summarize


class ReverbQualityTests(unittest.TestCase):
    @staticmethod
    def signals():
        clean = np.zeros(48000, dtype=np.float32)
        clean[4000:4400] = np.hanning(400).astype(np.float32)
        wet = clean.copy()
        for delay, gain in ((2400, 0.5), (4800, 0.25), (7200, 0.125)):
            wet[delay:] += clean[:-delay] * gain
        return clean, wet

    def test_clean_restoration_passes_real_tail_gate(self):
        clean, wet = self.signals()
        report = measure(wet, clean, clean)
        self.assertTrue(report["eligible"])
        self.assertTrue(report["passed"])
        self.assertGreaterEqual(report["tail_excess_reduction"], 0.99)

    def test_identity_and_added_reverb_fail(self):
        clean, wet = self.signals()
        self.assertFalse(measure(wet, wet, clean)["passed"])
        added = wet.copy(); added[9600:] += clean[:-9600] * 0.2
        self.assertFalse(measure(wet, added, clean)["passed"])

    def test_summary_requires_half_the_eligible_examples(self):
        clean, wet = self.signals()
        report = summarize([wet, wet], [clean, wet], [clean, clean])
        self.assertEqual(report["pass_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
