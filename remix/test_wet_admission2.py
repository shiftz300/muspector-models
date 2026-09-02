import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from .train_wet_admission2 import CONTROL_WIDTHS, FAMILIES, _rows, features


ROOT = Path(__file__).resolve().parents[1]


class WetAdmission2Tests(unittest.TestCase):
    def test_features_are_finite_and_fixed_width(self):
        left = features(np.zeros(48000, dtype=np.float32))
        right = features(np.random.default_rng(4).normal(0.0, 0.01, 48000).astype(np.float32))
        self.assertEqual(left.shape, right.shape)
        self.assertGreater(len(left), 32)
        self.assertTrue(np.isfinite(right).all())

    def test_rows_keep_family_and_knob_packages_separable(self):
        rows = _rows(ROOT, "development", 1, 20260911)
        self.assertEqual(Counter(row["family"] for row in rows), Counter({name: 1 for name in FAMILIES}))
        for row in rows:
            if row["family"] in CONTROL_WIDTHS:
                self.assertEqual(len(row["controls"]), CONTROL_WIDTHS[row["family"]])
            else:
                self.assertEqual(len(row["controls"]), 0)


if __name__ == "__main__":
    unittest.main()
