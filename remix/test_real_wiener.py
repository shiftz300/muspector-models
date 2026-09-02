import unittest

import numpy as np

from .real_wiener import fit, restore


class RealWienerTests(unittest.TestCase):
    def test_fit_and_restore_preserve_geometry(self):
        rng = np.random.default_rng(9)
        rows = []
        for _ in range(4):
            clean = rng.standard_normal(1024).astype(np.float32) * 0.01
            wet = np.asarray(0.8 * clean + 0.2 * np.roll(clean, 1), dtype=np.float32)
            rows.append({"source": wet, "target": clean})
        response = fit(rows, 1.0e-4)
        restored = restore(rows[0]["source"], response, 1.0)
        self.assertEqual(restored.shape, rows[0]["source"].shape)
        self.assertTrue(np.isfinite(restored).all())

    def test_invalid_strength_is_rejected(self):
        with self.assertRaises(ValueError):
            restore(np.zeros(32, dtype=np.float32), np.ones(17), 2.0)


if __name__ == "__main__":
    unittest.main()
