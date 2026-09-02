import unittest

import numpy as np

from .audible import measure


class AudibleTests(unittest.TestCase):
    def test_identity_is_rejected(self):
        clean = np.linspace(-1.0, 1.0, 128)
        wet = clean * 0.5
        report = measure(wet, wet.copy(), clean)
        self.assertFalse(report["audible"])
        self.assertEqual(report["closure"], 0.0)
        self.assertEqual(report["correction_ratio"], 0.0)

    def test_material_correction_is_accepted(self):
        clean = np.linspace(-1.0, 1.0, 128)
        wet = clean * 0.5
        restored = wet + 0.75 * (clean - wet)
        report = measure(wet, restored, clean)
        self.assertTrue(report["audible"])
        self.assertAlmostEqual(report["closure"], 0.75)
        self.assertAlmostEqual(report["correction_ratio"], 0.75)
        self.assertAlmostEqual(report["correction_direction"], 1.0)

    def test_wrong_direction_is_rejected(self):
        clean = np.linspace(-1.0, 1.0, 128)
        wet = clean * 0.5
        restored = wet - 0.5 * (clean - wet)
        report = measure(wet, restored, clean)
        self.assertFalse(report["audible"])
        self.assertLess(report["correction_direction"], 0.0)


if __name__ == "__main__":
    unittest.main()
