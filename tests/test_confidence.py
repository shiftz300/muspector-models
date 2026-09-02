import unittest

import numpy as np

from remix.confidence import features, metrics, threshold


class ConfidenceTests(unittest.TestCase):
    def test_features_are_finite_and_read_only(self):
        wet = np.linspace(-.2, .2, 48000, dtype=np.float32)
        before = wet.copy(); value = features(wet)
        self.assertEqual(value.shape, (33,))
        self.assertTrue(np.isfinite(value).all())
        np.testing.assert_array_equal(wet, before)

    def test_threshold_has_zero_calibration_false_accepts(self):
        truth = np.asarray([True, True, True, False, False])
        probability = np.asarray([.9, .8, .2, .7, .1])
        selected = threshold(truth, probability)
        report = metrics(truth, probability, selected)
        self.assertEqual(report["ineligible_false_accept"], 0.)
        self.assertGreaterEqual(report["eligible_recall"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
