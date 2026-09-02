import unittest

import numpy as np

from remix.gate import CLASSES, FEATURES, active, mask
from remix.train_gate import _threshold, _thresholds


class GateTests(unittest.TestCase):
    def test_family_masks_round_trip(self):
        for value in CLASSES:
            self.assertEqual(mask(active(value)), value)

    def test_feature_contract_is_fixed_and_unique(self):
        self.assertEqual(len(FEATURES), 39)
        self.assertEqual(len(FEATURES), len(set(FEATURES)))

    def test_threshold_excludes_every_calibration_error(self):
        labels = np.asarray([0, 1, 2, 3])
        predictions = np.asarray([0, 0, 2, 1])
        confidence = np.asarray([0.9, 0.8, 0.95, 0.7])
        threshold = _threshold(labels, predictions, confidence)
        selected = confidence >= threshold
        self.assertGreater(threshold, 0.8)
        self.assertFalse(np.any(selected & (labels != predictions)))

    def test_class_thresholds_do_not_let_one_class_suppress_another(self):
        labels = np.asarray([0, 1, 2, 3])
        predictions = np.asarray([0, 0, 2, 1])
        confidence = np.asarray([0.6, 0.99, 0.7, 0.8])
        thresholds = _thresholds(labels, predictions, confidence)
        selected = confidence >= np.asarray([thresholds[int(value)] for value in predictions])
        self.assertFalse(np.any(selected & (labels != predictions)))
        self.assertTrue(selected[2])


if __name__ == "__main__":
    unittest.main()
