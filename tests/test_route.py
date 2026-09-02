import unittest

import numpy as np

from remix.identity import LABELS
from remix.route import FEATURES, features


class RouteTests(unittest.TestCase):
    def test_features_keep_catalog_knownness_and_entropy(self):
        report = {"scores": {label: 0.1 for label in LABELS}}
        dry = np.linspace(-0.2, 0.2, 48_000, dtype=np.float32)
        wet = np.tanh(dry * 3.0).astype(np.float32)
        value = features(report, dry, wet, 48_000)
        self.assertEqual(value.shape, (len(FEATURES),))
        self.assertAlmostEqual(float(value[len(LABELS)]), 0.7, places=6)
        self.assertGreater(float(value[len(LABELS) + 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
