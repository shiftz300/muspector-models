import unittest

import numpy as np

from remix.cross import align


class CrossTests(unittest.TestCase):
    def test_alignment_recovers_delay_without_mutating_sources(self):
        dry = np.random.default_rng(7).normal(0, .1, 4096).astype(np.float32)
        wet = np.r_[np.zeros(73, np.float32), dry[:-73]]
        before = dry.copy(), wet.copy()
        clean, affected, lag, score = align(dry, wet, 128)
        self.assertEqual(lag, 73)
        self.assertGreater(score, .99)
        np.testing.assert_array_equal(clean, affected)
        np.testing.assert_array_equal(dry, before[0]); np.testing.assert_array_equal(wet, before[1])


if __name__ == "__main__":
    unittest.main()
