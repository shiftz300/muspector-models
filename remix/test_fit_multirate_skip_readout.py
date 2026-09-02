import unittest

import numpy as np

from .fit_multirate_skip_readout import fit_positions


class SkipSamplingTests(unittest.TestCase):
    def test_exact_predecessors_present_and_no_out_of_range(self):
        for peak in (0, 1, 128, 999):
            target = np.zeros(1000)
            target[peak] = -.7
            before = target.copy()
            for phase in range(16):
                selected, positions, priority = fit_positions(target, phase)
                self.assertGreaterEqual(selected.min(), 1)
                self.assertLess(selected.max(), len(target))
                self.assertTrue(set(selected).issubset(positions))
                self.assertTrue(set(selected-1).issubset(positions))
                expected = set(range(max(1, peak-128), min(len(target), peak+129)))
                self.assertTrue(expected.issubset(selected))
                self.assertTrue(np.all(priority[np.abs(selected-peak) <= 128] == 32.))
                self.assertTrue(np.all(priority[np.abs(selected-peak) > 128] == 16.))
            self.assertTrue(np.array_equal(target, before))

    def test_rotation_covers_each_noninitial_scored_frame(self):
        target = np.ones(1000)
        represented = set()
        for phase in range(16):
            represented.update(fit_positions(target, phase)[0])
        self.assertEqual(represented, set(range(1, 1000)))


if __name__ == "__main__":
    unittest.main()
