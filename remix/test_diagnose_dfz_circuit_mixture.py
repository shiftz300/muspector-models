import unittest
import numpy as np

from .diagnose_dfz_circuit_mixture import bootstrap_upper, fixed_circuit_mix, subset_summary


class CircuitMixtureTests(unittest.TestCase):
    def test_fixed_mix_endpoints_and_midpoint(self):
        left = np.array([-.5, .25], np.float32)
        right = np.array([.5, -.75], np.float32)
        np.testing.assert_array_equal(fixed_circuit_mix(left, right, 0.), left)
        np.testing.assert_array_equal(fixed_circuit_mix(left, right, 1.), right)
        np.testing.assert_array_equal(fixed_circuit_mix(left, right, .5), np.array([0., -.25], np.float32))

    def test_bootstrap_uses_paired_file_errors(self):
        base = [{'peak_error': value} for value in (.2, .3, .4, .5)]
        better = [{'peak_error': value-.1} for value in (.2, .3, .4, .5)]
        lo, hi = bootstrap_upper(better, base)
        self.assertAlmostEqual(lo, -.1)
        self.assertAlmostEqual(hi, -.1)

    def test_invalid_mix_is_rejected(self):
        with self.assertRaises(ValueError):
            fixed_circuit_mix(np.zeros(2), np.zeros(3), .5)
        with self.assertRaises(ValueError):
            fixed_circuit_mix(np.zeros(2), np.zeros(2), 1.1)

    def test_subset_summary_does_not_require_absent_blend_groups(self):
        rows = [{'peak_error': .1, 'error_energy': 1., 'wet_energy': 4.},
                {'peak_error': .2, 'error_energy': 2., 'wet_energy': 6.}]
        result = subset_summary(rows)
        self.assertEqual(result['examples'], 2)
        self.assertAlmostEqual(result['global_esr'], .3)


if __name__ == '__main__':
    unittest.main()
