import unittest
import numpy as np

from .diagnose_dfz_feasibility import epsilon_lower_bound, phase_one


class PhaseOneTests(unittest.TestCase):
    def test_same_feasible_region_under_extreme_positive_row_scales(self):
        a = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        b = np.array([2., -1., 4., -3.])
        scale = np.array([1e-8, 1e8, 1e-6, 1e6])
        result, audit = phase_one(a*scale[:, None], b*scale)
        self.assertTrue(result.success)
        self.assertLessEqual(float(np.max(a@result.x-b)), 1e-7)
        self.assertLessEqual(audit['slack'], 1e-7)

    def test_positive_slack_has_dual_certificate(self):
        result, audit = phase_one(np.array([[1.], [-1.]]), np.array([1., -2.]))
        self.assertFalse(result.success)
        self.assertEqual(result.status, 2)
        self.assertAlmostEqual(audit['slack'], .5)
        self.assertAlmostEqual(audit['dual_lower_bound'], .5)
        self.assertLess(audit['dual_balance_max'], 1e-7)

    def test_auxiliary_peak_tolerance_bound(self):
        # v<=.3+epsilon and v>=.4-epsilon need epsilon >= .05.
        a, eps = np.array([[1.], [-1.]]), .015
        result = epsilon_lower_bound(a, np.array([.3+eps, -.4+eps]), eps)
        self.assertTrue(result['verified'])
        self.assertAlmostEqual(result['minimum_epsilon_on_cuts'], .05)
        self.assertAlmostEqual(result['dual_lower_bound'], .05)

    def test_generic_constraints_do_not_get_peak_epsilon_interpretation(self):
        result, audit = phase_one(np.array([[1.], [-1.]]), np.array([1., -2.]), auxiliary_bound=False)
        self.assertEqual(result.status, 2)
        self.assertNotIn('auxiliary_absolute_epsilon_bound', audit)
        self.assertLessEqual(audit['dual_weight_sum'], 1+1e-7)


if __name__ == '__main__':
    unittest.main()
