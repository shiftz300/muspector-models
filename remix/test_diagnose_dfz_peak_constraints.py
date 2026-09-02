import time
import unittest

import numpy as np

from .diagnose_dfz_peak_constraints import EPSILON, constraints, solve_constrained


class PeakConstraintTests(unittest.TestCase):
    def test_positive_negative_anchors_and_complete_frame_verification(self):
        x = np.linspace(-1., 1., 2000, dtype=np.float32)[:, None]
        p = x[:, 0]*.2
        y = x[:, 0]*.25
        rows = [{'x': x, 'p': p, 'y': y}, {'x': -x, 'p': -p, 'y': -y}]
        xx = 2*x.astype(float).T@x.astype(float)
        xy = 2*x.astype(float).T@(y-p)
        weight, _, audit = solve_constrained(rows, xx, xy, time.perf_counter()+10)
        self.assertIsNotNone(weight)
        self.assertEqual(audit['status'], 'complete-training-waveforms-verified')
        for row in rows:
            prediction = row['p']+row['x']@weight
            self.assertLessEqual(abs(abs(prediction).max()-abs(row['y']).max()), EPSILON+5e-7)

    def test_incompatible_fixed_features_are_not_silently_relaxed(self):
        x = np.array([[1.], [1.]], np.float32)
        rows = [{'x': x, 'p': np.zeros(2, np.float32), 'y': np.array([.3, .2], np.float32)},
                {'x': x, 'p': np.zeros(2, np.float32), 'y': np.array([.8, .6], np.float32)}]
        weight, _, audit = solve_constrained(rows, np.eye(1), np.zeros(1), time.perf_counter()+10)
        self.assertIsNone(weight)
        self.assertEqual(audit['status'], 'cut-constraints-infeasible')

    def test_budget_prevents_optimization(self):
        weight, _, audit = solve_constrained([], np.eye(1), np.zeros(1), time.perf_counter()-1)
        self.assertIsNone(weight)
        self.assertEqual(audit['status'], 'budget-exhausted')

    def test_full_scan_discovers_unsampled_new_peak(self):
        x = np.array([[0.], [1.], [2.]], np.float32)
        p = np.zeros(3, np.float32)
        y = np.array([0., .25, .249], np.float32)
        # Initial anchors contain0 and1, not2. Satisfying the target peak at1
        # necessarily creates an oversized peak at2, caught by the full scan.
        weight, _, audit = solve_constrained([{'x': x, 'p': p, 'y': y}],
                                             x.astype(float).T@x, x.astype(float).T@y,
                                             time.perf_counter()+10)
        self.assertIsNone(weight)
        self.assertEqual(audit['status'], 'cut-constraints-infeasible')
        self.assertGreater(audit['history'][0]['cuts_added'], 0)
        self.assertGreater(audit['round'], 0)

    def test_explicit_tolerance_used_for_anchors_and_full_scan(self):
        x = np.ones((2, 1), np.float32)
        rows = [{'x': x, 'p': np.zeros(2, np.float32), 'y': np.array([.2, .2], np.float32)},
                {'x': x, 'p': np.zeros(2, np.float32), 'y': np.array([.239, .239], np.float32)}]
        xx, xy = np.eye(1), np.array([.2195])
        strict, _, _ = solve_constrained(rows, xx, xy, time.perf_counter()+10, epsilon=.015)
        original, _, audit = solve_constrained(rows, xx, xy, time.perf_counter()+10, epsilon=.019999)
        self.assertIsNone(strict)
        self.assertIsNotNone(original)
        self.assertEqual(audit['status'], 'complete-training-waveforms-verified')
        with self.assertRaises(ValueError):
            solve_constrained(rows, xx, xy, time.perf_counter()+10, epsilon=.020001)


if __name__ == '__main__':
    unittest.main()
