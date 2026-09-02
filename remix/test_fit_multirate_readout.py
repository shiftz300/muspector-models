import unittest

import numpy as np

from .fit_multirate_readout import anchored_solution, summarize


class ConditionalReadoutFitTests(unittest.TestCase):
    def test_zero_delta_keeps_source_exact(self):
        source = np.array([.3, -.7])
        self.assertTrue(np.array_equal(anchored_solution(np.eye(2), np.zeros(2), source, .1), source))

    def test_anchored_normal_equations_and_inputs(self):
        rng = np.random.default_rng(1005)
        features, target, source = rng.normal(size=(100, 4)), rng.normal(size=100), rng.normal(size=4)
        before = tuple(value.copy() for value in (features, target, source))
        gram, residual = features.T @ features, target - features @ source
        result = anchored_solution(gram, features.T @ residual, source, .01)
        self.assertLess(np.square(features @ result - target).mean(), np.square(residual).mean())
        penalty = np.diag(np.diag(gram)) * .01
        self.assertTrue(np.allclose((gram + penalty) @ (result - source), features.T @ residual))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(before, (features, target, source))))

    def test_provisional_peak_thresholds_remain_strict(self):
        targets = [np.array([-.5, .2, -.1], dtype=np.float32)] * 3
        controls = [0, 50, 100]
        self.assertTrue(summarize(targets, targets, controls)["passes_selection_gate"])
        changed = [target * 1.05 for target in targets]
        metrics = summarize(changed, targets, controls)
        self.assertFalse(metrics["passes_selection_gate"])
        self.assertGreater(metrics["absolute_peak_error_p95"], .02)


if __name__ == "__main__":
    unittest.main()
