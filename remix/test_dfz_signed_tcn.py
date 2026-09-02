"""Independent numerical contracts for the fit-only same-index signed loss."""
import unittest

import numpy as np
import torch

from .train_dfz_signed_tcn import (active_window_ranges, signed_frame_weights,
                                   signed_loss_per_example, uniform_active_start)


class SignedLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_exact_match_has_zero_loss_and_gradient(self):
        wet = torch.tensor([[0., .2, -.4, .1, 0.], [.1, -.1, .2, -.2, .01]])
        prediction = wet.clone().requires_grad_()
        loss = signed_loss_per_example(prediction, wet, wet.abs().amax(1)).mean()
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(int(torch.count_nonzero(prediction.grad)), 0)

    def test_same_absolute_peak_does_not_hide_sign_or_time_errors(self):
        wet = torch.tensor([[0., .4, -.2, .1, 0., 0.]])
        peak = wet.abs().amax(1)
        for incorrect in (-wet, wet.roll(1, 1), wet * .5, wet * 1.2):
            prediction = incorrect.clone().requires_grad_()
            loss = signed_loss_per_example(prediction, wet, peak).mean()
            loss.backward()
            self.assertGreater(float(loss.detach()), 0.)
            self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
            self.assertGreater(int(torch.count_nonzero(prediction.grad)), 0)

    def test_weights_are_fit_target_only_and_between_one_and_five(self):
        wet = torch.tensor([[0., .1, -.2, .4], [0., 1e-8, -2e-8, 3e-8]])
        peak = wet.abs().amax(1)
        weights = signed_frame_weights(wet, peak)
        self.assertGreaterEqual(float(weights.min()), 1.)
        self.assertLessEqual(float(weights.max()), 5.)
        self.assertEqual(float(weights[0, 0]), 1.)
        self.assertEqual(float(weights[0, -1]), 5.)
        torch.testing.assert_close(weights[0, 1], torch.tensor(1 + 4 * (.1 / .4) ** 4))
        differentiable_target = wet.clone().requires_grad_()
        self.assertFalse(signed_frame_weights(differentiable_target, peak).requires_grad)

    def test_exact_energy_denominators_and_internal_preemphasis_pairs(self):
        wet = torch.tensor([[0., 1e-4, -2e-4, 1e-4], [.1, -.4, .2, -.1]], dtype=torch.float64)
        predicted = wet * .8 + torch.tensor([[1e-4], [.03]], dtype=torch.float64)
        peaks = wet.abs().amax(1)
        weights = 1 + 4 * (wet.abs() / peaks.clamp_min(1e-5)[:, None]) ** 4
        direct = (weights * (predicted - wet).square()).mean(1) / wet.square().mean(1).clamp_min(1e-5)
        dp = predicted[:, 1:] - .95 * predicted[:, :-1]
        dt = wet[:, 1:] - .95 * wet[:, :-1]
        direct += .1 * (dp - dt).square().mean(1) / dt.square().mean(1).clamp_min(1e-5)
        torch.testing.assert_close(signed_loss_per_example(predicted, wet, peaks), direct, atol=0, rtol=0)
        zeros = torch.zeros(2, 4)
        torch.testing.assert_close(signed_loss_per_example(zeros, zeros, torch.zeros(2)), torch.zeros(2), atol=0, rtol=0)

    def test_active_uniform_uses_exact_disjoint_eligible_starts(self):
        dry = torch.zeros(16_000)
        dry[2_100] = .2
        dry[12_000] = -.2
        ranges = active_window_ranges(dry)
        self.assertEqual(ranges, [(1_844, 2_100), (7_905, 11_904)])
        eligible = {start for start in range(1_844, 11_905) if bool(dry[start:start + 4_096].abs().max() > .002)}
        represented = {start for first, last in ranges for start in range(first, last + 1)}
        self.assertEqual(represented, eligible)
        rng = np.random.default_rng(987)
        self.assertTrue(all(uniform_active_start(rng, ranges) in eligible for _ in range(1_000)))
        self.assertEqual(active_window_ranges(torch.zeros(10_000)), [(1_024, 5_904)])


if __name__ == "__main__":
    unittest.main()
