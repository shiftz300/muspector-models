import unittest

import torch

from .train_multirate_l1 import absolute_loss, learning_rate


class AbsoluteWaveformTests(unittest.TestCase):
    def test_exact_audio_has_zero_loss(self):
        expected = torch.tensor([[0., -.2, .5, -.3], [.1, -.1, .05, .2]])
        value = absolute_loss(expected, expected, expected.abs().amax(1))
        self.assertTrue(torch.equal(value, torch.zeros(2)))

    def test_same_time_signed_errors_not_independent_maxima(self):
        expected = torch.tensor([[.5, -.5, 0., 0.]])
        prediction = torch.tensor([[0., 0., .5, -.5]], requires_grad=True)
        value = absolute_loss(prediction, expected, expected.abs().amax(1)).sum()
        self.assertGreater(float(value.detach()), 0.)
        value.backward()
        self.assertLess(float(prediction.grad[0, 0]), 0.)
        self.assertGreater(float(prediction.grad[0, 1]), 0.)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_zero_and_quiet_targets_finite_without_mutation(self):
        expected = torch.zeros(2, 128)
        prediction = torch.full_like(expected, 1e-7, requires_grad=True)
        before = expected.clone()
        value = absolute_loss(prediction, expected, torch.zeros(2)).sum()
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue(torch.equal(expected, before))

    def test_fixed_schedule(self):
        self.assertAlmostEqual(learning_rate(500), 1e-4)
        self.assertAlmostEqual(learning_rate(8000), 1e-5)
        self.assertLess(learning_rate(4000), learning_rate(501))
        for step in (0, 8001):
            with self.assertRaises(ValueError):
                learning_rate(step)


if __name__ == "__main__":
    unittest.main()
