import unittest

import torch

from .refine_multirate_transients import STEPS, learning_rate, tail_loss


class SameTimeTailTests(unittest.TestCase):
    def test_exact_match_zero(self):
        target = torch.randn(2, 64)
        self.assertTrue(torch.equal(tail_loss(target, target, torch.ones(2)), torch.zeros(2)))

    def test_equal_amplitude_shift_is_not_match(self):
        target, predicted = torch.zeros(1, 64), torch.zeros(1, 64)
        target[0, 20], predicted[0, 40] = 1., 1.
        self.assertEqual(float(tail_loss(predicted, target, torch.ones(1))), .125)

    def test_gradient_corrects_both_signs_at_same_indices(self):
        prediction = torch.zeros(1, 64, requires_grad=True)
        target = torch.zeros(1, 64)
        target[0, 11], target[0, 39] = 1., -1.
        energy = torch.ones(1, requires_grad=True)
        tail_loss(prediction, target, energy).mean().backward()
        self.assertLess(float(prediction.grad[0, 11]), 0)
        self.assertGreater(float(prediction.grad[0, 39]), 0)
        self.assertEqual(int(prediction.grad.count_nonzero()), 2)
        self.assertIsNone(energy.grad)

    def test_fixed_schedule_and_validation(self):
        self.assertAlmostEqual(learning_rate(1), 3e-4)
        self.assertAlmostEqual(learning_rate(STEPS), 1e-5)
        with self.assertRaises(ValueError):
            learning_rate(STEPS + 1)
        with self.assertRaises(ValueError):
            tail_loss(torch.zeros(1, 8), torch.zeros(1, 8), torch.ones(1))


if __name__ == "__main__":
    unittest.main()
