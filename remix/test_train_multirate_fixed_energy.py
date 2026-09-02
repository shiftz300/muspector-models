import unittest

import torch

from .train_dfz_signed_tcn import signed_frame_weights
from .train_multirate_fixed_energy import fixed_energy_loss, learning_rate


class FixedEnergyLossTests(unittest.TestCase):
    def test_matches_declared_formula_not_window_denominator(self):
        expected = torch.tensor([[0., .1, -.1, 0.], [0., .5, -.5, 0.]])
        prediction = expected + .02
        peak = expected.abs().amax(1)
        energy, pre_energy = torch.tensor([.25, .25]), torch.tensor([.4, .4])
        actual = fixed_energy_loss(prediction, expected, peak, energy, pre_energy)
        error = prediction - expected
        formula = (signed_frame_weights(expected, peak)*error.square()).mean(1)/energy
        formula += .1*(error[:, 1:]-.95*error[:, :-1]).square().mean(1)/pre_energy
        self.assertTrue(torch.allclose(actual, formula))
        self.assertAlmostEqual(float(actual[0]), float(actual[1]), places=6)

    def test_exact_and_silent_pairs_remain_finite(self):
        expected = torch.zeros(2, 64)
        prediction = expected.clone().requires_grad_()
        scale = torch.zeros(2)
        value = fixed_energy_loss(prediction, expected, scale, scale, scale)
        self.assertTrue(torch.equal(value, torch.zeros(2)))
        value.sum().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_no_gradient_or_mutation_in_fixed_scales(self):
        target = torch.tensor([[0., -.1, .2, .1]])
        predicted = (target+.02).requires_grad_()
        energy = torch.tensor([.3], requires_grad=True)
        before = energy.detach().clone()
        fixed_energy_loss(predicted, target, target.abs().amax(1), energy, energy).sum().backward()
        self.assertIsNone(energy.grad)
        self.assertTrue(torch.equal(energy.detach(), before))
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_geometry_and_fixed_schedule(self):
        with self.assertRaises(ValueError):
            fixed_energy_loss(torch.zeros(2, 8), torch.zeros(2, 8), torch.zeros(2), torch.zeros(1), torch.zeros(2))
        self.assertAlmostEqual(learning_rate(250), 3e-4)
        self.assertEqual(learning_rate(3000), 3e-4)
        self.assertAlmostEqual(learning_rate(6000), 1e-5)
        for step in (0, 6001):
            with self.assertRaises(ValueError):
                learning_rate(step)


if __name__ == "__main__":
    unittest.main()
