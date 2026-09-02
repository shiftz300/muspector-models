import unittest

import torch

from .train_rat_restoration import RATNet, RATNet2, attack_loss, loss, relative_loss


class RATRestorationTests(unittest.TestCase):
    def test_zero_initialized_model_is_exact_wet_bypass(self):
        model = RATNet(8, (1, 2))
        wet = torch.randn(2, 4096)
        prior = torch.randn_like(wet)
        controls = torch.rand(2, 3)
        self.assertTrue(torch.equal(model(wet, prior, controls), wet))

    def test_loss_rewards_the_clean_target(self):
        clean = torch.randn(2, 4096) * 0.1
        wet = torch.tanh(clean * 5.0) * 0.4
        clean_loss, _ = loss(clean, clean)
        wet_loss, _ = loss(wet, clean)
        self.assertLess(float(clean_loss), float(wet_loss))

    def test_v2_zero_initialized_model_is_exact_wet_bypass(self):
        model = RATNet2(8, 4, (1, 2))
        wet = torch.randn(2, 4096)
        controls = torch.rand(2, 3)
        self.assertTrue(torch.equal(model(wet, torch.randn_like(wet), controls), wet))

    def test_relative_loss_is_one_scale_at_wet_and_zero_at_clean(self):
        clean = torch.randn(2, 4096) * 0.1
        wet = torch.tanh(clean * 5.0) * 0.4
        wet_loss, parts = relative_loss(wet, clean, wet)
        clean_loss, _ = relative_loss(clean, clean, wet)
        self.assertAlmostEqual(float(parts["derivative"]), 1.0, places=5)
        self.assertGreater(float(wet_loss), 5.0)
        self.assertLess(float(clean_loss), 1.0e-5)

    def test_attack_loss_directly_rewards_clean_attacks(self):
        clean = torch.zeros(2, 4096)
        clean[:, 1000:1240] = torch.hann_window(240)
        wet = torch.nn.functional.avg_pool1d(clean.unsqueeze(1), 31, 1, 15).squeeze(1)
        clean_loss, clean_parts = attack_loss(clean, clean, wet)
        wet_loss, wet_parts = attack_loss(wet, clean, wet)
        self.assertLess(float(clean_parts["attack"]), float(wet_parts["attack"]))
        self.assertLess(float(clean_loss), float(wet_loss))


if __name__ == "__main__":
    unittest.main()
