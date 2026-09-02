import unittest

import torch

from .cnet import ConditionalNet


class ConditionalNetTests(unittest.TestCase):
    def test_exact_bypass(self):
        model = ConditionalNet(channels=4, dilations=((1, 1),))
        wet = torch.randn(2, 2048)
        controls = torch.tensor([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]])
        self.assertTrue(torch.equal(model(wet, controls, strength=0.0), wet))

    def test_controls_change_the_learned_mapping(self):
        torch.manual_seed(4)
        model = ConditionalNet(channels=4, dilations=((1, 1),))
        torch.nn.init.normal_(model.head.weight, std=0.1)
        wet = torch.randn(1, 2048)
        low = model(wet, torch.zeros(1, 3))
        high = model(wet, torch.ones(1, 3))
        self.assertGreater(float((low - high).abs().max().detach()), 1.0e-6)

    def test_invalid_controls_are_rejected(self):
        model = ConditionalNet(channels=4, dilations=((1, 1),))
        wet = torch.zeros(1, 2048)
        for controls in (torch.zeros(1, 2), torch.tensor([[0.0, 2.0, 0.0]])):
            with self.subTest(shape=tuple(controls.shape)), self.assertRaises(ValueError):
                model(wet, controls)


if __name__ == "__main__":
    unittest.main()
