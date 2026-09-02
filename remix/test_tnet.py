import unittest

import torch

from .tnet import TimeNet


class TimeNetTests(unittest.TestCase):
    def test_exact_bypass(self):
        net = TimeNet(channels=4, dilations=(1, 2))
        wet = torch.randn(2, 1024)
        controls = torch.rand(2, 3)
        self.assertTrue(torch.equal(net(wet, controls, strength=0.0), wet))

    def test_controls_change_output(self):
        torch.manual_seed(5)
        net = TimeNet(channels=4, dilations=(1, 2))
        torch.nn.init.normal_(net.input_condition.weight, std=0.1)
        torch.nn.init.normal_(net.head.weight, std=0.1)
        wet = torch.randn(1, 1024)
        low = net(wet, torch.zeros(1, 3))
        high = net(wet, torch.ones(1, 3))
        self.assertGreater(float((low - high).abs().max().detach()), 1.0e-6)

    def test_invalid_controls_are_rejected(self):
        net = TimeNet(channels=4, dilations=(1,))
        with self.assertRaises(ValueError):
            net(torch.zeros(1, 512), torch.ones(1, 3) * -0.1)


if __name__ == "__main__":
    unittest.main()
