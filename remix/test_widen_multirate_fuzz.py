import unittest

import torch

from .multirate_fuzz import MultirateFuzz
from .widen_multirate_fuzz import widen


class WidenTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(993)
        torch.set_num_threads(1)
        self.source = MultirateFuzz().eval()
        with torch.no_grad():
            self.source.audio.output.weight.normal_(0, .05)

    @torch.inference_mode()
    def test_initial_function_and_source_unchanged(self):
        weights = {key: value.clone() for key, value in self.source.state_dict().items()}
        target = widen(self.source)
        dry, controls = torch.randn(2, 6173) * .03, torch.rand(2, 6173, 2)
        before = self.source(dry, controls)[0]
        after = target(dry, controls)[0]
        self.assertLessEqual(float((before - after).abs().max()), 2e-6)
        self.assertEqual(target.audio.receptive_field, self.source.audio.receptive_field)
        self.assertEqual(target.controller.receptive_field, self.source.controller.receptive_field)
        self.assertTrue(all(torch.equal(value, weights[key]) for key, value in self.source.state_dict().items()))
        self.assertEqual(float(target(torch.zeros_like(dry), controls)[0].abs().max()), 0.)

    def test_outgoing_split_breaks_gradient_symmetry(self):
        target = widen(self.source)
        dry, controls = torch.randn(1, 511) * .03, torch.rand(1, 511, 2)
        target(dry, controls)[0].square().mean().backward()
        gradient = target.audio.input.weight.grad
        self.assertGreater(float((gradient[::2] - gradient[1::2]).abs().max()), 0.)


if __name__ == "__main__":
    unittest.main()
