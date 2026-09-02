import unittest

import torch

from .dfz_dynamic_readout import CausalTapBank, DynamicCornerReadout, FEATURES, LAGS


class DynamicReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(986)

    def test_taps_match_exact_full_prefix(self):
        dry = torch.randn(144_000) * .03
        original = torch.tanh(dry * 7) * .8
        bank = CausalTapBank()
        expected = bank.cpu_features(dry, original)
        pieces, state = [], None
        with torch.inference_mode():
            for start in range(0, len(dry), 4096):
                value, state = bank(dry[None, start:start + 4096], original[None, start:start + 4096], state)
                pieces.append(value[0])
        self.assertTrue(torch.equal(torch.cat(pieces), expected))
        for index, lag in enumerate(LAGS):
            self.assertTrue(torch.equal(expected[lag:, index * 2], dry[:len(dry) - lag] * 21.4))
            self.assertTrue(torch.equal(expected[lag:, index * 2 + 1], original[:len(dry) - lag]))
            self.assertEqual(float(expected[:lag, index * 2:index * 2 + 2].abs().sum()), 0.)

    def test_nonzero_readout_zero_dynamic_and_fast_grid(self):
        readout = DynamicCornerReadout()
        with torch.no_grad():
            readout.output.normal_(0, .02)
            readout.first_bias.normal_(0, .1)
            readout.second_bias.normal_(0, .1)
        features = torch.randn(9, 57, FEATURES) * .3
        controls = torch.tensor([(a, b) for a in (0., .5, 1.) for b in (0., .5, 1.)])
        with torch.inference_mode():
            self.assertLess(float((readout(features, controls) - readout.at_training_knots(features, controls)).abs().max()), 2e-7)
            dynamic = torch.rand(9, 57, 2)
            self.assertEqual(float(readout(torch.zeros_like(features), dynamic).abs().max()), 0.)

    def test_bank_stream_future_and_zero(self):
        bank = CausalTapBank()
        dry = torch.randn(2, 6147) * .03
        original = torch.tanh(dry * 7)
        with torch.inference_mode():
            whole, _ = bank(dry, original)
            a, state = bank(dry[:, :617], original[:, :617])
            b, _ = bank(dry[:, 617:], original[:, 617:], state)
            self.assertTrue(torch.equal(whole, torch.cat((a, b), 1)))
            changed = original.clone()
            changed[:, 617:] += .5
            future, _ = bank(dry, changed)
            self.assertTrue(torch.equal(whole[:, :617], future[:, :617]))
            zero, _ = bank(torch.zeros_like(dry), torch.zeros_like(original))
            self.assertEqual(float(zero.abs().max()), 0.)

    def test_readout_rejects_bad_scale(self):
        with self.assertRaises(ValueError):
            DynamicCornerReadout(torch.zeros(FEATURES))


if __name__ == "__main__":
    unittest.main()
