import os
import unittest
from pathlib import Path

import torch

from remix.stable_rat import StableRatRenderer, load_stable_rat


ROOT = Path(__file__).resolve().parents[1]
CLIENT = Path(os.environ.get("MUSPECTOR_CLIENT", ROOT.parent / "muspector")).resolve()
BASELINE = CLIENT / "remix/runs/asrnn-rat-stable-pilot/rat-stable.pt"


class StableRatRefitTests(unittest.TestCase):
    def test_encode_exposes_the_exact_bias_free_readout_input(self):
        torch.manual_seed(7)
        model = StableRatRenderer(hidden_size=4, layers=2)
        dry = torch.randn(3, 257) * 0.03
        controls = torch.rand(3, 3)
        hidden, encoded_state = model.encode(dry, controls)
        rendered, forward_state = model(dry, controls)
        self.assertTrue(torch.equal(rendered, model.output_layer(hidden).squeeze(-1)))
        self.assertEqual(len(encoded_state), len(forward_state))
        for encoded, forwarded in zip(encoded_state, forward_state):
            self.assertTrue(torch.equal(encoded[0], forwarded[0]))
            self.assertTrue(torch.equal(encoded[1], forwarded[1]))

    @unittest.skipUnless(BASELINE.is_file(), "stable RAT baseline is unavailable")
    def test_admitted_core_stays_zero_after_readout_replacement(self):
        model = load_stable_rat(BASELINE)
        with torch.no_grad():
            model.output_layer.weight.copy_(torch.randn_like(model.output_layer.weight))
            output, _ = model(torch.zeros(2, 512), torch.rand(2, 3))
        self.assertEqual(float(output.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
