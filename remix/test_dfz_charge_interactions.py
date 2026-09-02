"""Small synthetic contracts; no official corpus, hardware, or GPU accesses."""
import unittest

import torch

from .fit_dfz_charge_interactions import (
    ChargeInteractionResidual, NEIGHBORHOOD_WEIGHTS, RIDGES, WIDTH, interaction_features,
)


class StatelessBase(torch.nn.Module):
    control_count = 2
    export_state_widths = [1]

    def forward(self, dry, controls, state=None):
        hidden = dry.new_zeros((1, dry.shape[0], 1))
        return dry * 0.6, ((hidden, hidden.clone()),)


class ChargeInteractionTests(unittest.TestCase):
    def test_feature_order_and_exact_zero_origin(self):
        charge = torch.arange(6, dtype=torch.float32)[None, None]
        original = torch.tensor([[6.0]])
        result = interaction_features(charge, original)
        expected = list(range(7)) + [i * j for i in range(7) for j in range(i, 7)]
        self.assertEqual(result.shape, (1, 1, 35))
        self.assertEqual(result.flatten().tolist(), expected)
        self.assertEqual(int(torch.count_nonzero(interaction_features(torch.zeros_like(charge), torch.zeros_like(original)))), 0)

    def test_fixed_menu_has_exactly_nine_global_regularized_candidates(self):
        self.assertEqual(NEIGHBORHOOD_WEIGHTS, (0.0, 1.0, 4.0))
        self.assertEqual(RIDGES, (1e-4, 0.01, 1.0))
        self.assertEqual(WIDTH, 35)

    def test_wrapper_is_causal_zero_origin_and_preserves_inputs(self):
        torch.manual_seed(761)
        model = ChargeInteractionResidual(StatelessBase(), torch.randn(9, WIDTH) * 0.002).eval()
        dry = torch.randn(2, 913) * 0.04
        controls = torch.rand(2, 913, 2)
        before = dry.clone(), controls.clone()
        with torch.inference_mode():
            whole, _ = model(dry, controls)
            state = None
            chunks = []
            for start, stop in ((0, 17), (17, 199), (199, 513), (513, 913)):
                value, state = model(dry[:, start:stop], controls[:, start:stop], state)
                chunks.append(value)
            zero, _ = model(torch.zeros_like(dry), controls)
        torch.testing.assert_close(whole, torch.cat(chunks, 1), rtol=0, atol=2e-6)
        self.assertEqual(int(torch.count_nonzero(zero)), 0)
        torch.testing.assert_close(dry, before[0], rtol=0, atol=0)
        torch.testing.assert_close(controls, before[1], rtol=0, atol=0)
        self.assertEqual(len(state), 2)


if __name__ == "__main__":
    unittest.main()
