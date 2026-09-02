import unittest

import torch

from .dfz_temporal_basis import TemporalBasisConv, attach_temporal_basis, materialized_state_dict
from .multirate_fuzz import MultirateFuzz


class TemporalBasisTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(996)

    def test_initial_weight_exact_and_source_intact(self):
        source = torch.nn.Conv1d(4, 8, 3, dilation=2, bias=False)
        saved = source.weight.detach().clone()
        layer = TemporalBasisConv(source)
        self.assertTrue(torch.equal(layer.effective_weight(), saved))
        value = torch.randn(2, 4, 31)
        self.assertTrue(torch.equal(source(value), layer(value)))
        self.assertTrue(torch.equal(source.weight, saved))

    def test_full_rank_and_gradients(self):
        source = torch.nn.Conv1d(2, 4, 3, bias=False)
        layer = TemporalBasisConv(source)
        self.assertEqual(int(torch.linalg.matrix_rank(layer.basis)), 3)
        layer(torch.randn(1, 2, 31)).square().sum().backward()
        self.assertIsNotNone(layer.coefficients.grad)
        self.assertTrue(torch.isfinite(layer.coefficients.grad).all())
        self.assertIsNone(layer.base_weight.grad)

    @torch.inference_mode()
    def test_complete_model_materialization(self):
        model = MultirateFuzz().eval()
        model.audio.output.weight.normal_(0, .05)
        dry, controls = torch.randn(1, 6173) * .03, torch.rand(1, 6173, 2)
        original = model(dry, controls)[0]
        attach_temporal_basis(model)
        self.assertTrue(torch.equal(original, model(dry, controls)[0]))
        for layer in model.audio.convolutions[:4]:
            layer.coefficients.normal_(0, 1e-4)
        expected = model(dry, controls)[0]
        materialized = MultirateFuzz().eval()
        state = materialized_state_dict(model)
        self.assertFalse(any("base_weight" in name or "coefficients" in name or "basis" in name for name in state))
        materialized.load_state_dict(state, strict=True)
        self.assertTrue(torch.equal(expected, materialized(dry, controls)[0]))
        self.assertEqual(float(materialized(torch.zeros_like(dry), controls)[0].abs().max()), 0.)

    def test_refuse_double_attachment(self):
        model = attach_temporal_basis(MultirateFuzz())
        with self.assertRaises(ValueError):
            attach_temporal_basis(model)

    def test_fixed_long_phase_schedule(self):
        from .train_multirate_basis import STEPS, learning_rate
        self.assertAlmostEqual(learning_rate(500), 5e-4)
        self.assertAlmostEqual(learning_rate(10000), 5e-4)
        self.assertAlmostEqual(learning_rate(STEPS), 1e-5)
        with self.assertRaises(ValueError):
            learning_rate(STEPS + 1)


if __name__ == "__main__":
    unittest.main()
