import unittest

import torch

from .amp_model import AmpInverseExpert, ControlFIR, OUTPUT_PEAK_CONTRACT, control_basis
from .quality2 import REQUIRED_IMPROVEMENTS


class AmpModelTests(unittest.TestCase):
    def test_amp_requires_all_four_restoration_dimensions(self):
        self.assertEqual(
            REQUIRED_IMPROVEMENTS["amp"],
            {"spectrum", "high_band", "transient", "dynamics"},
        )

    def test_identity_initialization_and_control_basis(self):
        wet = torch.linspace(-0.25, 0.25, 2048).repeat(2, 1)
        controls = torch.tensor([[0.0, 0.5, 1.0, 0.5], [1.0, 0.5, 0.0, 0.5]])
        profile = ControlFIR()
        with torch.inference_mode():
            restored = profile(wet, controls)
        torch.testing.assert_close(restored, wet)
        self.assertEqual(control_basis(controls).shape, (2, 15))

    def test_expert_has_no_graph_context_and_is_bounded(self):
        model = AmpInverseExpert(hidden_size=8, depth=3).eval()
        wet = torch.randn(2, 4096) * 0.05
        controls = torch.tensor([[0.2, 0.5, 0.8, 0.5], [0.8, 0.5, 0.2, 0.5]])
        with torch.inference_mode():
            restored, uncertainty, state = model(wet, controls)
        self.assertEqual(restored.shape, wet.shape)
        self.assertEqual(uncertainty.shape, wet.shape)
        self.assertIsNone(state)
        self.assertLessEqual(float(restored.abs().max()), OUTPUT_PEAK_CONTRACT + 1.0e-6)
        self.assertTrue(torch.all(uncertainty > 0.0))
        torch.testing.assert_close(restored, wet)
        manifest = model.manifest()
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbor_effect_input"])
        self.assertIn("B/M/T and released Gain", manifest["current_device_scope"])
        self.assertEqual(manifest["gain_development_holdouts"], [1.0, 8.0])

    def test_invalid_controls_and_state_are_rejected(self):
        model = AmpInverseExpert(hidden_size=8, depth=2)
        wet = torch.zeros(1, 512)
        with self.assertRaises(ValueError):
            model(wet, torch.full((1, 4), 1.1))
        with self.assertRaises(ValueError):
            model(wet, torch.zeros(1, 4), torch.zeros(1))


if __name__ == "__main__":
    unittest.main()
