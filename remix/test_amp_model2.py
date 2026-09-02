import unittest

import torch

from .amp_model import OUTPUT_PEAK_CONTRACT
from .amp_model2 import AmpDynamicsInverseExpert


class AmpDynamicsModelTests(unittest.TestCase):
    def test_identity_initialization_and_order_boundary(self):
        model = AmpDynamicsInverseExpert(hidden_size=8, depth=3).eval()
        wet = torch.randn(2, 4096) * 0.05
        controls = torch.tensor([[0.5, 0.5, 0.5, 0.2], [0.5, 0.5, 0.5, 0.8]])
        with torch.inference_mode():
            restored, uncertainty, state = model(wet, controls)
        torch.testing.assert_close(restored, wet)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        self.assertLessEqual(float(restored.abs().max()), OUTPUT_PEAK_CONTRACT + 1.0e-6)
        manifest = model.manifest()
        self.assertTrue(manifest["multiplicative_dynamics_path"])
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbor_effect_input"])
        self.assertFalse(manifest["recurrent_state_input"])

    def test_state_is_rejected(self):
        model = AmpDynamicsInverseExpert(hidden_size=8, depth=3)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 512), torch.zeros(1, 4), torch.zeros(1))


if __name__ == "__main__":
    unittest.main()
