import unittest

import torch

from .amp_model3 import AmpFrameDynamicsInverseExpert


class AmpFrameDynamicsModelTests(unittest.TestCase):
    def test_identity_initialization_and_order_boundary(self):
        model = AmpFrameDynamicsInverseExpert(hidden_size=8, depth=1).eval()
        wet = torch.randn(2, 4096) * 0.05
        controls = torch.tensor([[0.5, 0.5, 0.5, 0.2], [0.5, 0.5, 0.5, 0.8]])
        with torch.inference_mode():
            restored, uncertainty, state = model(wet, controls)
        torch.testing.assert_close(restored, wet)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertTrue(manifest["whole_chunk_context"])
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbor_effect_input"])
        self.assertFalse(manifest["recurrent_state_input"])

    def test_external_state_is_rejected(self):
        model = AmpFrameDynamicsInverseExpert(hidden_size=8, depth=1)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 512), torch.zeros(1, 4), torch.zeros(1))


if __name__ == "__main__":
    unittest.main()
