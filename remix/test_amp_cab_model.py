import unittest

import torch

from .amp_cab_model import AmpCabInverseExpert


class AmpCabModelTests(unittest.TestCase):
    def test_identity_and_profile_boundary(self):
        model = AmpCabInverseExpert(8, 1).eval()
        wet = torch.randn(2, 4096) * 0.05
        profile = torch.eye(2)
        with torch.inference_mode():
            restored, uncertainty, state = model(wet, profile)
        torch.testing.assert_close(restored, wet)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbor_effect_input"])
        self.assertFalse(manifest["p3_input_or_profile"])

    def test_external_state_rejected(self):
        with self.assertRaises(ValueError):
            AmpCabInverseExpert(8, 1)(torch.zeros(1, 512), torch.tensor([[1.0, 0.0]]), torch.zeros(1))


if __name__ == "__main__":
    unittest.main()
