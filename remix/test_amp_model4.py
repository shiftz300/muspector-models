import unittest

import torch

from .amp_model import OUTPUT_PEAK_CONTRACT
from .amp_model4 import AmpStructuredInverseExpert, MonotonicDynamicInverse


class AmpStructuredModelTests(unittest.TestCase):
    def test_monotonic_stage_starts_as_identity(self):
        stage = MonotonicDynamicInverse(8).eval()
        value = torch.linspace(-0.8, 0.8, 4096).repeat(2, 1)
        controls = torch.tensor([[0.2, 0.5, 0.8, 0.3], [0.8, 0.5, 0.2, 0.7]])
        with torch.inference_mode():
            restored = stage(value, controls)
        torch.testing.assert_close(restored, value, rtol=1.0e-5, atol=1.0e-6)

    def test_structured_expert_is_order_independent_and_bounded(self):
        model = AmpStructuredInverseExpert().eval()
        wet = torch.randn(2, 4096) * 0.05
        controls = torch.tensor([[0.2, 0.5, 0.8, 0.3], [0.8, 0.5, 0.2, 0.7]])
        with torch.inference_mode():
            restored, uncertainty, state = model(wet, controls)
            staged, preamp = model.forward_stages(wet, controls)
        torch.testing.assert_close(restored, wet, rtol=1.0e-5, atol=1.0e-6)
        torch.testing.assert_close(staged, wet, rtol=1.0e-5, atol=1.0e-6)
        torch.testing.assert_close(preamp, wet, rtol=1.0e-5, atol=1.0e-6)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        self.assertLessEqual(float(restored.abs().max()), OUTPUT_PEAK_CONTRACT + 1.0e-6)
        manifest = model.manifest()
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbor_effect_input"])
        self.assertFalse(manifest["recurrent_state_input"])
        self.assertFalse(manifest["opaque_temporal_network"])
        self.assertEqual(len(manifest["internal_reverse_stages"]), 4)

    def test_invalid_state_is_rejected(self):
        model = AmpStructuredInverseExpert()
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 4096), torch.zeros(1, 4), torch.zeros(1))


if __name__ == "__main__":
    unittest.main()
