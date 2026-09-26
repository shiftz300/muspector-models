import unittest

import torch

from .amp_forward_verifier import WetConditionedAmpForwardVerifier


class AmpForwardVerifierTests(unittest.TestCase):
    def test_starts_as_direct_identity_without_sample_rate_wet_path(self):
        model = WetConditionedAmpForwardVerifier(8, 4, 16).eval()
        clean = torch.randn(2, 4096) * 0.05
        wet = torch.randn(2, 4096) * 0.05
        reference = torch.randn(2, 8192) * 0.05
        with torch.inference_mode():
            replay = model(clean, wet, tone_reference=reference)
        torch.testing.assert_close(replay, clean, atol=1.0e-6, rtol=1.0e-6)
        manifest = model.manifest()
        self.assertFalse(manifest["sample_rate_wet_path"])
        self.assertFalse(manifest["profile_id_input"])
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbor_effect_input"])

    def test_rejects_mismatched_shapes_and_external_reference_batch(self):
        model = WetConditionedAmpForwardVerifier(8, 4, 16)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 4096), torch.zeros(1, 4095))
        with self.assertRaises(ValueError):
            model(
                torch.zeros(1, 4096), torch.zeros(1, 4096),
                tone_reference=torch.zeros(2, 8192),
            )


if __name__ == "__main__":
    unittest.main()
