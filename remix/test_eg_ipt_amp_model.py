import unittest

import torch

from .eg_ipt_amp_model import (
    EgIptFixedProfileAmpInverse,
    EgIptFixedProfileAmpTransientInverse,
)


class EgIptAmpModelTests(unittest.TestCase):
    def test_fixed_profile_model_is_wet_only_and_order_independent(self):
        model = EgIptFixedProfileAmpInverse(4, 16).eval()
        wet = torch.randn(2, 8192) * 0.05
        with torch.inference_mode():
            restored, uncertainty, state = model(wet)
        self.assertEqual(restored.shape, wet.shape)
        self.assertTrue(torch.isfinite(restored).all())
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertTrue(manifest["device_profile_baked_into_weights"])
        self.assertFalse(manifest["content_dependent_profile_estimation"])
        self.assertFalse(manifest["profile_id_input"])
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbor_effect_input"])

    def test_external_state_is_rejected(self):
        with self.assertRaises(ValueError):
            EgIptFixedProfileAmpInverse(4, 16)(torch.zeros(1, 4096), torch.zeros(1))

    def test_transient_refiner_starts_as_frozen_base(self):
        model = EgIptFixedProfileAmpTransientInverse(8, 4).eval()
        wet = torch.randn(1, 8192) * 0.05
        with torch.inference_mode():
            expected, _, _ = model.base(wet)
            restored, _, _ = model(wet)
        torch.testing.assert_close(restored, expected)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.base.parameters()))
        manifest = model.manifest()
        self.assertEqual(manifest["schema"], 20)
        self.assertFalse(manifest["content_dependent_profile_estimation"])


if __name__ == "__main__":
    unittest.main()
