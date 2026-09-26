import unittest

import torch

from .ambience_model2 import AmbienceExpert, _ambience_spectral_loss, ambience_loss


class AmbienceModel2Tests(unittest.TestCase):
    def test_geometry_context_and_order_independence_contract(self):
        model = AmbienceExpert(channels=4, depth=6, n_fft=512, hop=128).eval()
        wet = torch.randn(1, 16384) * 0.02
        controls = torch.rand(1, 3)
        with torch.inference_mode():
            restored, uncertainty = model(wet, controls, wet.clone())
        self.assertEqual(restored.shape, wet.shape)
        self.assertEqual(uncertainty.shape, wet.shape)
        self.assertFalse(model.manifest()["chain_order_input"])
        self.assertFalse(model.manifest()["neighbor_effect_input"])
        self.assertTrue(torch.all(uncertainty > 0))

    def test_loss_is_finite_and_backward_safe(self):
        model = AmbienceExpert(channels=4, depth=6, n_fft=512, hop=128)
        wet = torch.randn(1, 16384) * 0.02
        clean = wet * 0.8
        restored, uncertainty = model(wet, torch.rand(1, 3), wet.clone())
        loss, parts = ambience_loss(restored, uncertainty, wet, clean, 0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("tail", parts)

    def test_quiet_spectral_loss_has_an_audible_floor(self):
        target = torch.zeros(2, 16384)
        prediction = torch.full_like(target, 1.0e-5)
        loss = _ambience_spectral_loss(prediction, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss), 3.0)


if __name__ == "__main__":
    unittest.main()
