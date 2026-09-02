import unittest

import torch

from .foundation_model import Expert, MECHANISMS, restoration_loss


class FoundationModelTests(unittest.TestCase):
    def test_experts_have_equal_budget_and_preserve_geometry(self):
        value = torch.randn(2, 4096) * 0.05
        models = [Expert(name, channels=8, blocks=3) for name in MECHANISMS]
        self.assertEqual(len({model.manifest()["parameters"] for model in models}), 1)
        for model in models:
            restored = model(value)
            self.assertEqual(restored.shape, value.shape)
            self.assertTrue(torch.isfinite(restored).all())

    def test_manifest_discloses_mechanism_context_budget(self):
        self.assertEqual(Expert("nonlinear").manifest()["input_context_frames"], 65)
        self.assertEqual(Expert("dynamics").manifest()["input_context_frames"], 513)
        self.assertEqual(Expert("temporal").manifest()["input_context_frames"], 4095)

    def test_identity_target_has_finite_differentiable_loss(self):
        wet = torch.randn(2, 4096) * 0.05
        model = Expert("dynamics", channels=8, blocks=2)
        restored = model(wet)
        loss, parts = restoration_loss(restored, wet)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"waveform", "transient", "envelope", "long_shape"})
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_unknown_mechanism_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            Expert("rat")


if __name__ == "__main__":
    unittest.main()
