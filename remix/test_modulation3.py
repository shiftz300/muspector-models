import random
import unittest

import numpy as np
import torch

from .modulation3 import normalized_controls, tremolo
from .modulation_model3 import TremoloInverseV3


class Modulation3Tests(unittest.TestCase):
    def test_tremolo_pair_contains_hidden_phase_but_controls_do_not(self):
        clean = np.random.default_rng(5).standard_normal(192000).astype(np.float32) * 0.05
        wet, values, inverse = tremolo(clean, random.Random(7))
        controls = normalized_controls(values)
        self.assertEqual(controls.shape, (5,))
        self.assertIn("hidden_phase_radians", values)
        self.assertGreater(float(np.max(inverse)), 0.0)
        np.testing.assert_allclose(wet * np.exp(inverse), clean, atol=2.0e-7)

    def test_expert_is_order_independent_and_preserves_silence(self):
        model = TremoloInverseV3(channels=8, depth=6).eval()
        wet = torch.zeros(2, 96000)
        controls = torch.rand(2, 5)
        with torch.inference_mode():
            restored, uncertainty, trajectory = model(wet, controls)
        self.assertEqual(float(restored.abs().max()), 0.0)
        self.assertEqual(trajectory.shape, wet.shape)
        self.assertTrue(torch.all(uncertainty > 0.0))
        manifest = model.manifest()
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbouring_effect_input"])
        self.assertFalse(manifest["hidden_forward_state_is_inference_input"])


if __name__ == "__main__":
    unittest.main()
