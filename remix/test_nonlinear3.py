import math
import unittest

import numpy as np
import torch
from scipy.signal import butter, sosfilt

from .inverse3 import NonlinearInverseV3, OUTPUT_PEAK_CONTRACT, _regularized_deequalize


class Nonlinear3Tests(unittest.TestCase):
    def test_regularized_deequalizer_reduces_known_lowpass_error(self):
        frames = 8192
        time = np.arange(frames, dtype=np.float64) / 48_000.0
        source = (
            0.20 * np.sin(2.0 * np.pi * 440.0 * time)
            + 0.12 * np.sin(2.0 * np.pi * 2200.0 * time)
            + 0.05 * np.sin(2.0 * np.pi * 5200.0 * time)
        ).astype(np.float32)
        cutoff = 2900.0
        filtered = sosfilt(
            butter(2, cutoff, btype="lowpass", fs=48_000, output="sos"), source
        ).astype(np.float32)
        controls = torch.zeros(1, 5)
        controls[0, 2] = math.log(cutoff / 1800.0) / math.log(11000.0 / 1800.0)
        with torch.inference_mode():
            restored = _regularized_deequalize(torch.from_numpy(filtered).unsqueeze(0), controls)[0].numpy()
        interior = slice(512, -512)
        baseline = np.mean(np.square(filtered[interior] - source[interior]))
        candidate = np.mean(np.square(restored[interior] - source[interior]))
        self.assertLess(candidate, baseline * 0.55)
        self.assertLess(float(np.max(np.abs(restored))), 0.75)

    def test_expert_is_order_independent_and_shape_preserving(self):
        model = NonlinearInverseV3(hidden_size=8, layers=1).eval()
        wet = torch.randn(2, 4096) * 0.05
        controls = torch.tensor([
            [0.3, 0.5, 0.2, 0.5, 0.0],
            [0.7, 0.4, 0.8, 0.4, 1.0],
        ])
        with torch.inference_mode():
            restored, uncertainty, state = model(wet, controls)
        self.assertEqual(restored.shape, wet.shape)
        self.assertEqual(uncertainty.shape, wet.shape)
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbouring_effect_input"])
        self.assertGreater(manifest["local_receptive_field_frames"], manifest["deequalizer_taps"])
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertLessEqual(float(restored.abs().max()), OUTPUT_PEAK_CONTRACT + 1.0e-6)


if __name__ == "__main__":
    unittest.main()
