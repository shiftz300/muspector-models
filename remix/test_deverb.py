import unittest

import numpy as np

from .deverb import decode, restore


class _Profile:
    def impulse(self, decay_s, damping, frames):
        value = np.zeros(frames, dtype=np.float32)
        value[0], value[1], value[3] = 1.0, 0.15 * (1.0 - damping), 0.05
        return value


class DeverbTests(unittest.TestCase):
    def test_decode_bounds(self):
        effect = decode(np.asarray([0.0, 0.5, 1.0], dtype=np.float32))
        self.assertAlmostEqual(effect.decay_s, 0.2)
        self.assertAlmostEqual(effect.mix, 0.7)
        with self.assertRaises(ValueError):
            decode(np.asarray([0.0, 2.0, 0.5], dtype=np.float32))

    def test_restore_is_finite_and_preserves_frames(self):
        wet = np.random.default_rng(7).standard_normal(4096).astype(np.float32) * 0.01
        clean = restore(_Profile(), wet, np.asarray([0.4, 0.5, 0.6]), 0.05)
        self.assertEqual(clean.shape, wet.shape)
        self.assertTrue(np.isfinite(clean).all())

    def test_invalid_regularization_is_rejected(self):
        with self.assertRaises(ValueError):
            restore(_Profile(), np.zeros(32, dtype=np.float32), np.zeros(3), 0.0)


if __name__ == "__main__":
    unittest.main()
