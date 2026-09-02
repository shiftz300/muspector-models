import unittest

import numpy as np

from .features import context, extract


class FeatureTests(unittest.TestCase):
    def test_features_are_fixed_finite_and_deterministic(self):
        value = np.random.default_rng(4).standard_normal(4096).astype(np.float32)
        left, right = extract(value), extract(value.copy())
        self.assertEqual(left.shape, right.shape)
        self.assertGreater(len(left), 100)
        self.assertTrue(np.array_equal(left, right))
        self.assertTrue(np.isfinite(left).all())

    def test_invalid_audio_is_rejected(self):
        for value in (np.zeros((2, 512)), np.zeros(128), np.full(512, np.nan)):
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                extract(value)

    def test_context_features_extend_base_vector(self):
        value = np.random.default_rng(5).standard_normal(32768).astype(np.float32)
        base, long = extract(value), context(value)
        self.assertGreater(len(long), len(base))
        self.assertTrue(np.array_equal(long[: len(base)], base))
        self.assertTrue(np.array_equal(long, context(value.copy())))

    def test_short_context_is_rejected(self):
        with self.assertRaises(ValueError):
            context(np.zeros(32767, dtype=np.float32))


if __name__ == "__main__": unittest.main()
