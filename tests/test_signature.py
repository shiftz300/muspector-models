import numpy as np
import unittest

from remix.signature import FEATURES, PAIR, encode, select


class SignatureTests(unittest.TestCase):
    def test_selection_geometry_is_finite(self):
        result = select(np.linspace(-2.0, 2.0, PAIR, dtype=np.float32))
        self.assertEqual(result.shape, (len(FEATURES),))
        self.assertTrue(np.isfinite(result).all())

    def test_encoding_is_gain_invariant_and_read_only(self):
        time = np.arange(48_000, dtype=np.float32) / 48_000
        dry = (0.02 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
        wet = np.tanh(dry * 8).astype(np.float32)
        before = dry.copy(), wet.copy()
        first = encode(dry, wet, 48_000)
        second = encode(dry * 3, wet * 0.25, 48_000)
        # STFT phase in nearly empty bins has sub-milliradian numeric jitter.
        self.assertTrue(np.allclose(first, second, atol=1.1e-3, rtol=2e-4))
        self.assertTrue(np.array_equal(dry, before[0]) and np.array_equal(wet, before[1]))

    def test_rejects_non_48k_runtime_contract(self):
        with self.assertRaises(ValueError):
            encode(np.ones(32, np.float32), np.ones(32, np.float32), 44_100)


if __name__ == "__main__":
    unittest.main()
