import unittest

import numpy as np

from .real_runtime import Runtime


class _Identity(Runtime):
    def __init__(self):
        self.frame, self.hop = 32, 16
        index = np.arange(self.frame, dtype=np.float64)
        self.window = np.square(np.sin(np.pi * (index + 0.5) / self.frame)).astype(np.float32)
        self.drive = self.reverb = {}

    def one(self, audio, kind):
        return super().one(audio, kind)


class RealRuntimeTests(unittest.TestCase):
    def test_bypass_is_bit_exact_for_arbitrary_length(self):
        runtime = _Identity()
        value = np.random.default_rng(13).standard_normal(77).astype(np.float32)
        self.assertTrue(np.array_equal(runtime.run(value, ()), value))

    def test_repeated_or_unknown_order_is_rejected(self):
        runtime = _Identity()
        value = np.zeros(64, dtype=np.float32)
        for order in (("drive", "drive"), ("delay",)):
            with self.subTest(order=order), self.assertRaises(ValueError):
                runtime.run(value, order)


if __name__ == "__main__":
    unittest.main()
