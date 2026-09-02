import unittest

import numpy as np

from .real_chain import run


class RealChainTests(unittest.TestCase):
    def test_order_is_reversed_without_merging_models(self):
        frames = 32
        identity = np.ones(frames // 2 + 1, dtype=np.complex64)
        bank = {"drive": (identity, 1.0, frames), "reverb": (identity, 1.0, frames)}
        source = np.random.default_rng(10).standard_normal(frames).astype(np.float32)
        restored = run(source, ("drive", "reverb"), bank)
        self.assertTrue(np.allclose(restored, source, atol=1.0e-6))

    def test_unknown_or_repeated_order_is_rejected(self):
        bank = {}
        for order in (("delay", "drive"), ("drive", "drive")):
            with self.subTest(order=order), self.assertRaises(ValueError):
                run(np.zeros(32, dtype=np.float32), order, bank)


if __name__ == "__main__":
    unittest.main()
