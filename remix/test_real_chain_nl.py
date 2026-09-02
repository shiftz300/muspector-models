import unittest

import numpy as np

from .real_chain_nl import run


class RealChainNonlinearTests(unittest.TestCase):
    def test_nonlinear_drive_makes_order_noncommutative(self):
        frames = 64
        response = np.ones(frames // 2 + 1, dtype=np.complex64)
        drive = {"response": response, "linear_strength": np.asarray(1.0), "nonlinear_coefficients": np.asarray([1.0, 0.25, 0.0, 0.0, 0.0]), "nonlinear_strength": np.asarray(1.0)}
        reverb = {"response": np.linspace(0.7, 1.1, len(response)).astype(np.complex64), "strength": np.asarray(1.0)}
        source = np.random.default_rng(12).standard_normal(frames).astype(np.float32) * 0.1
        left = run(source, ("drive", "reverb"), drive, reverb)
        right = run(source, ("reverb", "drive"), drive, reverb)
        self.assertGreater(float(np.linalg.norm(left - right)), 1.0e-5)

    def test_unknown_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            run(np.zeros(32, dtype=np.float32), ("delay", "drive"), {}, {})


if __name__ == "__main__":
    unittest.main()
