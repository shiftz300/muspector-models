import unittest

import numpy as np

from .real_drive_nl import design, restore


class RealDriveNonlinearTests(unittest.TestCase):
    def test_design_is_odd(self):
        value = np.asarray([-0.2, -0.1, 0.1, 0.2], dtype=np.float32)
        features = design(value)
        self.assertTrue(np.allclose(features[:2], -features[:1:-1]))

    def test_restore_preserves_geometry(self):
        frames = 32
        response = np.ones(frames // 2 + 1, dtype=np.complex64)
        coefficients = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0])
        source = np.random.default_rng(11).standard_normal(frames).astype(np.float32) * 0.01
        result = restore(source, response, 1.0, coefficients, 1.0)
        self.assertTrue(np.allclose(result, source, atol=1.0e-6))


if __name__ == "__main__":
    unittest.main()
