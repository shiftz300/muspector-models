import unittest

import numpy as np

from remix.router import level


class RouterTests(unittest.TestCase):
    def test_level_is_gain_invariant_and_read_only(self):
        source = np.linspace(-0.2, 0.2, 1000, dtype=np.float32)
        before = source.copy()
        np.testing.assert_allclose(level(source), level(source * 7.0), rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(source, before)

    def test_level_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            level(np.array([0.0, np.nan], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
