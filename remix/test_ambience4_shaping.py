import unittest

import numpy as np
from scipy.signal import fftconvolve

from .ambience4_shaping import profile_shortening_inverse


class Ambience4ShapingTest(unittest.TestCase):
    def test_profile_shortening_is_finite_and_order_independent(self):
        rng = np.random.default_rng(7)
        clean = rng.normal(0.0, 0.03, 32_768).astype(np.float32)
        impulse = np.zeros(8_192, dtype=np.float32)
        impulse[0] = 1.0
        impulse[480:] = 0.02 * np.exp(-np.arange(len(impulse) - 480) / 2_000.0)
        impulse /= np.sqrt(np.sum(impulse.astype(np.float64) ** 2))
        controls = {"mix": 0.45, "room_gain_db": -8.0}
        transfer = impulse * (controls["mix"] * 10.0 ** (controls["room_gain_db"] / 20.0))
        transfer[0] += 1.0 - controls["mix"]
        wet = fftconvolve(clean, transfer, mode="full")[: len(clean)].astype(np.float32)
        restored, report = profile_shortening_inverse(
            wet,
            impulse,
            controls,
            early_ms=10.0,
            target_rt60_ms=160.0,
            maximum_gain=4.0,
            low_band_strength=0.30,
            high_band_strength=0.0,
        )
        self.assertEqual(restored.shape, wet.shape)
        self.assertTrue(np.isfinite(restored).all())
        self.assertFalse(report["clean_input"])
        self.assertFalse(report["chain_order_input"])
        self.assertFalse(report["graph_order_input"])
        self.assertFalse(report["neighbor_effect_input"])
        self.assertEqual(report["transition_band_hz"], [2_000.0, 4_000.0])

    def test_profile_shortening_rejects_invalid_strength(self):
        with self.assertRaises(ValueError):
            profile_shortening_inverse(
                np.zeros(8_192, dtype=np.float32),
                np.asarray([1.0], dtype=np.float32),
                {"mix": 0.45, "room_gain_db": -8.0},
                early_ms=10.0,
                target_rt60_ms=160.0,
                maximum_gain=4.0,
                low_band_strength=0.20,
                high_band_strength=0.30,
            )


if __name__ == "__main__":
    unittest.main()
