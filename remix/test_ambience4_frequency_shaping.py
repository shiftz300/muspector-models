import unittest

import numpy as np
from scipy.signal import fftconvolve

from .ambience4_frequency_shaping import (
    FREQUENCY_SHAPING_CANDIDATES,
    frequency_profile_candidate_kernels,
    frequency_faded_transfer,
    partitioned_convolve_full,
    partitioned_frequency_profile_candidate_bank,
    profile_frequency_shortening_inverse,
)


class Ambience4FrequencyShapingTest(unittest.TestCase):
    def _fixture(self):
        rng = np.random.default_rng(23)
        clean = rng.normal(0.0, 0.03, 32_768).astype(np.float32)
        impulse = np.zeros(12_000, dtype=np.float32)
        impulse[0] = 1.0
        positions = np.arange(len(impulse) - 480)
        impulse[480:] = (
            0.012 * np.exp(-positions / 4_000.0) * np.sin(2 * np.pi * positions / 37)
            + 0.009 * np.exp(-positions / 1_400.0) * np.sin(2 * np.pi * positions / 7)
        )
        impulse /= np.sqrt(np.sum(impulse.astype(np.float64) ** 2))
        controls = {"mix": 0.45, "room_gain_db": -8.0}
        transfer = impulse * (controls["mix"] * 10.0 ** (controls["room_gain_db"] / 20.0))
        transfer[0] += 1.0 - controls["mix"]
        wet = fftconvolve(clean, transfer, mode="full")[: len(clean)].astype(np.float32)
        return wet, impulse, controls

    def test_frequency_fade_is_profile_only_and_changes_late_response(self):
        _, impulse, controls = self._fixture()
        desired, report = frequency_faded_transfer(
            impulse, controls, decay_ratio=0.45, early_ms=10.0
        )
        original = impulse * (controls["mix"] * 10.0 ** (controls["room_gain_db"] / 20.0))
        original[0] += 1.0 - controls["mix"]
        self.assertEqual(desired.shape, original.shape)
        self.assertTrue(np.isfinite(desired).all())
        self.assertTrue(np.allclose(desired[:480], original[:480], atol=1.0e-8))
        self.assertLess(np.linalg.norm(desired[2_000:]), np.linalg.norm(original[2_000:]))
        self.assertEqual(report["frequency_bins"], 513)

    def test_inverse_is_finite_and_order_independent(self):
        wet, impulse, controls = self._fixture()
        restored, report = profile_frequency_shortening_inverse(
            wet,
            impulse,
            controls,
            decay_ratio=0.45,
            early_ms=10.0,
            correction_strength=0.30,
        )
        self.assertEqual(restored.shape, wet.shape)
        self.assertTrue(np.isfinite(restored).all())
        self.assertFalse(report["clean_input"])
        self.assertFalse(report["chain_order_input"])
        self.assertFalse(report["graph_order_input"])
        self.assertFalse(report["neighbor_effect_input"])
        self.assertEqual(report["high_frequency_ratio"], 0.0)

    def test_invalid_decay_ratio_fails_closed(self):
        _, impulse, controls = self._fixture()
        with self.assertRaises(ValueError):
            frequency_faded_transfer(impulse, controls, decay_ratio=0.01, early_ms=10.0)

    def test_partitioned_convolution_matches_offline_for_irregular_tail(self):
        rng = np.random.default_rng(91)
        signal = rng.normal(0.0, 0.02, 12_345)
        kernel = rng.normal(0.0, 0.01, 3_111)
        expected = fftconvolve(signal, kernel, mode="full")
        actual = partitioned_convolve_full(signal, kernel, block_frames=1_777)
        self.assertLess(float(np.max(np.abs(actual - expected))), 1.0e-10)

    def test_partitioned_profile_bank_matches_offline_candidates(self):
        wet, impulse, controls = self._fixture()
        kernels, reports = frequency_profile_candidate_kernels(impulse, controls)
        actual = partitioned_frequency_profile_candidate_bank(
            wet, kernels, block_frames=3_071
        )
        expected = np.stack([
            fftconvolve(wet, kernel, mode="full")[4_096 : 4_096 + len(wet)]
            for kernel in kernels
        ]).astype(np.float32)
        self.assertEqual(actual.shape, (len(FREQUENCY_SHAPING_CANDIDATES), len(wet)))
        self.assertEqual(len(reports), len(FREQUENCY_SHAPING_CANDIDATES))
        self.assertLess(float(np.max(np.abs(actual - expected))), 2.0e-6)
        self.assertTrue(all(not report["clean_input"] for report in reports))


if __name__ == "__main__":
    unittest.main()
