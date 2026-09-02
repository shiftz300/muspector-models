import unittest
from pathlib import Path

import numpy as np

from .ambience2 import (
    HISTORY_FRAMES,
    AmbienceAbstention,
    AmbiencePairsV2,
    restore_known_ambience_profile,
)
from .product_data import _rir


ROOT = Path(__file__).resolve().parents[1]


class Ambience2Tests(unittest.TestCase):
    def test_long_state_and_realized_product_provenance(self):
        dataset = AmbiencePairsV2(ROOT, "development", 4, 16384, 91)
        self.assertEqual(HISTORY_FRAMES, 108000)
        self.assertEqual(len(dataset[0]["wet"]), HISTORY_FRAMES + 16384)
        self.assertEqual(dataset.realized_source_counts(), {
            "dafx25-guitar-effects-chains": 2,
            "egfxset": 2,
        })
        self.assertTrue(dataset.authorization["authorized"])
        self.assertIn("aachen-chapel-rir", dataset.authorization["sources"])

    def test_target_contains_reverberation_and_finite_controls(self):
        row = AmbiencePairsV2(ROOT, "development", 1, 16384, 92)[0]
        start = row["target_start"]
        distance = np.sqrt(np.mean((row["wet"][start:].numpy() - row["clean"][start:].numpy()) ** 2))
        self.assertGreater(distance, 1.0e-5)
        self.assertTrue(np.isfinite(row["controls"].numpy()).all())

    def test_known_stable_profile_is_exact_and_unstable_profile_abstains(self):
        dataset = AmbiencePairsV2(ROOT, "development", 1, 16384, 20260910)
        row = dataset[0]
        impulse = _rir(dataset.rirs[0])
        restored, report = restore_known_ambience_profile(
            row["wet"].numpy(), impulse, row["control_values"]
        )
        np.testing.assert_allclose(restored, row["clean"].numpy(), atol=3.0e-6, rtol=3.0e-6)
        self.assertFalse(report["chain_order_input"])
        with self.assertRaises(AmbienceAbstention):
            restore_known_ambience_profile(
                row["wet"].numpy(), impulse, row["control_values"], maximum_inverse_coefficient=1.0
            )


if __name__ == "__main__":
    unittest.main()
