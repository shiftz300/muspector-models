"""Tests for the safe Stone-style Phaser inverse."""

from __future__ import annotations

import random
import unittest

import numpy as np

from .phaser3 import invert_known_phase, manifest, render_phaser


class StonePhaserV3Tests(unittest.TestCase):
    def test_known_phase_round_trip(self) -> None:
        rng = np.random.default_rng(19)
        clean = (0.08 * rng.standard_normal(12_000)).astype(np.float32)
        wet, values, _ = render_phaser(clean, random.Random(23))
        restored = invert_known_phase(wet, values, values["hidden_phase_radians"])
        self.assertLess(float(np.max(np.abs(restored - clean))), 2.0e-6)

    def test_manifest_forbids_order_and_oracle_inputs(self) -> None:
        record = manifest(0.04)
        self.assertFalse(record["graph_order_input"])
        self.assertFalse(record["neighbouring_effect_input"])
        self.assertFalse(record["clean_or_oracle_input"])
        self.assertEqual(record["ambiguous_input_behavior"], "abstain-and-pass-through")
        self.assertEqual(record["feedback_domain"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
