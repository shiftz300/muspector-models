import tempfile
import unittest
from pathlib import Path

import numpy as np

from .blind2 import LABELS
from .guitar_chain_data import DIRECTORIES, discover, split_for_group, target_from_bits


class GuitarChainDataTests(unittest.TestCase):
    def test_presence_bits_never_become_order_labels(self) -> None:
        target = target_from_bits("10101")
        expected = {
            "nonlinear": 1.0,
            "echo": 0.0,
            "ambience": 1.0,
            "unknown": 1.0,
        }
        np.testing.assert_array_equal(target, [expected[label] for label in LABELS])
        with self.assertRaises(ValueError):
            target_from_bits("overdrive-first")

    def test_every_variant_of_a_performance_stays_in_one_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean = root / DIRECTORIES["clean"]
            effected = root / DIRECTORIES["random_position"]
            clean.mkdir()
            effected.mkdir()
            group = "strat_neck_pick15"
            (clean / f"{group}.wav").touch()
            (effected / f"{group}__10000.wav").touch()
            (effected / f"{group}__01101.wav").touch()
            split = split_for_group(group)
            rows = discover(root, split)
            self.assertEqual(len(rows), 3)
            self.assertEqual({row.group for row in rows}, {group})
            self.assertEqual({row.split for row in rows}, {split})

    def test_whole_guitar_locked_final_boundary_matches_existing_policy(self) -> None:
        self.assertEqual(split_for_group("les_bridge_fing01"), "fit")
        self.assertEqual(split_for_group("prs_neck_pick25"), "fit")
        self.assertEqual(split_for_group("strat_neck_pick12"), "calibration")
        self.assertEqual(split_for_group("strat_neck_pick13"), "development")
        self.assertEqual(split_for_group("tele_bridge_pick01"), "locked-final")


if __name__ == "__main__":
    unittest.main()
