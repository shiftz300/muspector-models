import unittest
from pathlib import Path

import torch

from .ambience4 import (
    DECAY_DOMAIN_SECONDS,
    DECAY_BANK_VARIANTS,
    DECAY_STRATA_SECONDS,
    PRODUCT4_CLEAN_SOURCE_IDS,
    TARGET_FRAMES_MINIMUM,
    AmbiencePairsV4,
    AmbiencePairsV4DecayBank,
    AmbiencePairsV4ProfileBank,
    AmbiencePairsV4WetBase,
    multiband_decay_suppression,
    release_event_frames,
)


ROOT = Path(__file__).resolve().parents[1]


class Ambience4Tests(unittest.TestCase):
    def test_room_disjoint_but_pair_and_authorization(self):
        dataset = AmbiencePairsV4(ROOT, "development", 2, TARGET_FRAMES_MINIMUM, 20260904)
        row = dataset[0]
        self.assertIn("but-reverbdb", dataset.authorization["sources"])
        self.assertEqual(dataset.authorization["required_uses"]["but-reverbdb"], ["train-reverb"])
        self.assertEqual(DECAY_DOMAIN_SECONDS, (0.10, 1.00))
        self.assertEqual(len(row["wet"]), dataset.history_frames + TARGET_FRAMES_MINIMUM)
        self.assertTrue(row["rir"].endswith(".wav"))
        self.assertTrue(row["rir"].startswith(("Hotel_SkalskyDvur_ConferenceRoom2/", "VUT_FIT_E112/")))
        self.assertGreaterEqual(
            release_event_frames(row["clean"][row["target_start"]:].numpy()), 4
        )

    def test_product4_has_three_tail_strata_and_artifact_safe_wet_base(self):
        self.assertEqual(DECAY_STRATA_SECONDS, (0.40, 0.75))
        self.assertEqual(AmbiencePairsV4.decay_stratum(0.39), "short-tail")
        self.assertEqual(AmbiencePairsV4.decay_stratum(0.40), "medium-tail")
        self.assertEqual(AmbiencePairsV4.decay_stratum(0.75), "long-tail")
        dataset = AmbiencePairsV4WetBase(
            ROOT, "development", 2, TARGET_FRAMES_MINIMUM, 20260904
        )
        row = dataset[0]
        self.assertTrue(row["late_base"].equal(row["wet"]))

    def test_product4_crosses_sources_with_rooms(self):
        dataset = AmbiencePairsV4(ROOT, "development", 8, TARGET_FRAMES_MINIMUM, 20260904)
        pairs = {
            (
                dataset._selection(index).source_id,
                dataset._rir_selection(index).relative_to(dataset.rir_root).parts[0],
            )
            for index in range(8)
        }
        expected = {
            (source, room)
            for source in dataset.source_ids
            for room in ("Hotel_SkalskyDvur_ConferenceRoom2", "VUT_FIT_E112")
        }
        self.assertEqual(pairs, expected)
        cells = {
            (
                dataset.room_group(dataset[index]),
                dataset.decay_stratum(
                    float(dataset[index]["control_values"]["decay_p999_seconds"])
                ),
            )
            for index in range(8)
        }
        self.assertEqual(cells, set(dataset.rir_cells))

    def test_product4_rejects_a_target_shorter_than_the_reverb_domain(self):
        with self.assertRaisesRegex(ValueError, "target must cover"):
            AmbiencePairsV4(ROOT, "development", 1, 65_535, 20260904)

    def test_product4_excludes_short_note_sources_from_reverb_gradients(self):
        dataset = AmbiencePairsV4(ROOT, "fit", 5, TARGET_FRAMES_MINIMUM, 20260904)
        self.assertEqual(set(dataset.source_ids), set(PRODUCT4_CLEAN_SOURCE_IDS))
        self.assertNotIn("eg-ipt", dataset.authorization["sources"])
        self.assertNotIn("freepats-electric-guitar-direct", dataset.authorization["sources"])
        self.assertNotIn("karoryfer-emilyguitar", dataset.authorization["sources"])

    def test_multiband_decay_candidate_is_bounded_and_wet_only(self):
        torch.manual_seed(7)
        wet = torch.randn(2, 4096) * 0.05
        restored = multiband_decay_suppression(
            wet, half_life_ms=350.0, ratio_threshold=0.65, strength=0.8
        )
        self.assertEqual(restored.shape, wet.shape)
        self.assertTrue(torch.isfinite(restored).all())
        self.assertLessEqual(float(restored.square().mean()), float(wet.square().mean()) * 1.02)
        silence = torch.zeros(1, 4096)
        self.assertTrue(torch.equal(
            multiband_decay_suppression(
                silence, half_life_ms=350.0, ratio_threshold=0.65, strength=0.8
            ),
            silence,
        ))

    def test_decay_bank_is_six_bounded_order_independent_candidates(self):
        self.assertEqual(len(DECAY_BANK_VARIANTS), 6)
        self.assertTrue(all(row[2] == 0.8 for row in DECAY_BANK_VARIANTS))
        dataset = AmbiencePairsV4DecayBank(
            ROOT, "development", 1, TARGET_FRAMES_MINIMUM, 20260904
        )
        row = dataset[0]
        self.assertEqual(row["late_base"].shape, (6, len(row["wet"])))
        self.assertEqual(
            dataset.quality_contract, "tail-removal-with-global-nonregression"
        )

    def test_profile_bank_is_physical_and_order_independent(self):
        dataset = AmbiencePairsV4ProfileBank(
            ROOT, "development", 1, TARGET_FRAMES_MINIMUM, 20260912
        )
        row = dataset[0]
        self.assertEqual(row["late_base"].shape, (4, len(row["wet"])))
        self.assertIn(row["profile_mode"], {"exact", "shaping-bank"})
        self.assertNotIn("chain_order", row)
        self.assertNotIn("neighbor_effect", row)
        self.assertTrue(torch.isfinite(row["late_base"]).all())


if __name__ == "__main__":
    unittest.main()
