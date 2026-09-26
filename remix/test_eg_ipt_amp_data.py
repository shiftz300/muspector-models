import unittest
from pathlib import Path

from .eg_ipt_amp_data import EgIptAmpPairs, discover, split_summary, wet_relative


class EgIptAmpDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = Path(__file__).resolve().parents[1]

    def test_wet_path_mapping_is_exact(self):
        source = Path("EG-IPT/HB-neck/DI/vibrato/HB-neck_vibrato-fast_1_DI.wav")
        self.assertEqual(
            wet_relative(source),
            Path("EG-IPT/HB-neck/dyn/vibrato/HB-neck_vibrato-fast_1_dyn.wav"),
        )

    def test_audited_pair_count_and_fit_only_splits(self):
        self.assertEqual(len(discover(self.workspace)), 8717)
        summary = split_summary(self.workspace)
        self.assertEqual(sum(row["pairs"] for row in summary.values()), 8717)
        for row in summary.values():
            self.assertEqual(len(row["pickups"]), 3)
            self.assertEqual(len(row["techniques"]), 19)
            self.assertEqual(row["product_gate_role"], "fit-only")

    def test_crop_is_finite_aligned_and_order_free(self):
        dataset = EgIptAmpPairs(self.workspace, "fit", 2, 4096, 1024, 20260905)
        row = dataset[0]
        self.assertEqual(tuple(row["wet"].shape), (6144,))
        self.assertEqual(tuple(row["clean"].shape), (6144,))
        self.assertTrue(row["wet"].isfinite().all())
        self.assertTrue(row["clean"].isfinite().all())
        self.assertEqual(row["product_gate_role"], "fit-only")
        self.assertNotIn("graph_order", row)
        self.assertNotIn("neighbor_effect", row)
        self.assertGreaterEqual(dataset.short_pairs_excluded, 0)


if __name__ == "__main__":
    unittest.main()
