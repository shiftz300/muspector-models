import unittest
from pathlib import Path

import numpy as np

from .product_data import ProductPairs, audit, discover_clean


ROOT = Path(__file__).resolve().parents[1]


class ProductDataTests(unittest.TestCase):
    def test_product_audit_is_authorized_and_research_free(self):
        report = audit(ROOT)
        self.assertEqual(report["status"], "product-pair-generation-authorized")
        self.assertTrue(report["authorization"]["authorized"])
        self.assertFalse(report["research_sources_present"])
        self.assertFalse(report["physical_audio_devices_used"])

    def test_three_mechanisms_are_deterministic_finite_pairs(self):
        for mechanism in ("nonlinear", "dynamics", "temporal"):
            dataset = ProductPairs(ROOT, mechanism, "calibration", 2, 4096, 77)
            first, replay = dataset[0], dataset[0]
            self.assertEqual(first["wet"].shape, first["clean"].shape)
            self.assertTrue(np.isfinite(first["wet"].numpy()).all())
            np.testing.assert_array_equal(first["wet"].numpy(), replay["wet"].numpy())
            self.assertNotEqual(float(np.mean(np.abs(first["wet"].numpy() - first["clean"].numpy()))), 0.0)

    def test_locked_final_is_not_needed_for_development_dataset(self):
        dataset = ProductPairs(ROOT, "nonlinear", "development", 2, 4096, 88)
        self.assertTrue(all(item.split == "development" for item in dataset.clean))

    def test_single_session_repeated_routine_and_sample_banks_are_fit_only(self):
        rows = [
            item
            for item in discover_clean(ROOT)
            if item.source_id
            in {
                "eg-ipt",
                "longitudinal-guitar-string-ageing",
                "freepats-electric-guitar-direct",
                "karoryfer-emilyguitar",
            }
        ]
        longitudinal = [
            item for item in rows if item.source_id == "longitudinal-guitar-string-ageing"
        ]
        self.assertEqual(len(longitudinal), 56)
        self.assertEqual(len({item.group for item in longitudinal}), 2)
        self.assertTrue(any(item.source_id == "freepats-electric-guitar-direct" for item in rows))
        self.assertTrue(any(item.source_id == "karoryfer-emilyguitar" for item in rows))
        self.assertTrue(all(item.split == "fit" for item in rows))


if __name__ == "__main__":
    unittest.main()
