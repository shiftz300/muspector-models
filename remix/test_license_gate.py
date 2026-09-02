import copy
import unittest
from pathlib import Path

from .license_gate import audit, authorize_product_weights, load_registry


REGISTRY = Path(__file__).with_name("data_sources.json")


class LicenseGateTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_registry(REGISTRY)

    def test_live_registry_passes(self):
        report = audit(self.registry)
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["foundation_product_training"]["authorized"])

    def test_nc_apple_gpl_and_unclear_sources_are_blocked(self):
        result = authorize_product_weights(
            self.registry,
            (
                "asrnn-physical-effects",
                "apple-au-local",
                "spotify-pedalboard-renderer",
                "pod-set",
            ),
        )
        self.assertFalse(result["authorized"])
        self.assertEqual(len(result["blocked"]), 4)

    def test_cc_by_product_source_emits_attribution(self):
        result = authorize_product_weights(self.registry, ("aachen-chapel-rir",))
        self.assertTrue(result["authorized"])
        self.assertEqual(result["required_attribution"][0]["license"], "CC-BY-4.0")

    def test_marshall_amp_source_is_product_authorized_and_scoped(self):
        result = authorize_product_weights(self.registry, ("marshall-jvm410h",))
        self.assertTrue(result["authorized"], result)
        self.assertEqual(result["required_attribution"][0]["source_id"], "marshall-jvm410h")
        source = next(row for row in self.registry["sources"] if row["id"] == "marshall-jvm410h")
        self.assertEqual(source["gradient_scope"], "product-amp-restoration")
        self.assertIn("Gain values are 0, 2, 4, 5 and 10", source["control_labels"])
        self.assertIn("Gain 1 and 8 are development-only", source["control_labels"])
        self.assertIn("one quarantined Gain=6 naming contradiction", source["pairing"])

    def test_random_position_chain_source_separates_presence_and_order(self):
        source = next(
            row for row in self.registry["sources"] if row["id"] == "dafx25-guitar-effects-chains"
        )
        self.assertIn("train-family", source["allowed_uses"])
        self.assertIn("validate-family", source["allowed_uses"])
        self.assertIn("train-order", source["allowed_uses"])
        self.assertIn("validate-order", source["allowed_uses"])
        self.assertIn("publishes realized orders separately", source["pairing"])
        self.assertIn("separate order package", source["audio_quality_caveats"][1])

    def test_new_clean_sources_are_product_authorized(self):
        result = authorize_product_weights(
            self.registry,
            (
                "eg-ipt",
                "longitudinal-guitar-string-ageing",
                "freepats-electric-guitar-direct",
                "karoryfer-emilyguitar",
            ),
        )
        self.assertTrue(result["authorized"], result)
        self.assertEqual(
            result["sources"],
            [
                "eg-ipt",
                "longitudinal-guitar-string-ageing",
                "freepats-electric-guitar-direct",
                "karoryfer-emilyguitar",
            ],
        )
        self.assertEqual(
            [row["source_id"] for row in result["required_attribution"]],
            ["eg-ipt", "longitudinal-guitar-string-ageing"],
        )

    def test_discovered_but_ineligible_candidates_stay_blocked(self):
        result = authorize_product_weights(
            self.registry,
            ("goat-guitar-di", "gada-guitar-audio", "five-guitar-dataset", "does-it-chug"),
        )
        self.assertFalse(result["authorized"])
        self.assertEqual(
            {row["id"] for row in result["blocked"]},
            {"goat-guitar-di", "gada-guitar-audio", "five-guitar-dataset", "does-it-chug"},
        )

    def test_restricted_license_cannot_be_mislabeled_as_releasable(self):
        registry = copy.deepcopy(self.registry)
        source = next(row for row in registry["sources"] if row["id"] == "asrnn-physical-effects")
        source["weight_scope"] = "redistributable"
        report = audit(registry)
        self.assertFalse(report["passed"])
        self.assertTrue(report["contradictions"])


if __name__ == "__main__":
    unittest.main()
