import copy
import unittest
from pathlib import Path

from .license_gate import (
    audit,
    authorize_product_uses,
    authorize_product_weights,
    load_registry,
)


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
                "remfx-pretrained-models",
                "pod-set",
            ),
        )
        self.assertFalse(result["authorized"])
        self.assertEqual(len(result["blocked"]), 5)

    def test_remfx_code_license_does_not_admit_nc_checkpoints(self):
        result = authorize_product_weights(self.registry, ("remfx-pretrained-models",))
        self.assertFalse(result["authorized"])
        self.assertEqual(
            result["blocked"],
            [{"id": "remfx-pretrained-models", "reason": "restricted-license:CC-NC"}],
        )

    def test_cc_by_product_source_emits_attribution(self):
        result = authorize_product_weights(self.registry, ("aachen-chapel-rir",))
        self.assertTrue(result["authorized"])
        self.assertEqual(result["required_attribution"][0]["license"], "CC-BY-4.0")

    def test_task_use_blocks_egfx_wet_from_restoration_gradients(self):
        result = authorize_product_uses(
            self.registry,
            {"egfxset": "train-restoration"},
        )
        self.assertFalse(result["authorized"])
        self.assertEqual(
            result["blocked"],
            [{"id": "egfxset", "reason": "use-not-allowed:train-restoration"}],
        )

    def test_task_use_allows_egfx_family_and_clean_roles(self):
        result = authorize_product_uses(
            self.registry,
            {"egfxset": ("train-family", "product-clean-source")},
        )
        self.assertTrue(result["authorized"], result)

    def test_task_use_allows_measured_reverb_and_amp_training(self):
        result = authorize_product_uses(
            self.registry,
            {
                "aachen-chapel-rir": "train-reverb",
                "marshall-jvm410h": "train-amp",
            },
        )
        self.assertTrue(result["authorized"], result)

    def test_task_use_blocks_guitar_techs_amp_graybox_training(self):
        result = authorize_product_uses(
            self.registry,
            {"guitar-techs": "train-amp"},
        )
        self.assertFalse(result["authorized"])
        self.assertEqual(result["blocked"][0]["reason"], "use-not-allowed:train-amp")

    def test_ok5_is_external_validation_only(self):
        validation = authorize_product_uses(
            self.registry,
            {"ok5-rir": "validate-reverb"},
        )
        training = authorize_product_uses(
            self.registry,
            {"ok5-rir": "train-reverb"},
        )
        self.assertTrue(validation["authorized"], validation)
        self.assertFalse(training["authorized"])
        self.assertEqual(training["blocked"][0]["reason"], "use-not-allowed:train-reverb")

    def test_openair_is_fresh_external_validation_only(self):
        validation = authorize_product_uses(
            self.registry,
            {"openair-rir-external-v1": "validate-reverb"},
        )
        training = authorize_product_uses(
            self.registry,
            {"openair-rir-external-v1": "train-reverb"},
        )
        self.assertTrue(validation["authorized"], validation)
        self.assertFalse(training["authorized"])
        self.assertEqual(training["blocked"][0]["reason"], "use-not-allowed:train-reverb")

    def test_openslr26_is_product_authorized_and_room_split(self):
        validation = authorize_product_uses(
            self.registry,
            {"openslr26-simulated-rir-external-v1": "validate-reverb"},
        )
        training = authorize_product_uses(
            self.registry,
            {"openslr26-simulated-rir-external-v1": "train-reverb"},
        )
        self.assertTrue(validation["authorized"], validation)
        self.assertTrue(training["authorized"], training)
        source = next(
            row for row in self.registry["sources"]
            if row["id"] == "openslr26-simulated-rir-external-v1"
        )
        self.assertIn("Room001-180 only", source["gradient_scope"])
        self.assertIn("Room191-200", source["audio_quality_caveats"][2])

    def test_rochester_rir_is_reserved_validation_only_and_unopened(self):
        validation = authorize_product_uses(
            self.registry,
            {"rochester-rir-fresh-v2": "validate-reverb"},
        )
        training = authorize_product_uses(
            self.registry,
            {"rochester-rir-fresh-v2": "train-reverb"},
        )
        self.assertTrue(validation["authorized"], validation)
        self.assertFalse(training["authorized"])
        self.assertEqual(training["blocked"][0]["reason"], "use-not-allowed:train-reverb")
        source = next(
            row for row in self.registry["sources"]
            if row["id"] == "rochester-rir-fresh-v2"
        )
        self.assertEqual(source["local_paths"], [])
        self.assertIn("unopened", source["admission"])

    def test_marshall_amp_source_is_product_authorized_and_scoped(self):
        result = authorize_product_weights(self.registry, ("marshall-jvm410h",))
        self.assertTrue(result["authorized"], result)
        self.assertEqual(result["required_attribution"][0]["source_id"], "marshall-jvm410h")
        source = next(row for row in self.registry["sources"] if row["id"] == "marshall-jvm410h")
        self.assertEqual(source["gradient_scope"], "product-amp-restoration")
        self.assertIn("Gain values are 0, 2, 4, 5 and 10", source["control_labels"])
        self.assertIn("Gain 1 and 8 are development-only", source["control_labels"])
        self.assertIn("one quarantined Gain=6 naming contradiction", source["pairing"])

    def test_tonetwist_nc_amp_records_are_metadata_only(self):
        ids = (
            "tonetwist-blackstar-ht1-overdrive",
            "tonetwist-blackstar-ht5-metal-overdrive",
            "tonetwist-engl-retro-tube-50-drive",
            "tonetwist-fender-blues-jr",
            "tonetwist-ibanez-tsa15-crunch",
            "tonetwist-mesaboogie-550-clean",
            "tonetwist-mesaboogie-550-crunch",
            "tonetwist-mesaboogie-550-burn",
            "tonetwist-mesaboogie-mark-v-clean",
            "tonetwist-mesaboogie-mark-v-crunch",
            "tonetwist-mesaboogie-mark-v-extreme",
            "tonetwist-ua-6176-610b-preamp",
            "tonetwist-harley-benton-rodent",
            "tonetwist-multidrive-808-scream",
        )
        result = authorize_product_uses(
            self.registry, {source_id: "train-amp" for source_id in ids}
        )
        self.assertFalse(result["authorized"])
        self.assertEqual(len(result["blocked"]), len(ids))
        self.assertTrue(
            all(row["reason"] == "restricted-license:CC-BY-NC-4.0" for row in result["blocked"])
        )

    def test_new_online_amp_candidates_remain_fail_closed(self):
        ids = (
            "rockingface-amp-space",
            "automated-guitar-amp-modelling-example-audio",
            "openamp-proteus-tone-packs",
            "open-riff-box-stage-renderer",
            "byod-source-renderer",
            "swankyamp-source-renderer",
            "goat-guitar-audio-tablatures",
            "five-guitar-dataset",
            "egdb-bias-fx2-distortion-recovery",
            "guitar-fx-dist",
        )
        result = authorize_product_weights(self.registry, ids)
        self.assertFalse(result["authorized"])
        self.assertEqual({row["id"] for row in result["blocked"]}, set(ids))

    def test_tonetwist_software_chain_deposit_stays_metadata_only(self):
        source_id = "tonetwist-audacity-comp-dist-reverb-chain"
        result = authorize_product_weights(self.registry, (source_id,))
        self.assertFalse(result["authorized"])
        self.assertEqual(result["blocked"][0]["id"], source_id)
        source = next(row for row in self.registry["sources"] if row["id"] == source_id)
        self.assertEqual(source["license"], "CC-BY-4.0")
        self.assertFalse(source["hardware"])
        self.assertEqual(source["allowed_uses"], ["metadata-review"])
        self.assertIn("YouTube-derived", source["audio_quality_caveats"][0])

    def test_open_riff_box_stage_renderer_is_research_only_and_asset_free(self):
        result = authorize_product_uses(
            self.registry, {"open-riff-box-stage-renderer": "internal-stage-supervision"}
        )
        self.assertFalse(result["authorized"])
        source = next(
            row for row in self.registry["sources"]
            if row["id"] == "open-riff-box-stage-renderer"
        )
        self.assertEqual(source["gradient_scope"], "research-only-architecture-selection")
        self.assertEqual(source["weight_scope"], "blocked-from-product-weights")
        self.assertIn("no-cabinet", source["audio_scope"])
        self.assertIn("discarded", source["audio_quality_caveats"][4])

    def test_rusty_amp_stage_renderer_is_product_scoped_and_asset_free(self):
        result = authorize_product_uses(
            self.registry, {"rusty-amp-stage-renderer": "train-amp"}
        )
        self.assertTrue(result["authorized"], result)
        source = next(
            row for row in self.registry["sources"]
            if row["id"] == "rusty-amp-stage-renderer"
        )
        self.assertEqual(source["license"], "Apache-2.0")
        self.assertIn("no-cabinet", source["audio_scope"])
        self.assertIn("physical-amplifier", source["audio_quality_caveats"][0])

    def test_eg_ipt_physical_amp_pairs_are_fit_only_and_attributed(self):
        result = authorize_product_uses(self.registry, {"eg-ipt": "train-amp"})
        self.assertTrue(result["authorized"], result)
        self.assertEqual(result["required_attribution"][0]["source_id"], "eg-ipt")
        source = next(row for row in self.registry["sources"] if row["id"] == "eg-ipt")
        self.assertEqual(source["license"], "CC-BY-4.0")
        self.assertIn("fit-only", source["gradient_scope"])
        self.assertNotIn("validate-amp", source["allowed_uses"])
        self.assertEqual(source["amp_cab_sm57_audio"]["files"], 8717)

    def test_pretrained_audio_encoder_candidates_remain_fail_closed(self):
        ids = (
            "mert-pretrained-audio-encoder",
            "audiomae-pretrained-audio-encoder",
            "laion-clap-pretrained-audio-encoder",
        )
        result = authorize_product_weights(self.registry, ids)
        self.assertFalse(result["authorized"])
        self.assertEqual({row["id"] for row in result["blocked"]}, set(ids))

    def test_egdb_pg_amp_cab_subset_is_product_authorized_and_scoped(self):
        result = authorize_product_uses(
            self.registry, {"egdb-pg-v2": ("train-amp-cab", "validate-amp-cab")}
        )
        self.assertTrue(result["authorized"], result)
        self.assertEqual(result["required_attribution"][0]["source_id"], "egdb-pg-v2")
        source = next(row for row in self.registry["sources"] if row["id"] == "egdb-pg-v2")
        self.assertIn("profile-disjoint", source["gradient_scope"])
        self.assertIn("Amp+cab", source["audio_quality_caveats"][2])

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
