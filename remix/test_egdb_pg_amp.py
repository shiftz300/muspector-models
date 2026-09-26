import unittest
import json
from pathlib import Path

import torch

from .egdb_pg_amp_model import WetConditionedAmpCabInverse
from .egdb_pg_amp_model2 import WetSpectralAmpCabInverse
from .egdb_pg_amp_model3 import WetHybridAmpCabInverse
from .egdb_pg_amp_model4 import WetToneConditionedAmpCabInverse
from .egdb_pg_amp_model5 import WetToneComplexAmpCabInverse
from .egdb_pg_amp_model6 import WetToneComplexDynamicsAmpCabInverse
from .egdb_pg_amp_model7 import WetToneComplexTemporalAmpCabInverse
from .egdb_pg_amp_model8 import (
    WetToneComplexUNetAmpCabInverse,
    WetToneComplexUNetJointAmpCabInverse,
)
from .egdb_pg_amp_model9 import WetToneComplexDemucsJointAmpCabInverse
from .egdb_pg_amp_model10 import (
    WetToneGrayBoxAmpCabInverse,
    WetTonePhaseGrayBoxAmpCabInverse,
)
from .egdb_pg_amp_model11 import WetTonePhaseGrayBoxDynamicsAmpCabInverse
from .egdb_pg_amp_model12 import WetTonePhaseGrayBoxMultibandDynamicsAmpCabInverse
from .open_riff_box_stage_model import (
    OpenRiffBoxStagewiseGrayBoxInverse,
    OpenRiffBoxStagewiseTransientGrayBoxInverse,
)
from .rusty_amp_stage_model import RustyAmpStagewiseGrayBoxInverse
from .egdb_pg_tone_encoder import WetToneEncoder


class EgdbPgAmpTests(unittest.TestCase):
    def test_wet_only_safe_scale_initialization_and_contract(self):
        model = WetConditionedAmpCabInverse(16, 4, 16).eval()
        wet = torch.randn(2, 8192) * 0.05
        with torch.inference_mode():
            restored, uncertainty, state = model(wet)
        torch.testing.assert_close(restored, wet * 0.1, atol=1.0e-6, rtol=1.0e-6)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        manifest = model.manifest()
        for key in (
            "profile_id_input", "gain_category_input", "graph_order_input",
            "neighbor_effect_input", "recurrent_state_input", "clean_or_oracle_input",
        ):
            self.assertFalse(manifest[key])

    def test_external_state_is_rejected(self):
        with self.assertRaises(ValueError):
            WetConditionedAmpCabInverse(16, 4, 16)(
                torch.zeros(1, 4096), torch.zeros(1)
            )

    def test_spectral_model_is_wet_only_and_starts_at_safe_scale(self):
        model = WetSpectralAmpCabInverse(12, 4, 12).eval()
        wet = torch.randn(2, 8192) * 0.05
        with torch.inference_mode():
            restored, uncertainty, state = model(wet)
        torch.testing.assert_close(restored, wet * 0.1, atol=2.0e-5, rtol=2.0e-4)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        self.assertEqual(model.manifest()["conditioning_source"], "Wet log-spectrum statistics only")

    def test_hybrid_starts_as_spectral_stage(self):
        spectral = WetSpectralAmpCabInverse(16, 4, 16).eval()
        hybrid = WetHybridAmpCabInverse(16, 4, 16).eval()
        hybrid.spectral.load_state_dict(spectral.state_dict())
        wet = torch.randn(1, 8192) * 0.05
        with torch.inference_mode():
            expected, _, _ = spectral(wet)
            restored, _, _ = hybrid(wet)
        torch.testing.assert_close(restored, expected)
        self.assertFalse(hybrid.manifest()["profile_id_input"])

    def test_tone_encoder_is_wet_only_and_normalized(self):
        model = WetToneEncoder(16).eval()
        with torch.inference_mode():
            embedding = model(torch.randn(2, 8192) * 0.05)
        self.assertEqual(embedding.shape, (2, 16))
        torch.testing.assert_close(torch.linalg.vector_norm(embedding, dim=1), torch.ones(2))
        self.assertFalse(model.manifest()["profile_id_input"])

    def test_tone_conditioned_decoder_is_wet_only_and_starts_safe(self):
        model = WetToneConditionedAmpCabInverse(16, 4, 16).eval()
        wet = torch.randn(2, 8192) * 0.05
        reference = torch.randn(2, 3 * 44100) * 0.05
        with torch.inference_mode():
            restored, uncertainty, state = model(wet, tone_reference=reference)
        torch.testing.assert_close(restored, wet * 0.1, atol=1.0e-6, rtol=1.0e-6)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertTrue(manifest["tone_encoder_frozen"])
        for key in (
            "profile_id_input", "gain_category_input", "graph_order_input",
            "neighbor_effect_input", "recurrent_state_input", "clean_or_oracle_input",
        ):
            self.assertFalse(manifest[key])

    def test_complex_decoder_changes_phase_policy_but_starts_safe(self):
        model = WetToneComplexAmpCabInverse(12, 4, 16).eval()
        wet = torch.randn(1, 8192) * 0.05
        reference = torch.randn(1, 3 * 44100) * 0.05
        with torch.inference_mode():
            restored, uncertainty, state = model(wet, tone_reference=reference)
        torch.testing.assert_close(restored, wet * 0.1, atol=3.0e-5, rtol=3.0e-4)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertIn("do not preserve Wet phase", manifest["phase_policy"])
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbor_effect_input"])

    def test_dynamics_stage_starts_as_identity_over_complex_base(self):
        model = WetToneComplexDynamicsAmpCabInverse(12, 4, 16).eval()
        wet = torch.randn(1, 8192) * 0.05
        reference = torch.randn(1, 3 * 44100) * 0.05
        with torch.inference_mode():
            expected, _, _ = model.base(wet, tone_reference=reference)
            restored, uncertainty, state = model(wet, tone_reference=reference)
        torch.testing.assert_close(restored, expected, atol=1.0e-6, rtol=1.0e-6)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertTrue(manifest["base_complex_inverse_frozen"])
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["clean_or_oracle_input"])

    def test_temporal_stage_starts_as_identity_over_complex_base(self):
        model = WetToneComplexTemporalAmpCabInverse(12, 4, 16).eval()
        wet = torch.randn(1, 8192) * 0.05
        reference = torch.randn(1, 3 * 44100) * 0.05
        with torch.inference_mode():
            expected, _, _ = model.base(wet, tone_reference=reference)
            restored, uncertainty, state = model(wet, tone_reference=reference)
        torch.testing.assert_close(restored, expected, atol=1.0e-6, rtol=1.0e-6)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertGreater(manifest["temporal_receptive_field_seconds"], 0.70)
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["gain_category_input"])

    def test_waveform_unet_starts_as_identity_over_complex_base(self):
        model = WetToneComplexUNetAmpCabInverse(12, 4, 16).eval()
        wet = torch.randn(1, 8192) * 0.05
        reference = torch.randn(1, 3 * 44100) * 0.05
        with torch.inference_mode():
            expected, _, _ = model.base(wet, tone_reference=reference)
            restored, uncertainty, state = model(wet, tone_reference=reference)
        torch.testing.assert_close(restored, expected, atol=1.0e-6, rtol=1.0e-6)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertEqual(manifest["downsample_factor"], 256)
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["clean_or_oracle_input"])

    def test_waveform_unet_accepts_arbitrary_runtime_length(self):
        model = WetToneComplexUNetAmpCabInverse(12, 4, 16).eval()
        wet = torch.randn(1, 8201) * 0.05
        reference = torch.randn(1, 3 * 44100) * 0.05
        with torch.inference_mode():
            restored, uncertainty, _ = model(wet, tone_reference=reference)
        self.assertEqual(restored.shape, wet.shape)
        self.assertEqual(uncertainty.shape, wet.shape)

    def test_joint_unet_keeps_only_tone_encoder_frozen(self):
        model = WetToneComplexUNetJointAmpCabInverse(12, 4, 16)
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in model.base.tone_encoder.parameters()
        ))
        self.assertTrue(any(
            parameter.requires_grad
            for name, parameter in model.base.named_parameters()
            if not name.startswith("tone_encoder.")
        ))
        manifest = model.manifest()
        self.assertFalse(manifest["base_complex_inverse_frozen"])
        self.assertTrue(manifest["complex_decoder_trainable"])

    def test_joint_demucs_is_order_free_and_starts_as_complex_base(self):
        model = WetToneComplexDemucsJointAmpCabInverse(16, 6, 16).eval()
        wet = torch.randn(1, 8201) * 0.05
        reference = torch.randn(1, 3 * 44100) * 0.05
        with torch.inference_mode():
            expected, _, _ = model.base(wet, tone_reference=reference)
            restored, uncertainty, state = model(wet, tone_reference=reference)
        torch.testing.assert_close(restored, expected, atol=1.0e-6, rtol=1.0e-6)
        self.assertEqual(restored.shape, wet.shape)
        self.assertEqual(uncertainty.shape, wet.shape)
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertEqual(manifest["schema"], 10)
        self.assertGreater(manifest["bottleneck_receptive_field_seconds"], 1.0)
        self.assertFalse(manifest["base_complex_inverse_frozen"])
        for key in (
            "profile_id_input", "gain_category_input", "graph_order_input",
            "neighbor_effect_input", "recurrent_state_input", "clean_or_oracle_input",
        ):
            self.assertFalse(manifest[key])

    def test_wet_graybox_starts_safe_and_has_explicit_reverse_stages(self):
        model = WetToneGrayBoxAmpCabInverse(16, 6, 16).eval()
        wet = torch.randn(2, 8192) * 0.05
        reference = torch.randn(2, 3 * 44100) * 0.05
        with torch.inference_mode():
            restored, uncertainty, state = model(wet, tone_reference=reference)
        torch.testing.assert_close(restored, wet * 0.1, atol=1.0e-6, rtol=1.0e-6)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertEqual(manifest["schema"], 11)
        self.assertFalse(manifest["opaque_temporal_network"])
        self.assertEqual(len(manifest["internal_reverse_stages"]), 5)
        for key in (
            "profile_id_input", "gain_category_input", "graph_order_input",
            "neighbor_effect_input", "recurrent_state_input", "clean_or_oracle_input",
        ):
            self.assertFalse(manifest[key])

    def test_phase_graybox_starts_safe_with_non_symmetric_lti_stages(self):
        model = WetTonePhaseGrayBoxAmpCabInverse(16, 6, 16).eval()
        wet = torch.randn(1, 8201) * 0.05
        reference = torch.randn(1, 3 * 44100) * 0.05
        with torch.inference_mode():
            restored, uncertainty, state = model(wet, tone_reference=reference)
        torch.testing.assert_close(restored, wet * 0.1, atol=1.0e-6, rtol=1.0e-6)
        self.assertEqual(restored.shape, wet.shape)
        self.assertEqual(uncertainty.shape, wet.shape)
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertEqual(manifest["schema"], 12)
        self.assertEqual(manifest["arbitrary_phase_fir_taps"], [513, 257, 129])
        self.assertFalse(manifest["graph_order_input"])

    def test_graybox_diagnostic_condition_matches_production_condition(self):
        model = WetTonePhaseGrayBoxAmpCabInverse(16, 6, 16).eval()
        wet = torch.randn(2, 8192) * 0.05
        reference = torch.randn(2, 3 * 44100) * 0.05
        with torch.inference_mode():
            tone = model.tone_encoder(reference)
            condition = model.condition(tone)
            expected, _, _ = model(wet, tone_reference=reference)
            actual = model.forward_with_condition(wet, condition)
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            model.forward_with_condition(wet, torch.zeros(2, 15))

    def test_phase_graybox_dynamics_starts_as_frozen_base(self):
        model = WetTonePhaseGrayBoxDynamicsAmpCabInverse(16, 6, 16).eval()
        wet = torch.randn(1, 8192) * 0.05
        reference = torch.randn(1, 3 * 44100) * 0.05
        with torch.inference_mode():
            expected, _, _ = model.base(wet, tone_reference=reference)
            restored, uncertainty, state = model(wet, tone_reference=reference)
        torch.testing.assert_close(restored, expected, atol=1.0e-6, rtol=1.0e-6)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertEqual(manifest["schema"], 13)
        self.assertTrue(manifest["base_phase_gray_box_frozen"])
        self.assertFalse(manifest["graph_order_input"])

    def test_multiband_graybox_dynamics_starts_as_frozen_base(self):
        model = WetTonePhaseGrayBoxMultibandDynamicsAmpCabInverse(16, 6, 16).eval()
        wet = torch.randn(1, 8192) * 0.05
        reference = torch.randn(1, 3 * 44100) * 0.05
        with torch.inference_mode():
            expected, _, _ = model.base(wet, tone_reference=reference)
            restored, uncertainty, state = model(wet, tone_reference=reference)
        torch.testing.assert_close(restored, expected, atol=2.0e-5, rtol=2.0e-4)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertEqual(manifest["schema"], 14)
        self.assertEqual(len(manifest["band_edges_hz"]), 5)
        self.assertFalse(manifest["graph_order_input"])

    def test_product_stagewise_graybox_is_fresh_and_order_free(self):
        model = OpenRiffBoxStagewiseGrayBoxInverse(16, 6, 16).eval()
        wet = torch.randn(1, 8192) * 0.05
        reference = torch.randn(1, 3 * 44100) * 0.05
        with torch.inference_mode():
            restored, uncertainty, state = model(wet, tone_reference=reference)
        self.assertEqual(restored.shape, wet.shape)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertEqual(manifest["schema"], 15)
        self.assertFalse(manifest["research_probe"])
        self.assertFalse(manifest["research_pretraining_weights_reusable"])
        self.assertFalse(manifest["graph_order_input"])

    def test_transient_stagewise_graybox_records_loss_contract(self):
        model = OpenRiffBoxStagewiseTransientGrayBoxInverse(16, 6, 16)
        manifest = model.manifest()
        self.assertEqual(manifest["schema"], 16)
        self.assertIn("attack", manifest["product_loss_policy"])
        self.assertFalse(manifest["graph_order_input"])

    def test_rusty_amp_stagewise_graybox_is_wet_only_and_order_free(self):
        model = RustyAmpStagewiseGrayBoxInverse(24, 8, 64).eval()
        wet = torch.randn(1, 8192) * 0.05
        reference = torch.randn(1, 3 * 44100) * 0.05
        controls_a = torch.zeros(1, 4)
        controls_b = torch.ones(1, 4)
        with torch.inference_mode():
            restored_a, uncertainty, state = model(
                wet, controls_a, tone_reference=reference
            )
            restored_b, _, _ = model(wet, controls_b, tone_reference=reference)
        torch.testing.assert_close(restored_a, restored_b)
        self.assertEqual(restored_a.shape, wet.shape)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertEqual(manifest["schema"], 17)
        self.assertEqual(len(manifest["internal_reverse_stages"]), 8)
        for key in (
            "profile_id_input", "gain_category_input", "graph_order_input",
            "neighbor_effect_input", "recurrent_state_input", "clean_or_oracle_input",
        ):
            self.assertFalse(manifest[key])

    def test_fresh_validation_contract_is_disjoint(self):
        contract = json.loads((Path(__file__).with_name("egdb_pg_subset_v1.json")).read_text())
        track_sets = [set(value) for value in contract["tracks"].values()]
        for index, left in enumerate(track_sets):
            for right in track_sets[index + 1:]:
                self.assertFalse(left & right)
        fresh = set(contract["profiles"]["fresh_validation_not_downloaded"].values())
        others = set()
        for name, group in contract["profiles"].items():
            if name == "fresh_validation_not_downloaded":
                continue
            for value in group.values():
                others.update(value if isinstance(value, list) else [value])
        self.assertFalse(fresh & others)


if __name__ == "__main__":
    unittest.main()
