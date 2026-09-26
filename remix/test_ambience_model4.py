import unittest

import torch

from .ambience_model4 import (
    AmbienceDecayBankExpert,
    AmbienceFrequencyProfileBankExpert,
    AmbienceGrayboxExpert,
    AmbienceProfileBankExpert,
    AmbienceSparseMaskExpert,
)


class AmbienceModel4Tests(unittest.TestCase):
    def test_graybox_is_bounded_and_order_independent(self):
        model = AmbienceGrayboxExpert(channels=4, depth=6, n_fft=512, hop=128).eval()
        wet = torch.randn(1, 16384) * 0.02
        candidate = wet * 0.8
        with torch.inference_mode():
            restored, uncertainty = model(wet, torch.rand(1, 3), candidate)
        self.assertEqual(restored.shape, wet.shape)
        self.assertEqual(uncertainty.shape, wet.shape)
        manifest = model.manifest()
        self.assertTrue(manifest["wet_identity_reachable"])
        self.assertFalse(manifest["candidate_extrapolation"])
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbor_effect_input"])

    def test_training_oracle_is_finite_and_backward_safe(self):
        model = AmbienceGrayboxExpert(channels=4, depth=6, n_fft=512, hop=128)
        clean = torch.randn(1, 16384) * 0.02
        wet = clean + torch.randn_like(clean) * 0.005
        candidate = wet * 0.9
        loss, parts = model.training_loss(
            wet, torch.rand(1, 3), candidate, clean, 0, 0.0, 2.0, 5.0
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("gate_oracle", parts)

    def test_sparse_mask_adds_frequency_context_without_order_input(self):
        model = AmbienceSparseMaskExpert(channels=4, depth=6, n_fft=512, hop=128).eval()
        manifest = model.manifest()
        self.assertTrue(manifest["cross_frequency_context"])
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbor_effect_input"])
        self.assertGreater(manifest["parameters"], AmbienceGrayboxExpert(
            channels=4, depth=6, n_fft=512, hop=128
        ).manifest()["parameters"])

    def test_decay_bank_is_convex_and_order_independent(self):
        model = AmbienceDecayBankExpert(
            channels=4, depth=6, n_fft=512, hop=128
        ).eval()
        wet = torch.randn(1, 4096) * 0.02
        bank = torch.stack([wet * scale for scale in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7)], dim=1)
        with torch.inference_mode():
            restored, uncertainty = model(wet, torch.rand(1, 3), bank)
        self.assertEqual(restored.shape, wet.shape)
        self.assertEqual(uncertainty.shape, wet.shape)
        manifest = model.manifest()
        self.assertEqual(manifest["candidate_count"], 6)
        self.assertTrue(manifest["wet_identity_reachable"])
        self.assertFalse(manifest["candidate_extrapolation"])
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbor_effect_input"])

    def test_decay_bank_oracle_loss_is_backward_safe(self):
        model = AmbienceDecayBankExpert(channels=4, depth=6, n_fft=512, hop=128)
        clean = torch.randn(1, 4096) * 0.02
        wet = clean + torch.randn_like(clean) * 0.005
        bank = torch.stack([wet * scale for scale in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7)], dim=1)
        loss, parts = model.training_loss(
            wet, torch.rand(1, 3), bank, clean, 0, 0.0, 2.0, 1.0
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("candidate_oracle", parts)

    def test_profile_bank_forces_exact_path_and_declares_profile_contract(self):
        model = AmbienceProfileBankExpert(
            channels=4, depth=6, n_fft=512, hop=128
        ).eval()
        wet = torch.randn(1, 4096) * 0.05
        exact = wet * 0.8
        bank = exact[:, None].repeat(1, model.candidate_count, 1)
        with torch.inference_mode():
            restored, uncertainty = model(wet, torch.zeros(1, 3), bank)
        self.assertTrue(torch.equal(restored, exact))
        self.assertEqual(uncertainty.shape, wet.shape)
        manifest = model.manifest()
        self.assertTrue(manifest["profile_required"])
        self.assertTrue(manifest["exact_profile_path_is_immutable"])
        self.assertFalse(manifest["chain_order_input"])
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbor_effect_input"])

    def test_frequency_profile_bank_is_bounded_and_order_independent(self):
        model = AmbienceFrequencyProfileBankExpert(
            channels=4, depth=6, n_fft=512, hop=128
        ).eval()
        wet = torch.randn(1, 4096) * 0.03
        bank = torch.stack([wet * scale for scale in (0.98, 0.95, 0.92, 0.9, 0.88, 0.85, 0.82, 0.8)], dim=1)
        with torch.inference_mode():
            restored, uncertainty = model(wet, torch.rand(1, 3), bank)
        self.assertEqual(restored.shape, wet.shape)
        self.assertEqual(uncertainty.shape, wet.shape)
        manifest = model.manifest()
        self.assertEqual(manifest["candidate_count"], 8)
        self.assertTrue(manifest["profile_required"])
        self.assertTrue(manifest["wet_identity_reachable"])
        self.assertFalse(manifest["chain_order_input"])
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbor_effect_input"])


if __name__ == "__main__":
    unittest.main()
