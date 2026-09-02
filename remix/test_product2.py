import unittest
from pathlib import Path

import numpy as np
import torch

from .inverse2 import InverseExpert
from .product2 import HISTORY_FRAMES, ProductPairsV2, restore_dynamics, restore_echo
from .quality2 import reduction


ROOT = Path(__file__).resolve().parents[1]


class Product2Tests(unittest.TestCase):
    def test_realized_sources_are_balanced_and_authorized(self):
        dataset = ProductPairsV2(ROOT, "nonlinear", "fit", 6, 4096, 77)
        self.assertEqual(dataset.realized_source_counts(), {
            "dafx25-guitar-effects-chains": 2,
            "egfxset": 2,
            "guitarjam": 2,
        })
        self.assertTrue(dataset.authorization["authorized"])

    def test_dynamics_exposes_exact_inverse_gain_target(self):
        dataset = ProductPairsV2(ROOT, "dynamics", "development", 2, 4096, 88)
        row = dataset[0]
        reconstructed = row["wet"] * torch.exp(row["inverse_log_gain"])
        torch.testing.assert_close(reconstructed, row["clean"], atol=2.0e-6, rtol=2.0e-6)
        restored, envelope = restore_dynamics(row["wet"].numpy(), row["control_values"])
        np.testing.assert_allclose(restored, row["clean"].numpy(), atol=3.0e-6, rtol=3.0e-6)
        self.assertGreaterEqual(envelope, 0.0)

    def test_echo_pair_contains_echo_and_analytic_inverse_is_exact(self):
        dataset = ProductPairsV2(ROOT, "echo", "development", 2, 4096, 99)
        row = dataset[0]
        restored = restore_echo(row["wet"].numpy(), row["control_values"])
        start = row["target_start"]
        np.testing.assert_allclose(restored[start:], row["clean"].numpy()[start:], atol=2.0e-6, rtol=2.0e-6)

    def test_old_style_gate_semantics_are_preserved_and_negative_is_bounded(self):
        self.assertAlmostEqual(reduction(1.0, 0.85)["reduction"], 0.15)
        self.assertEqual(reduction(1.0e-12, 1.0)["reduction"], -1.0)

    def test_inverse_experts_are_independent_and_report_uncertainty(self):
        nonlinear = InverseExpert("nonlinear", hidden_size=8, layers=1).eval()
        dynamics = InverseExpert("dynamics", hidden_size=8, layers=1).eval()
        self.assertNotEqual(nonlinear.manifest()["architecture"], dynamics.manifest()["architecture"])
        self.assertFalse(nonlinear.manifest()["causal"])
        self.assertTrue(dynamics.manifest()["causal"])
        prefix = torch.randn(1, 1024) * 0.05
        controls = torch.zeros(1, 5)
        with torch.inference_mode():
            restored, uncertainty, state = dynamics(prefix, controls)
        self.assertEqual(restored.shape, prefix.shape)
        self.assertEqual(uncertainty.shape, prefix.shape)
        self.assertIsNotNone(state)
        self.assertTrue(torch.all(uncertainty > 0))


if __name__ == "__main__":
    unittest.main()
