import random
import unittest
from pathlib import Path

import numpy as np
import torch

from .quality2 import summarize
from .spectral3 import SpectralPairsV3, normalized_controls, spectral_eq
from .spectral_model3 import MAXIMUM_INVERSE_GAIN_DB, SpectralInverseV3, bounded_inverse_base


class Spectral3Tests(unittest.TestCase):
    def test_analytic_inverse_recovers_repository_eq(self):
        rng = random.Random(17)
        clean = np.random.default_rng(19).standard_normal(16384).astype(np.float32) * 0.06
        wet, values = spectral_eq(clean, rng)
        controls = torch.from_numpy(normalized_controls(values)).unsqueeze(0)
        with torch.inference_mode():
            restored = bounded_inverse_base(torch.from_numpy(wet).unsqueeze(0), controls)[0].numpy()
        baseline = float(np.mean((wet - clean) ** 2))
        candidate = float(np.mean((restored - clean) ** 2))
        self.assertLess(candidate, baseline * 1.0e-5)

    def test_quality_gate_accepts_exact_spectral_restoration(self):
        rng = random.Random(23)
        clean = np.random.default_rng(29).standard_normal(16384).astype(np.float32) * 0.04
        wet, values = spectral_eq(clean, rng)
        controls = torch.from_numpy(normalized_controls(values)).unsqueeze(0)
        with torch.inference_mode():
            restored = bounded_inverse_base(torch.from_numpy(wet).unsqueeze(0), controls)[0].numpy()
        report = summarize("spectral", [wet], [restored], [clean])
        self.assertTrue(report["accepted"], report)

    def test_expert_has_no_order_or_neighbour_inputs(self):
        model = SpectralInverseV3(channels=6, depth=2).eval()
        wet = torch.randn(2, 4096) * 0.05
        controls = torch.rand(2, 5)
        with torch.inference_mode():
            restored, uncertainty = model(wet, controls)
        self.assertEqual(restored.shape, wet.shape)
        self.assertEqual(uncertainty.shape, wet.shape)
        manifest = model.manifest()
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbouring_effect_input"])
        self.assertEqual(manifest["maximum_inverse_gain_db"], MAXIMUM_INVERSE_GAIN_DB)

    def test_product_pairs_do_not_open_locked_final(self):
        workspace = Path(__file__).resolve().parents[1]
        dataset = SpectralPairsV3(workspace, "development", 2, 4096, 31)
        row = dataset[0]
        self.assertEqual(row["target_end"] - row["target_start"], 4096)
        self.assertNotIn("locked-final", row["group"])
        self.assertNotIn(row["source_id"], {"asrnn-physical-effects", "tonetwist-local-collection"})


if __name__ == "__main__":
    unittest.main()
