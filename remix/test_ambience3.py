import unittest
from pathlib import Path

import numpy as np
import torch

from .ambience3 import (
    AmbiencePairsV3,
    profile_inverse_base,
    regularized_profile_inverse,
    wet_candidate_tail_ratio,
)
from .ambience_model3 import AmbienceProfileExpert


ROOT = Path(__file__).resolve().parents[1]


class Ambience3Tests(unittest.TestCase):
    def test_exact_profile_path_is_preserved(self):
        clean = np.linspace(-0.2, 0.2, 4096, dtype=np.float32)
        impulse = np.asarray([1.0, 0.0], dtype=np.float32)
        controls = {"mix": 0.2, "room_gain_db": -12.0}
        wet = clean * (0.8 + 0.2 * 10.0 ** (-12.0 / 20.0))
        restored, report = profile_inverse_base(wet, impulse, controls)
        np.testing.assert_allclose(restored, clean, atol=2.0e-6, rtol=2.0e-6)
        self.assertFalse(report["fallback"])

    def test_regularized_profile_inverse_is_finite_and_bounded(self):
        wet = np.zeros(8192, dtype=np.float32)
        wet[1024] = 0.2
        impulse = np.exp(-np.arange(4096) / 900.0).astype(np.float32)
        impulse /= np.linalg.norm(impulse)
        restored, report = regularized_profile_inverse(
            wet, impulse, {"mix": 0.6, "room_gain_db": -4.0}
        )
        self.assertTrue(np.isfinite(restored).all())
        self.assertLessEqual(report["maximum_frequency_gain"], 6.0 + 1.0e-6)
        self.assertLessEqual(float(np.max(np.abs(restored))), 1.05)

    def test_model_never_changes_exact_profile_base(self):
        model = AmbienceProfileExpert(channels=4, depth=6).eval()
        wet = torch.randn(1, 4096) * 0.03
        base = torch.randn(1, 4096) * 0.02
        with torch.no_grad():
            model.correction.bias.fill_(1.0)
            restored, uncertainty = model(wet, torch.zeros(1, 3), base, torch.zeros(1, dtype=torch.bool))
        torch.testing.assert_close(restored, base, atol=0.0, rtol=0.0)
        self.assertEqual(uncertainty.shape, wet.shape)
        self.assertFalse(model.manifest()["graph_order_input"])

    def test_dataset_exposes_profile_mode_without_locked_final(self):
        dataset = AmbiencePairsV3(ROOT, "calibration", 2, 16384, 71)
        row = dataset[0]
        self.assertEqual(row["profile_base"].shape, row["wet"].shape)
        self.assertIn(row["profile_report"]["mode"], {"exact", "regularized-fallback"})
        self.assertNotEqual(dataset.split, "locked-final")

    def test_observable_tail_ratio_detects_added_energy(self):
        wet = np.zeros(8192, dtype=np.float32)
        wet[1000:1600] = 0.2
        candidate = wet.copy()
        candidate[1600:2600] = 0.1
        self.assertGreater(wet_candidate_tail_ratio(wet, candidate, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
