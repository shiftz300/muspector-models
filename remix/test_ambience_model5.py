import unittest

import numpy as np
import torch

from .ambience5_profile_direct import PROFILE_CHANNELS, profile_response_features
from .ambience_model5 import AmbienceProfileDirectExpert


class AmbienceProfileDirectTest(unittest.TestCase):
    def test_profile_encoding_keeps_long_rir_and_order_contract(self):
        impulse = np.zeros(4_000, dtype=np.float32)
        impulse[0] = 1.0
        impulse[1_500] = 0.2
        features, transfer = profile_response_features(
            impulse, {"mix": 0.4, "room_gain_db": -8.0}
        )
        self.assertEqual(features.shape, (PROFILE_CHANNELS, 513))
        self.assertEqual(len(transfer), len(impulse))
        self.assertTrue(np.isfinite(features).all())
        model = AmbienceProfileDirectExpert(channels=6, depth=6)
        manifest = model.manifest()
        self.assertFalse(manifest["chain_order_input"])
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbor_effect_input"])
        self.assertFalse(manifest["clean_input_at_runtime"])

    def test_identity_initialization_and_exact_override(self):
        torch.manual_seed(7)
        model = AmbienceProfileDirectExpert(channels=6, depth=6)
        wet = torch.randn(2, 4_096) * 0.02
        controls = torch.full((2, 3), 0.5)
        profile = torch.zeros(2, PROFILE_CHANNELS, 513)
        exact = wet * 0.75
        analytic = wet * 0.9
        restored, uncertainty = model(
            wet, controls, profile, analytic, exact, torch.tensor([False, True])
        )
        self.assertTrue(torch.allclose(restored[0], analytic[0], atol=1.0e-6))
        self.assertTrue(torch.equal(restored[1], exact[1]))
        self.assertEqual(uncertainty.shape, wet.shape)

    def test_physical_replay_is_exact_for_delta(self):
        audio = torch.randn(1, 2_048) * 0.02
        transfer = torch.zeros(1, 37)
        transfer[:, 0] = 1.0
        replay = AmbienceProfileDirectExpert._replay(audio, transfer)
        self.assertTrue(torch.allclose(replay, audio, atol=1.0e-6))


if __name__ == "__main__":
    unittest.main()
