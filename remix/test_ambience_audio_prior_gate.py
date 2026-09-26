import unittest

import torch

from .ambience_audio_prior_gate import (
    FEATURE_BINS,
    FEATURE_FRAMES,
    ReverbCandidateAudioPriorGate,
    candidate_audio_features,
)
from .ambience5_profile_direct import PROFILE_CHANNELS, PROFILE_N_FFT


class AmbienceAudioPriorGateTests(unittest.TestCase):
    def test_features_and_gate_exclude_clean_and_order(self):
        wet = torch.randn(65536) * 0.05
        candidate = wet * 0.8
        profile = torch.randn(PROFILE_CHANNELS, PROFILE_N_FFT // 2 + 1)
        features = candidate_audio_features(wet, candidate, profile)
        self.assertEqual(
            features.shape,
            (4 + PROFILE_CHANNELS, FEATURE_BINS, FEATURE_FRAMES),
        )
        model = ReverbCandidateAudioPriorGate(8).eval()
        with torch.inference_mode():
            score = model(features[None], torch.zeros(1, 3), torch.zeros(1, 1))
        self.assertEqual(score.shape, (1,))
        manifest = model.manifest()
        self.assertFalse(manifest["clean_input_at_inference"])
        self.assertFalse(manifest["chain_order_input"])
        self.assertFalse(manifest["neighbor_effect_input"])

    def test_feature_geometry_is_fail_closed(self):
        with self.assertRaises(ValueError):
            candidate_audio_features(
                torch.zeros(4096), torch.zeros(4095),
                torch.zeros(PROFILE_CHANNELS, PROFILE_N_FFT // 2 + 1),
            )


if __name__ == "__main__":
    unittest.main()
