import unittest

import torch

from .blind2 import (
    FRAMES,
    LABELS,
    MELS,
    WINDOW,
    BlindFamilyPresence,
    BlindPresence,
    BlindPresenceMultiAxis,
    aggregate_windows,
    log_mel,
)


class Blind2Test(unittest.TestCase):
    def test_frontend_and_model_contract(self) -> None:
        timeline = torch.arange(WINDOW) / 44_100.0
        audio = (0.2 * torch.sin(2.0 * torch.pi * 440.0 * timeline))[None]
        feature = log_mel(audio)
        self.assertEqual(tuple(feature.shape), (1, 1, MELS, FRAMES))
        self.assertTrue(torch.isfinite(feature).all())
        self.assertLess(abs(float(feature.mean())), 1.0e-4)
        self.assertLess(abs(float(feature.std(correction=1)) - 1.0), 1.0e-4)
        model = BlindPresence()
        self.assertEqual(tuple(model.features(feature).shape), (1, 256))
        self.assertEqual(tuple(model(feature).shape), (1, len(LABELS)))
        self.assertFalse(model.manifest()["order_output"])
        self.assertFalse(model.manifest()["controls_output"])
        multiaxis = BlindPresenceMultiAxis()
        self.assertEqual(tuple(multiaxis.features(feature).shape), (1, 768))
        self.assertEqual(tuple(multiaxis(feature).shape), (1, len(LABELS)))
        self.assertEqual(multiaxis.manifest()["architecture"], "multiaxis-audio-resnet18")
        family = BlindFamilyPresence("nonlinear")
        self.assertEqual(tuple(family(feature).shape), (1, 1))
        self.assertEqual(family.manifest()["family"], "nonlinear")
        self.assertEqual(tuple(BlindFamilyPresence("any")(feature).shape), (1, 1))
        with self.assertRaises(ValueError):
            BlindFamilyPresence("order")

    def test_top_two_aggregation_rejects_one_spike(self) -> None:
        values = torch.tensor([
            [0.02, 0.03, 0.04, 0.05],
            [0.91, 0.04, 0.05, 0.06],
            [0.03, 0.05, 0.06, 0.07],
        ])
        score = aggregate_windows(values)
        self.assertAlmostEqual(float(score[0]), 0.47, places=5)


if __name__ == "__main__":
    unittest.main()
