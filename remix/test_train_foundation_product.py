import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from . import train_foundation_product as trainer


class _FakeProductPairs(torch.utils.data.Dataset):
    authorization = {
        "authorized": True,
        "sources": ["guitarjam", "muspector-dsp"],
        "blocked": [],
        "required_attribution": [],
    }

    def __init__(self, _workspace, mechanism, split, samples, frames, _seed):
        self.mechanism = mechanism
        self.split = split
        self.samples = samples
        self.frames = frames

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        clean = torch.zeros(self.frames)
        clean[index % self.frames] = 0.1
        return {
            "wet": clean.clone(),
            "clean": clean,
            "source_id": "guitarjam",
            "group": f"fake:{self.split}:{index}",
            "controls": {"kind": self.mechanism},
        }


class TrainFoundationProductTests(unittest.TestCase):
    def test_every_mechanism_card_records_product_only_provenance(self):
        quality = {"accepted": False, "gates": {}, "examples": 1}
        runtime = {"ordinary_cpu": True, "audio_callback": False, "realtime_factor": 0.01}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(trainer, "ProductPairs", _FakeProductPairs), patch.object(
                trainer, "_quality", return_value=quality
            ), patch.object(trainer, "_runtime", return_value=runtime):
                for mechanism in trainer.MECHANISMS:
                    report = trainer.train_one(
                        Path(temporary),
                        output,
                        mechanism,
                        epochs=1,
                        train_samples=1,
                        calibration_samples=1,
                        development_samples=1,
                        frames=4096,
                        batch_size=1,
                        channels=4,
                        blocks=1,
                        quick=True,
                    )
                    card = json.loads((output / mechanism / "metrics.json").read_text())
                    provenance = card["provenance"]
                    self.assertEqual(card["mechanism"], mechanism)
                    self.assertEqual(report["provenance"], provenance)
                    self.assertTrue(provenance["product_sources_only"])
                    self.assertTrue(provenance["research_data_isolated"])
                    self.assertEqual(provenance["research_source_ids"], [])
                    self.assertFalse(provenance["research_data_used_for_gradients"])
                    self.assertFalse(provenance["research_data_used_for_calibration_or_selection"])
                    self.assertEqual(provenance["source_ids"], ["guitarjam", "muspector-dsp"])


if __name__ == "__main__":
    unittest.main()
