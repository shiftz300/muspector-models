import json
import unittest
from pathlib import Path

import torch

from .rusty_amp_stage_model import RustyAmpStagewiseGrayBoxInverse


class RustyAmpStageTests(unittest.TestCase):
    def test_model_is_eight_stage_wet_only_and_order_free(self):
        model = RustyAmpStagewiseGrayBoxInverse(16, 6, 16).eval()
        wet = torch.randn(1, 8192) * 0.05
        with torch.inference_mode():
            stages = model.stage_outputs(wet)
            restored, uncertainty, state = model(wet)
        self.assertEqual(len(stages), 8)
        self.assertEqual(restored.shape, wet.shape)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertEqual(manifest["schema"], 17)
        self.assertTrue(manifest["deep_supervision_pretraining"])
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbor_effect_input"])

    def test_registry_pins_renderer_and_excludes_cabinet(self):
        registry = json.loads(Path(__file__).with_name("data_sources.json").read_text())
        source = next(row for row in registry["sources"] if row["id"] == "rusty-amp-stage-renderer")
        self.assertEqual(source["artifact_audit"]["source_commit"], "831d6bba4ad3a1a8da958c510cbc96a9e55b73f9")
        self.assertIn("no-cabinet", source["audio_scope"])


if __name__ == "__main__":
    unittest.main()
