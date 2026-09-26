import unittest
from pathlib import Path

import torch

from .open_riff_box_stage_data import PINNED_COMMIT, render_command, settings
from .open_riff_box_stage_model import OpenRiffBoxStagewiseGrayBoxInverse


class OpenRiffBoxStageTests(unittest.TestCase):
    def test_renderer_command_is_pinned_asset_free_and_deterministic(self):
        row = settings(7, 1)[0]
        command = render_command(Path("renderer"), Path("clean.wav"), Path("tap.wav"), row, 9)
        self.assertEqual(PINNED_COMMIT, "c980ae874c87e16835d87b265ea58079fc69e7f5")
        self.assertEqual(command[command.index("--cabinet") + 1], "200")
        self.assertEqual(command[command.index("--plat-diag") + 1], "noiselevel=0")
        self.assertEqual(command[command.index("--stage-limit") + 1], "9")

    def test_stagewise_model_is_wet_only_order_free(self):
        model = OpenRiffBoxStagewiseGrayBoxInverse(condition_size=16, segments=8)
        wet = torch.randn(1, 4096) * 0.05
        outputs = model.stage_outputs(wet)
        self.assertEqual(len(outputs), 9)
        self.assertTrue(all(value.shape == wet.shape for value in outputs))
        restored, uncertainty, state = model(wet)
        self.assertEqual(restored.shape, wet.shape)
        self.assertEqual(uncertainty.shape, wet.shape)
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertTrue(manifest["wet_only_inference"])
        for key in (
            "profile_id_input", "gain_category_input", "graph_order_input",
            "neighbor_effect_input", "recurrent_state_input", "clean_or_oracle_input",
        ):
            self.assertFalse(manifest[key])
        self.assertFalse(manifest["research_pretraining_weights_reusable"])


if __name__ == "__main__":
    unittest.main()
