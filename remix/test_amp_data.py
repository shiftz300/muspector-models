import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile

from .amp_data import (
    DEVELOPMENT_ONLY_CONTROLS,
    QUARANTINED_SETTING,
    RATE,
    ROOT_RELATIVE,
    discover,
    parse_controls,
)


class AmpDataTests(unittest.TestCase):
    def test_controls_and_locked_final_identity_are_explicit(self):
        self.assertEqual(parse_controls("B5_M5_T5_G5"), (5.0, 5.0, 5.0, 5.0))
        self.assertEqual(parse_controls("B6.5_M8.5_T3.5_G5"), (6.5, 8.5, 3.5, 5.0))
        with self.assertRaises(ValueError):
            parse_controls("B11_M5_T5_G5")

    def test_discovery_pairs_only_matching_input_and_marks_unseen_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            root = workspace / ROOT_RELATIVE
            audio = np.zeros(256, dtype=np.float32)
            for setting in ("B5_M5_T5_G5", "B5_M5_T5_G8", "B6.5_M8.5_T3.5_G5"):
                directory = root / setting
                directory.mkdir(parents=True)
                soundfile.write(directory / f"{setting}-input.wav", audio, RATE)
                soundfile.write(directory / f"{setting}-speakerout.wav", audio, RATE)
            pairs = discover(workspace)
            self.assertEqual([pair.setting for pair in pairs], [
                "B5_M5_T5_G5", "B5_M5_T5_G8", "B6.5_M8.5_T3.5_G5",
            ])
            self.assertFalse(pairs[0].locked_final)
            self.assertTrue(pairs[1].development_only)
            self.assertIn(pairs[1].controls, DEVELOPMENT_ONLY_CONTROLS)
            self.assertTrue(pairs[2].locked_final)
            self.assertEqual(pairs[0].normalized_controls, (0.5, 0.5, 0.5, 0.5))

    def test_known_gain6_filename_contradiction_is_quarantined(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            root = workspace / ROOT_RELATIVE
            audio = np.zeros(256, dtype=np.float32)
            valid = root / "B5_M5_T5_G5"
            valid.mkdir(parents=True)
            soundfile.write(valid / "B5_M5_T5_G5-input.wav", audio, RATE)
            soundfile.write(valid / "B5_M5_T5_G5-speakerout.wav", audio, RATE)
            quarantined = root / QUARANTINED_SETTING
            quarantined.mkdir(parents=True)
            soundfile.write(quarantined / "B5_M5_T5_G4-input.wav", audio, RATE)
            soundfile.write(quarantined / "B5_M5_T5_G4-speakerout.wav", audio, RATE)
            pairs = discover(workspace)
            self.assertEqual([pair.setting for pair in pairs], ["B5_M5_T5_G5"])


if __name__ == "__main__":
    unittest.main()
