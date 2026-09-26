import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile

from .openair_rir_data import ROOM_SPECS, load_openair_rir


class OpenAIRRIRDataTest(unittest.TestCase):
    def test_loader_uses_w_channel_resamples_and_normalizes(self):
        with tempfile.TemporaryDirectory() as directory:
            room = "creswell-crags"
            path = Path(directory) / room / ROOM_SPECS[room]["branch"] / "test.wav"
            path.parent.mkdir(parents=True)
            value = np.zeros((192_200, 4), dtype=np.float32)
            value[100, 0] = 1.0
            value[100, 1] = 100.0
            soundfile.write(path, value, 96_000, subtype="FLOAT")
            impulse = load_openair_rir(path)
            self.assertEqual(impulse.shape, (48_000,))
            self.assertAlmostEqual(float(np.sum(np.square(impulse))), 1.0)
            self.assertGreater(float(impulse[0]), 0.9)


if __name__ == "__main__":
    unittest.main()
