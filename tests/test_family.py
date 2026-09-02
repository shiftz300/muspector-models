import unittest

import numpy as np

from remix.family import FRAMES, _margin, frontend


class FamilyTests(unittest.TestCase):
    def test_frontend_preserves_audio_and_geometry(self):
        time = np.arange(FRAMES, dtype=np.float32) / 44_100.0
        audio = (0.1 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)
        before = audio.copy()
        mel = frontend(audio)
        self.assertEqual(mel.shape, (1, 1, 128, 216))
        self.assertTrue(np.isfinite(mel).all())
        np.testing.assert_array_equal(audio, before)

    def test_margin_is_distance_inside_decision(self):
        self.assertAlmostEqual(_margin(True, 0.8, 0.6), 0.2)
        self.assertAlmostEqual(_margin(False, 0.2, 0.6), 0.4)

    def test_frontend_rejects_wrong_geometry(self):
        with self.assertRaisesRegex(ValueError, "expected"):
            frontend(np.zeros(100, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
