import tempfile
import unittest
from pathlib import Path

from .but_reverb_data import EXPECTED_ROOM_SPLITS, SPLITS, rir_splits


class ButReverbDataTests(unittest.TestCase):
    def test_rooms_are_wholly_split_and_locked_final_is_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for room, split in EXPECTED_ROOM_SPLITS.items():
                for mic in ("MicID01", "MicID02"):
                    path = root / room / mic / "SpkID01_20180101_S" / "01" / "RIR" / "RIR.v00.wav"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
            result = rir_splits(root)
            self.assertEqual(set(result), set(SPLITS))
            observed = {
                split: {path.relative_to(root).parts[0] for path in paths}
                for split, paths in result.items()
            }
            for room, split in EXPECTED_ROOM_SPLITS.items():
                self.assertIn(room, observed[split])
                self.assertTrue(all(room not in observed[other] for other in SPLITS if other != split))
            self.assertEqual(observed["locked-final"], {"VUT_FIT_D105", "VUT_FIT_C236"})
            for split in ("fit", "development", "locked-final"):
                early_rooms = {
                    path.relative_to(root).parts[0]
                    for path in result[split][:
                        len(observed[split])
                    ]
                }
                self.assertEqual(early_rooms, observed[split])

    def test_unexpected_room_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for room in (*EXPECTED_ROOM_SPLITS, "Unknown_Room"):
                path = root / room / "MicID01" / "SpkID01_20180101_S" / "01" / "RIR" / "RIR.v00.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with self.assertRaisesRegex(ValueError, "room inventory differs"):
                rir_splits(root)


if __name__ == "__main__":
    unittest.main()
