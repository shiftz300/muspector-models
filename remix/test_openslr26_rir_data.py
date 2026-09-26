import unittest

from .openslr26_rir_data import (
    EXPECTED_CATEGORIES,
    EXPECTED_POSITIONS,
    EXPECTED_ROOMS,
    _expected_relative_paths,
)


class OpenSLR26RIRDataTest(unittest.TestCase):
    def test_frozen_subset_is_balanced_and_path_safe(self):
        paths = _expected_relative_paths()
        self.assertEqual(
            len(paths),
            len(EXPECTED_CATEGORIES) * len(EXPECTED_ROOMS) * len(EXPECTED_POSITIONS),
        )
        self.assertTrue(all(not path.is_absolute() for path in paths))
        self.assertTrue(all(".." not in path.parts for path in paths))
        self.assertEqual({path.parts[1] for path in paths}, set(EXPECTED_CATEGORIES))

    def test_expected_room_range_was_frozen_before_signal_audit(self):
        self.assertEqual(EXPECTED_ROOMS[0], "Room191")
        self.assertEqual(EXPECTED_ROOMS[-1], "Room200")
        self.assertEqual(EXPECTED_POSITIONS, ("00001", "00025", "00050", "00100"))


if __name__ == "__main__":
    unittest.main()
