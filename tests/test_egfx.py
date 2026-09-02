import unittest

from remix.egfx import partition


class EgfxTests(unittest.TestCase):
    def test_all_pickups_of_one_performance_share_split(self):
        self.assertEqual(partition("2-13"), partition("2-13"))
        self.assertIn(partition("2-13"), {"fit", "calibration", "development"})


if __name__ == "__main__": unittest.main()
