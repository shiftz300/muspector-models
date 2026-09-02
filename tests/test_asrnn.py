import unittest
from pathlib import Path

from remix.asrnn import spread


class AsrnnTests(unittest.TestCase):
    def test_spread_is_deterministic_and_includes_boundaries(self):
        paths = [Path(f"{index:02}.wav") for index in range(10)]
        self.assertEqual(spread(paths, 3), [paths[0], paths[4], paths[9]])


if __name__ == "__main__":
    unittest.main()
