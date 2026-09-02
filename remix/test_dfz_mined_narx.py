"""Controlled-intervention checks independent of the full fit miner tests."""
import unittest

import numpy as np

from .train_dfz_mined_narx import choose_event, expected_generation


class FakeFitMiner:
    def __init__(self):
        self.calls = 0

    def next_event(self, path):
        if path != "fit.wav":
            raise ValueError("not a fit path")
        self.calls += 1
        return (777, "positive") if self.calls % 2 else (888, "negative")


class MinedTrainerTests(unittest.TestCase):
    def test_exact_snapshot_generation_and_no_post4000_training(self):
        for step, generation in ((1, 0), (250, 0), (251, 250), (500, 250), (3999, 3750), (4000, 3750)):
            self.assertEqual(expected_generation(step), generation)
        for step in (0, 4001, -1):
            with self.assertRaises(ValueError):
                expected_generation(step)

    def test_only_core_event_changes_and_non_core_random_stream_is_preserved(self):
        row = {"path": "fit.wav", "wet_events": (100, 200), "core_events": (300, 400)}
        previous, current = np.random.default_rng(988), np.random.default_rng(988)
        miner = FakeFitMiner()
        for _ in range(100):
            for kind in ("wet_events", "core_events"):
                old_event = row[kind][int(previous.integers(len(row[kind])))]
                event, sign = choose_event(row, kind, current, miner)
                if kind == "wet_events":
                    self.assertEqual(event, old_event)
                    self.assertEqual(sign, "fixed-wet")
                else:
                    self.assertIn(event, (777, 888))
                    self.assertIn(sign, ("positive", "negative"))
                self.assertEqual(int(previous.integers(96, 161)), int(current.integers(96, 161)))
        np.testing.assert_array_equal(previous.random(16), current.random(16))
        self.assertEqual(miner.calls, 100)

    def test_current_events_cannot_bypass_fit_miner_path_validation(self):
        row = {"path": "cal.wav", "core_events": (300, 400)}
        with self.assertRaises(ValueError):
            choose_event(row, "core_events", np.random.default_rng(988), FakeFitMiner())


if __name__ == "__main__":
    unittest.main()
