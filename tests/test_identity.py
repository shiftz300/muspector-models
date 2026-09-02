import unittest
from unittest.mock import Mock

import numpy as np

from remix.identity import FRAMES, Router, audio, starts


class IdentityTests(unittest.TestCase):
    def test_analysis_copy_resamples_without_mutating_source(self):
        source = np.linspace(-0.5, 0.5, 44_100, dtype=np.float32)
        before = source.copy()
        result = audio(source, 44_100)
        self.assertEqual(result.shape, (48_000,))
        np.testing.assert_array_equal(source, before)

    def test_window_selection_is_bounded_and_prefers_signal(self):
        source = np.zeros(FRAMES * 3, dtype=np.float32)
        source[FRAMES * 2 :] = 0.5
        selected = starts(source)
        self.assertLessEqual(len(selected), 3)
        self.assertIn(FRAMES * 2, selected)

    def test_router_stays_shadow_after_rat_candidate(self):
        identity = Mock()
        identity.infer_pair.return_value = {"decision": "candidate", "label": "RAT"}
        result = Router(identity).infer(
            {"decision": "accepted", "active": ["drive"]},
            np.ones(48_000, dtype=np.float32),
            np.ones(48_000, dtype=np.float32),
            48_000,
        )
        self.assertEqual(result["device"], "rat")
        self.assertEqual(result["decision"], "candidate")
        self.assertFalse(result["automatic_delivery"])

    def test_router_does_not_run_identity_for_ineligible_chain(self):
        identity = Mock()
        result = Router(identity).infer(
            {"decision": "bypass", "active": []},
            np.ones(48_000, dtype=np.float32),
            np.ones(48_000, dtype=np.float32),
            48_000,
        )
        self.assertEqual(result["decision"], "abstain")
        identity.infer_pair.assert_not_called()


if __name__ == "__main__":
    unittest.main()
