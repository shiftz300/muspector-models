import unittest

import numpy as np

from .recalibrate_blind2_family import (
    _select_blend,
    _select_family_fallback,
    _select_or_gate,
    _select_threshold,
    _wilson_upper,
)


class RecalibrateBlind2FamilyTest(unittest.TestCase):
    def test_threshold_must_satisfy_internal_and_external_clean_gates(self) -> None:
        internal_probabilities = np.asarray([[0.1], [0.2], [0.8], [0.9]])
        internal_targets = np.asarray([[0.0], [0.0], [1.0], [1.0]])
        external_probabilities = np.concatenate(
            (np.full((100, 1), 0.4), np.asarray([[0.7], [0.8]]))
        )
        external_targets = np.concatenate((np.zeros((100, 1)), np.ones((2, 1))))
        threshold, metrics = _select_threshold(
            internal_probabilities,
            internal_targets,
            external_probabilities,
            external_targets,
            np.asarray([True] * 100 + [False, False]),
        )
        self.assertGreater(threshold, 0.4)
        self.assertLessEqual(metrics["internal_negative_false_positive_rate"], 0.05)
        self.assertLessEqual(metrics["external_clean_false_positive_rate"], 0.05)
        self.assertLessEqual(
            metrics["external_clean_false_positive_wilson_95_upper"], 0.05
        )

    def test_wilson_gate_requires_zero_errors_for_ninety_six_clean_examples(self) -> None:
        self.assertLess(_wilson_upper(0, 96), 0.05)
        self.assertGreater(_wilson_upper(1, 96), 0.05)

    def test_blend_search_can_prefer_complementary_base_and_expert(self) -> None:
        targets = np.asarray([[0.0], [0.0], [1.0], [1.0]])
        base = np.asarray([[0.1], [0.2], [0.55], [0.6]])
        expert = np.asarray([[0.7], [0.8], [0.9], [0.95]])
        external_base = np.concatenate(
            (np.full((100, 1), 0.1), np.asarray([[0.55], [0.6]]))
        )
        external_expert = np.concatenate(
            (np.full((100, 1), 0.7), np.asarray([[0.9], [0.95]]))
        )
        external_targets = np.concatenate((np.zeros((100, 1)), np.ones((2, 1))))
        weight, threshold, metrics = _select_blend(
            base,
            expert,
            targets,
            external_base,
            external_expert,
            external_targets,
            np.asarray([True] * 100 + [False, False]),
        )
        self.assertGreaterEqual(weight, 0.0)
        self.assertLessEqual(weight, 1.0)
        self.assertGreater(threshold, 0.0)
        self.assertEqual(metrics["internal_recall"], 1.0)

    def test_or_gate_can_use_base_as_false_negative_fallback(self) -> None:
        internal_targets = np.asarray([[0.0], [0.0], [1.0], [1.0]])
        internal_gate = np.asarray([[0.1], [0.2], [0.9], [0.3]])
        internal_base = np.asarray([[0.1], [0.2], [0.4], [0.95]])
        external_targets = np.concatenate((np.zeros((100, 1)), np.ones((2, 1))))
        external_gate = np.concatenate(
            (np.full((100, 1), 0.1), np.asarray([[0.9], [0.3]]))
        )
        external_base = np.concatenate(
            (np.full((100, 1), 0.1), np.asarray([[0.4], [0.95]]))
        )
        _, fallback, metrics = _select_or_gate(
            internal_base,
            internal_gate,
            internal_targets,
            external_base,
            external_gate,
            external_targets,
            np.asarray([True] * 100 + [False, False]),
        )
        self.assertLess(fallback, 1.0)
        self.assertEqual(metrics["internal_recall"], 1.0)
        self.assertEqual(metrics["external_recall"], 1.0)

    def test_family_fallback_can_rescue_without_opening_clean_gate(self) -> None:
        internal_active = np.asarray([False, False, True, False])
        internal_targets = np.asarray([[0.0], [0.0], [1.0], [1.0]])
        internal_family = np.asarray([
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.95],
        ])
        external_active = np.asarray([False] * 100 + [True, False])
        external_targets = np.concatenate((np.zeros((100, 1)), np.ones((2, 1))))
        external_family = np.full((102, 4), 0.1)
        external_family[-1, 3] = 0.95
        family, threshold, metrics = _select_family_fallback(
            internal_active,
            external_active,
            internal_family,
            external_family,
            internal_targets,
            external_targets,
            np.asarray([True] * 100 + [False, False]),
        )
        self.assertEqual(family, "unknown")
        self.assertLess(threshold, 0.95)
        self.assertEqual(metrics["internal_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
