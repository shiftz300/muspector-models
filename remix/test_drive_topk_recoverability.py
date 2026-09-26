from __future__ import annotations

import unittest

from .audit_drive_topk_recoverability import THRESHOLDS, summarize_rows


def rows(
    *, true_control: int, all_grid: int, top_k: int, top1: int, count: int = 10
) -> list[dict]:
    return [
        {
            "true_control_passed": index < true_control,
            "all_grid_has_pass": index < all_grid,
            "top_k_has_pass": index < top_k,
            "top1_passed": index < top1,
            "first_passing_replay_rank": 1 if index < all_grid else None,
        }
        for index in range(count)
    ]


class DriveTopKRecoverabilityTests(unittest.TestCase):
    def test_thresholds_are_frozen_before_compute(self) -> None:
        self.assertEqual(THRESHOLDS, {
            "true_control_pass_fraction_minimum": 0.80,
            "all_grid_oracle_pass_fraction_minimum": 0.90,
            "top_k_oracle_pass_fraction_minimum": 0.80,
            "replay_top1_pass_fraction_minimum": 0.80,
        })

    def test_top_k_and_replay_selector_can_both_pass(self) -> None:
        result = summarize_rows(rows(
            true_control=8, all_grid=9, top_k=8, top1=8,
        ))
        self.assertEqual(
            result["status"],
            "development-graybox-topk-and-selector-pass-not-promoted",
        )
        self.assertTrue(result["selector_viable"])

    def test_top_k_can_pass_while_selector_is_still_needed(self) -> None:
        result = summarize_rows(rows(
            true_control=8, all_grid=9, top_k=8, top1=7,
        ))
        self.assertEqual(
            result["status"],
            "development-graybox-topk-recoverable-selector-needed",
        )
        self.assertTrue(result["top_k_viable"])
        self.assertFalse(result["selector_viable"])

    def test_capacity_without_top_k_rejects_replay_ranking(self) -> None:
        result = summarize_rows(rows(
            true_control=8, all_grid=9, top_k=7, top1=7,
        ))
        self.assertEqual(result["status"], "rejected-replay-ranking")
        self.assertTrue(result["capacity_viable"])

    def test_failed_capacity_closes_graybox_route(self) -> None:
        result = summarize_rows(rows(
            true_control=7, all_grid=10, top_k=10, top1=10,
        ))
        self.assertEqual(result["status"], "rejected-graybox-capacity")
        self.assertFalse(result["capacity_viable"])

    def test_empty_summary_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            summarize_rows([])


if __name__ == "__main__":
    unittest.main()
