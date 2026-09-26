from __future__ import annotations

import unittest

import numpy as np

from .audit_drive_control_identifiability import (
    AXES,
    TOP_K,
    candidate_grid,
    clean_example,
    normalized_controls,
    render_batch,
    summarize_rows,
    truth_controls,
)


class DriveControlIdentifiabilityTests(unittest.TestCase):
    def test_grid_is_complete_unique_and_in_contract(self) -> None:
        rows = candidate_grid()
        self.assertEqual(len(rows), 324)
        self.assertEqual(len({tuple(row.items()) for row in rows}), len(rows))
        normalized = normalized_controls(rows)
        self.assertEqual(normalized.shape, (324, 5))
        self.assertTrue(((0.0 <= normalized) & (normalized <= 1.0)).all())

    def test_truth_schedule_is_balanced_and_grid_contained(self) -> None:
        grid = candidate_grid()
        truth = [truth_controls(index) for index in range(12)]
        self.assertTrue(all(row in grid for row in truth))
        self.assertEqual({row["shape"] for row in truth}, set(AXES["shape"]))
        self.assertEqual({row["drive"] for row in truth}, set(AXES["drive"]))

    def test_renderer_accepts_shared_and_per_candidate_audio(self) -> None:
        controls = candidate_grid()[:2]
        clean = clean_example(0)[:512]
        shared = render_batch(clean, controls)
        separate = render_batch(np.stack((clean, clean)), controls)
        self.assertEqual(shared.shape, (2, 512))
        self.assertTrue(np.array_equal(shared, separate))
        with self.assertRaises(ValueError):
            render_batch(np.stack((clean, clean, clean)), controls)

    def test_frozen_summary_requires_narrow_truth_covering_top_k(self) -> None:
        rows = []
        for index in range(12):
            truth = truth_controls(index)
            intervals = {}
            for name, axis in AXES.items():
                if name == "shape":
                    intervals[name] = {"values": [truth[name]], "covers_truth": True}
                else:
                    intervals[name] = {
                        "minimum": truth[name], "maximum": truth[name],
                        "covers_truth": True, "normalized_width": 0.0,
                    }
            rows.append({
                "truth_in_top_k": True,
                "shape_top1": True,
                "top_k_boundary_margin": 0.1,
                "intervals": intervals,
            })
        self.assertLess(TOP_K, len(candidate_grid()))
        self.assertTrue(summarize_rows(rows)["accepted"])
        rows[0]["intervals"]["bias"]["covers_truth"] = False
        rows[1]["intervals"]["bias"]["covers_truth"] = False
        self.assertFalse(summarize_rows(rows)["accepted"])


if __name__ == "__main__":
    unittest.main()
