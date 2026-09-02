import unittest

from remix.order2 import accepted, summarize


class Order2Tests(unittest.TestCase):
    def test_summary_reports_zero_selected_regression_after_guard(self):
        rows = [
            {
                "truth": [True, False, False],
                "mask": [True, False, False],
                "original": ["delay", "drive"],
                "proposed": ["drive", "delay"],
                "selected": ["delay", "drive"],
                "original_error": 0.1,
                "proposed_error": 0.2,
                "selected_error": 0.1,
                "guarded": True,
                "inputs_unchanged": True,
            }
        ]
        result = summarize(rows)
        self.assertEqual(result["audio"]["proposed_regressions"], 1)
        self.assertEqual(result["audio"]["selected_regressions"], 0)

    def test_gate_rejects_any_selected_audio_regression(self):
        value = {
            "proposed": {"exact": 0.8, "pairwise": 0.85},
            "selected": {"exact": 0.8, "pairwise": 0.85},
            "original": {"exact": 0.8, "pairwise": 0.85},
            "audio": {"selected_regressions": 1},
            "inputs_unchanged": True,
        }
        domains = {name: dict(value) for name in ("real", "reference", "alternate", "stress", "pedalboard")}
        self.assertFalse(accepted(domains)[0])

    def test_gate_separates_recognition_from_safe_delivery(self):
        value = {
            "proposed": {"exact": 0.8, "pairwise": 0.85},
            "selected": {"exact": 0.64, "pairwise": 0.73},
            "original": {"exact": 0.63, "pairwise": 0.72},
            "audio": {"selected_regressions": 0},
            "inputs_unchanged": True,
        }
        domains = {
            name: dict(value)
            for name in ("real", "reference", "alternate", "stress", "pedalboard")
        }
        passed, failures = accepted(domains)
        self.assertTrue(passed, failures)

    def test_gate_still_rejects_weak_proposal(self):
        value = {
            "proposed": {"exact": 0.5, "pairwise": 0.6},
            "selected": {"exact": 0.8, "pairwise": 0.85},
            "original": {"exact": 0.8, "pairwise": 0.85},
            "audio": {"selected_regressions": 0},
            "inputs_unchanged": True,
        }
        domains = {
            name: dict(value)
            for name in ("real", "reference", "alternate", "stress", "pedalboard")
        }
        passed, failures = accepted(domains)
        self.assertFalse(passed)
        self.assertIn("pedalboard.exact", failures)


if __name__ == "__main__":
    unittest.main()
