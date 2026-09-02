import unittest

from remix.seal import CONTROL_NAMES, accepted, summarize


def domain(exact=0.8, pairwise=0.85, error=0.1, control=0.05):
    controls = {
        name: {"values": 2, "mae": control, "p95": control, "raw": [control, control]}
        for name in CONTROL_NAMES
    }
    return {
        "examples": 2,
        "eligible": 2,
        "relations": 3,
        "exact": exact,
        "pairwise": pairwise,
        "baseline_error": error,
        "selected_error": error,
        "macro_mae": control,
        "macro_p95": control,
        "controls": controls,
        "inputs_unchanged": True,
    }


class SealTests(unittest.TestCase):
    def test_summarize_masks_relations_and_controls(self):
        row = {
            "truth": [True, False, True],
            "prediction": [True, True, True],
            "mask": [True, False, True],
            "baseline_error": 0.2,
            "selected_error": 0.1,
            "controls": {CONTROL_NAMES[0]: 0.05},
            "inputs_unchanged": True,
        }
        result = summarize([row])
        self.assertEqual(result["exact"], 1.0)
        self.assertEqual(result["pairwise"], 1.0)
        self.assertAlmostEqual(result["macro_mae"], 0.05)

    def test_gate_rejects_audio_regression(self):
        domains = {
            name: domain()
            for name in ("real", "reference", "alternate", "stress", "pedalboard")
        }
        self.assertTrue(accepted(domains)[0])
        domains["stress"]["selected_error"] = 0.1001
        passed, failures, capabilities = accepted(domains)
        self.assertFalse(passed)
        self.assertIn("stress.audio-regression", failures)
        self.assertFalse(capabilities["order"]["accepted"])
        self.assertTrue(capabilities["knob"]["accepted"])


if __name__ == "__main__":
    unittest.main()
