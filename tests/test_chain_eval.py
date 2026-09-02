import unittest

from remix.chain_eval import failures, summarize


class ChainEvalTests(unittest.TestCase):
    def test_empty_report_fails_all_capability_gates(self):
        metrics = summarize([])
        rejected = failures(metrics)
        self.assertIn("coverage", rejected)
        self.assertIn("family-exact", rejected)
        self.assertIn("order-exact", rejected)

    def test_audio_regression_is_never_tolerated(self):
        row = {
            "accepted": True,
            "family_exact": True,
            "domain": "reference",
            "order_truth": [1, 0, 0],
            "order_mask": [1, 0, 0],
            "proposed": ["drive", "delay"],
            "selected": ["drive", "delay"],
            "original_error": 0.1,
            "selected_error": 0.2,
            "controls": {},
            "inputs_unchanged": True,
        }
        self.assertEqual(summarize([row])["order"]["audio_regressions"], 1)


if __name__ == "__main__":
    unittest.main()
