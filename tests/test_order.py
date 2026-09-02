import unittest

from remix.order import guard


def report(proposed_error: float, original_error: float = 0.1):
    return {
        "normalized_controls": [0.5] * 9,
        "decision": "transfer-order",
        "chain": {},
        "order": {
            "selected": ["drive", "delay"],
            "selected_reconstruction_error": proposed_error,
            "ranked": [
                {"topology": ["delay", "drive"], "reconstruction_error": original_error},
                {"topology": ["drive", "delay"], "reconstruction_error": proposed_error},
            ],
        },
        "transfer": {"original_selected": ["delay", "drive"]},
    }


class OrderTests(unittest.TestCase):
    def test_guard_reverts_audio_regression(self):
        value = guard(report(0.100001))
        self.assertTrue(value["order2"]["guarded"])
        self.assertEqual(value["order"]["selected"], ["delay", "drive"])
        self.assertEqual(value["order"]["selected_reconstruction_error"], 0.1)
        self.assertEqual(value["decision"], "order-fallback")

    def test_guard_keeps_non_regressing_proposal(self):
        value = guard(report(0.09))
        self.assertFalse(value["order2"]["guarded"])
        self.assertEqual(value["order"]["selected"], ["drive", "delay"])
        self.assertTrue(value["order2"]["order_used"])

    def test_single_family_never_uses_order(self):
        value = guard({"order": {"selected": ["drive"]}})
        self.assertFalse(value["order2"]["order_used"])


if __name__ == "__main__":
    unittest.main()
