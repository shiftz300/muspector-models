import unittest

from remix.route_seal import gate


def rows(scope, models, routed):
    return [
        {"scope": scope, "model": model, "routed": routed,
         "automatic_delivery": False, "inputs_unchanged": True, "nonfinite": 0}
        for model in models for _ in range(4)
    ]


class RouteSealTests(unittest.TestCase):
    def test_frozen_gate_accepts_full_balanced_result(self):
        result = rows("rat", ("a", "b", "c"), True) + rows(
            "other", ("d", "e", "f", "g", "h", "i"), False
        )
        self.assertTrue(gate(result)[0])

    def test_frozen_gate_rejects_one_model_short(self):
        result = rows("rat", ("a", "b"), True) + rows(
            "other", ("d", "e", "f", "g", "h", "i"), False
        )
        passed, failures, _ = gate(result)
        self.assertFalse(passed)
        self.assertIn("rat_models<3", failures)


if __name__ == "__main__":
    unittest.main()
