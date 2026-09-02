import unittest

from remix.identity_eval import GATES, spread


class IdentityEvalTests(unittest.TestCase):
    def test_shadow_gates_forbid_delivery(self):
        self.assertEqual(GATES["automatic_deliveries"], 0)
        self.assertLessEqual(GATES["dfz_false_route_total"], 0.10)

    def test_spread_is_deterministic_and_includes_bounds(self):
        values = list(range(10))
        self.assertEqual(spread(values, 3), [0, 4, 9])


if __name__ == "__main__":
    unittest.main()
