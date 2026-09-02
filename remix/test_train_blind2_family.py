import unittest

import numpy as np

import torch

from .train_blind2_family import _binary_metrics, _family_target, _weighted_masked_bce


class TrainBlind2FamilyTest(unittest.TestCase):
    def test_binary_metrics_count_support_without_sklearn_support_value(self) -> None:
        probabilities = np.asarray([[0.1], [0.9], [0.8], [0.2]])
        targets = np.asarray([[0.0], [1.0], [1.0], [0.0]])
        sources = [
            "egfxset-real",
            "dafx-random-position-presence",
            "synthetic-product-render:no-nonlinear:count-0:egfxset",
            "synthetic-product-render:no-nonlinear:count-0:egfxset",
        ]
        metrics = _binary_metrics(probabilities, targets, 0.5, sources)
        self.assertEqual(metrics["positive_examples"], 2)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(
            metrics["strata"]["synthetic-product-render"]["positive_examples"], 1
        )

    def test_any_gate_target_is_order_free_presence_union(self) -> None:
        values = torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
        torch.testing.assert_close(_family_target(values, "any"), torch.tensor([[1.0], [0.0]]))

    def test_weighted_bce_emphasizes_selected_positive_examples(self) -> None:
        logits = torch.zeros((2, 1))
        expected = torch.tensor([[1.0], [0.0]])
        mask = torch.ones((2, 1))
        ordinary = _weighted_masked_bce(logits, expected, mask, torch.ones((2, 1)))
        weighted = _weighted_masked_bce(
            logits + torch.tensor([[-1.0], [0.0]]),
            expected,
            mask,
            torch.tensor([[2.0], [1.0]]),
        )
        self.assertGreater(weighted, ordinary)


if __name__ == "__main__":
    unittest.main()
