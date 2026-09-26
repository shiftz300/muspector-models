from __future__ import annotations

import unittest

import numpy as np
import torch

from .train_blind2_nonlinear_hard import (
    FROZEN_GATES,
    MAX_EPOCHS,
    _acceptance,
    _hard_weights,
    _metrics,
    _parse_nonlinear_source,
    _selection_key,
)


class NonlinearHardTrainingTests(unittest.TestCase):
    def test_training_budget_and_gates_are_frozen(self) -> None:
        self.assertEqual(MAX_EPOCHS, 4)
        self.assertEqual(FROZEN_GATES["synthetic_count_1_recall"], 0.75)
        self.assertEqual(FROZEN_GATES["negative_false_positive_rate"], 0.05)

    def test_source_parser_is_strict(self) -> None:
        source = "synthetic-product-render:nonlinear:tanh:low:weak:count-1:medley-solos-db"
        self.assertEqual(
            _parse_nonlinear_source(source),
            {
                "implementation": "tanh",
                "drive": "low",
                "residual": "weak",
                "chain_count": "count-1",
                "clean_source": "medley-solos-db",
            },
        )
        self.assertIsNone(_parse_nonlinear_source("egfxset-real"))
        with self.assertRaises(ValueError):
            _parse_nonlinear_source("synthetic-product-render:nonlinear:broken")

    def test_hard_weights_are_positive_only_and_capped(self) -> None:
        expected = torch.tensor([[1.0], [1.0], [0.0], [1.0]])
        sources = [
            "synthetic-product-render:nonlinear:tanh:low:weak:count-1:a",
            "synthetic-product-render:nonlinear:tanh:high:strong:count-1:b",
            "synthetic-product-render:nonlinear:tanh:low:weak:count-1:c",
            "egfxset-real",
        ]
        self.assertEqual(_hard_weights(expected, sources).tolist(), [[6.0], [3.0], [1.0], [1.0]])

    def test_metrics_expose_every_frozen_hard_stratum(self) -> None:
        sources = [
            "synthetic-product-render:nonlinear:tanh:low:weak:count-1:a",
            "egfxset-real",
            "dafx-random-position-presence",
            "synthetic-product-render:no-nonlinear:count-0:b",
        ]
        targets = np.asarray([[1.0], [1.0], [1.0], [0.0]])
        probabilities = np.asarray([[0.9], [0.9], [0.9], [0.1]])
        metrics = _metrics(probabilities, targets, 0.5, sources)
        self.assertTrue(_acceptance(metrics)["accepted"])
        self.assertEqual(
            set(metrics["strata"]),
            {
                "egfxset-real",
                "dafx-random-position-presence",
                "synthetic-nonlinear-chain-count:count-1",
                "synthetic-nonlinear-drive:low",
                "synthetic-nonlinear-residual:weak",
            },
        )

    def test_selection_prefers_more_frozen_gates_before_f1(self) -> None:
        sources = [
            "synthetic-product-render:nonlinear:tanh:low:weak:count-1:a",
            "egfxset-real",
            "dafx-random-position-presence",
            "synthetic-product-render:no-nonlinear:count-0:b",
        ]
        targets = np.asarray([[1.0], [1.0], [1.0], [0.0]])
        complete = _metrics(np.asarray([[0.9], [0.9], [0.9], [0.1]]), targets, 0.5, sources)
        misses_hard = _metrics(np.asarray([[0.1], [0.9], [0.9], [0.1]]), targets, 0.5, sources)
        self.assertGreater(_selection_key(complete), _selection_key(misses_hard))


if __name__ == "__main__":
    unittest.main()
