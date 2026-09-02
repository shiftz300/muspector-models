import unittest

import numpy as np
import torch

from .train_blind2 import (
    _acceptance,
    _fit_calibration,
    _masked_bce,
    _metrics,
    _probabilities,
    _thresholds,
)


class TrainBlind2Test(unittest.TestCase):
    def test_calibration_and_metrics_contract(self) -> None:
        targets = np.asarray([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ] * 8, dtype=np.float32)
        probabilities = targets * 0.96 + (1.0 - targets) * 0.01
        thresholds = _thresholds(probabilities, targets)
        names = (
            "egfxset-real",
            "dafx-random-position-presence",
            "synthetic-product-render:no-nonlinear",
        )
        sources = [names[index % len(names)] for index in range(len(targets))]
        metrics = _metrics(probabilities, targets, thresholds, sources)
        self.assertGreater(metrics["macro_f1"], 0.99)
        self.assertEqual(metrics["clean_false_positive_rate"], 0.0)
        self.assertEqual(set(metrics["strata"]), {
            "egfxset-real",
            "dafx-random-position-presence",
            "synthetic-product-render",
        })
        self.assertGreater(
            metrics["strata"]["dafx-random-position-presence"]["per_label"]["nonlinear"]["recall"],
            0.99,
        )
        self.assertTrue(_acceptance(metrics)["accepted"])

    def test_probability_calibration_is_finite(self) -> None:
        logits = np.asarray([[100.0, -100.0, 0.0, 1.0]])
        calibration = [{"scale": 1.0, "bias": 0.0}] * 4
        values = _probabilities(logits, calibration)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(((values >= 0.0) & (values <= 1.0)).all())

    def test_binary_family_calibration_and_threshold(self) -> None:
        logits = np.asarray([[-4.0], [-3.0], [2.0], [4.0]])
        targets = np.asarray([[0.0], [0.0], [1.0], [1.0]])
        calibration = _fit_calibration(logits, targets)
        probabilities = _probabilities(logits, calibration)
        thresholds = _thresholds(probabilities, targets)
        self.assertEqual(len(calibration), 1)
        self.assertEqual(len(thresholds), 1)
        np.testing.assert_array_equal(probabilities >= thresholds, targets.astype(bool))

    def test_masked_bce_ignores_unknown_negative_labels(self) -> None:
        logits = torch.tensor([[2.0, 100.0, -100.0, 0.0]], requires_grad=True)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        loss = _masked_bce(logits, expected, mask)
        reference = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, :1], expected[:, :1])
        torch.testing.assert_close(loss, reference)
        loss.backward()
        self.assertEqual(float(logits.grad[0, 1:].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
