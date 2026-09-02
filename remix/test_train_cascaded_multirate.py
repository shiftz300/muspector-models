import copy
import unittest

import torch

from .cascaded_candidate import model_from_payload
from .cascaded_multirate_fuzz import ARCHITECTURE, GEOMETRY, CascadedMultirateFuzz
from .train_cascaded_multirate import learning_rate, pooled_loss


class CascadedTrainingTests(unittest.TestCase):
    def test_fixed_schedule(self):
        self.assertEqual(learning_rate(500), .001)
        self.assertEqual(learning_rate(10000), .001)
        self.assertAlmostEqual(learning_rate(20000), .00003)
        for step in (0, 20001):
            with self.assertRaises(ValueError):
                learning_rate(step)

    def test_pooled_loss_is_same_time_and_finite(self):
        target = torch.tensor([[0., .1, -.1, .02], [0., .01, -.01, .002]])
        predicted = (target+.03).requires_grad_()
        original = target.clone()
        loss = pooled_loss(predicted, target, target.abs().amax(1), .02, .03)
        self.assertAlmostEqual(float(loss[0].detach()), float(loss[1].detach()), places=5)
        loss.mean().backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertTrue(torch.equal(target, original))
        self.assertEqual(float(pooled_loss(target, target, target.abs().amax(1), .02, .03).sum()), 0)
        for value in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                pooled_loss(predicted, target, target.abs().amax(1), value, .03)

    def test_exact_typed_schema_and_no_dtype_coercion(self):
        model = CascadedMultirateFuzz(**GEOMETRY)
        payload = {"experimental_schema": 1, "architecture": ARCHITECTURE, "device": "dfz", "sample_rate": 48000,
                   "geometry": GEOMETRY, "state_dict": model.state_dict()}
        actual = model_from_payload(payload)
        self.assertTrue(all(torch.equal(value, actual.state_dict()[name]) for name, value in model.state_dict().items()))
        for key, value in (("architecture", "causal-multirate-gcn"), ("sample_rate", 44100), ("geometry", {})):
            with self.assertRaises(ValueError):
                model_from_payload({**payload, key: value})
        for value in (torch.ones_like(next(iter(model.state_dict().values()))).double(), torch.tensor([float("nan")])):
            broken = copy.deepcopy(payload)
            broken["state_dict"][next(iter(model.state_dict()))] = value
            with self.assertRaises(ValueError):
                model_from_payload(broken)


if __name__ == "__main__":
    unittest.main()
