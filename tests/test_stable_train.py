import unittest

import torch

from remix.train_stable_rat import (
    GammatoneLoss,
    StableRatTrain,
    _calibration_score,
    _training_loss,
)


class StableRatTrainingTests(unittest.TestCase):
    def test_projection_enforces_paper_stability_constraints(self):
        model = StableRatTrain(hidden=4, layers=2, input_coef=21.4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(0.0, 10.0)
        model.project_()
        audit = model.audit()
        self.assertTrue(audit["passed"])
        self.assertTrue(all(row["candidate_infinity_norm"] <= 0.995001 for row in audit["layers"]))

    def test_projected_model_has_exact_static_and_dynamic_silence(self):
        model = StableRatTrain(hidden=4, layers=2, input_coef=21.4)
        model.project_()
        silence = torch.zeros(3, 257)
        static = torch.rand(3, 3)
        dynamic = torch.rand(3, 257, 3)
        first, _ = model(silence, static)
        second, _ = model(silence, dynamic)
        self.assertEqual(float(first.detach().abs().max()), 0.0)
        self.assertEqual(float(second.detach().abs().max()), 0.0)

    def test_gammatone_loss_is_finite_and_differentiable(self):
        loss = GammatoneLoss()
        prediction = torch.randn(2, 2_048, requires_grad=True)
        target = torch.randn(2, 2_048)
        value = loss(prediction, target)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_fft_gammatone_matches_direct_causal_convolution(self):
        loss = GammatoneLoss()
        prediction = torch.randn(2, 257)
        target = torch.randn(2, 257)
        error = target - prediction
        padded = torch.nn.functional.pad(error, (loss.filters.shape[1] - 1, 0))
        direct = torch.nn.functional.conv1d(
            padded.unsqueeze(1).expand(-1, loss.filters.shape[0], -1),
            loss.filters.flip(1).unsqueeze(1),
            groups=loss.filters.shape[0],
        )
        expected = (error.sub(direct.sum(1)).abs() + direct.abs().sum(1)).mean()
        self.assertTrue(torch.allclose(loss(prediction, target), expected, rtol=2e-5, atol=2e-5))

    def test_checkpoint_score_rejects_average_gain_with_worse_tail_and_quiet_output(self):
        baseline = {
            "global_esr": 0.0527654981,
            "mean_per_file_esr": 0.0915334543,
            "median_per_file_esr": 0.0424222741,
            "p95_per_file_esr": 0.2971220911,
            "peak_ratio_median": 0.9627334774,
            "peak_ratio_p95": 1.1175410509,
            "absolute_peak_error_p95": 0.0091291312,
            "quiet_prediction_peak_maximum": 0.0017679264,
        }
        average_only_gain = {
            **baseline,
            "global_esr": 0.0517177861,
            "mean_per_file_esr": 0.0844282637,
            "p95_per_file_esr": 0.3332287073,
            "quiet_prediction_peak_maximum": 0.0022083432,
        }
        self.assertGreater(
            _calibration_score(average_only_gain),
            _calibration_score(baseline),
        )

    def test_quality_loss_adds_relative_and_peak_constraints(self):
        auditory = GammatoneLoss()
        target = torch.zeros(2, 257)
        prediction = torch.full((2, 257), 0.002, requires_grad=True)
        base = _training_loss(prediction, target, auditory, 0.0, 0.0, 0.0)
        guarded = _training_loss(prediction, target, auditory, 0.02, 0.1, 0.5)
        self.assertGreater(float(guarded.detach()), float(base.detach()))
        guarded.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == "__main__":
    unittest.main()
