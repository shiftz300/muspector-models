from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from .evaluate_multirate_fuzz import evaluate, quality_metrics, validate_calibration_entry
from .evaluate_asrnn_effect import _quality_gate
from .test_dfz_admission import fixtures


class ExactGain:
    def __call__(self, dry, controls, state=None):
        return dry * 1.4, state


class MultirateQualityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(999)

    def test_complete_metric_formulas_on_exact_model(self):
        dry = torch.randn(2, 4096) * .1
        rows = [{"dry": value, "wet": value * 1.4, "controls": torch.tensor([.5, .5])} for value in dry]
        measured = quality_metrics(ExactGain(), rows)
        self.assertTrue(_quality_gate(measured))
        for name in ("global_esr", "global_mae", "preemphasis_esr", "multiresolution_spectral_loss", "absolute_peak_error_p95"):
            self.assertEqual(measured[name], 0.)
        self.assertEqual(measured["global_esr_relative_improvement"], 1.)
        self.assertAlmostEqual(measured["bypass_global_esr"], (.4 / 1.4) ** 2, places=6)
        self.assertEqual(measured["scored_frames"], 2 * (4096 - 1024))
        self.assertTrue(measured["quiet_dataset_gate_vacuous"])

    def test_in_memory_audio_mutation_rejected(self):
        class BadGain:
            def __call__(self, dry, controls, state=None):
                dry.add_(.01)
                return dry * 1.4, state
        row = {"dry": torch.randn(4096) * .1, "wet": torch.randn(4096) * .1, "controls": torch.tensor([.5, .5])}
        with self.assertRaisesRegex(ValueError, "in-memory"):
            quality_metrics(BadGain(), [row])

    def calibration(self):
        calibration, _, _, _, digest, _ = fixtures()
        calibration.update(architecture="causal-multirate-gcn", runtime="multirate-pytorch-cpu",
                           source_and_audio_reverified=True, admitted=False)
        return calibration, digest

    def test_frozen_complete_calibration_contract(self):
        report, digest = self.calibration()
        before = deepcopy(report)
        partitions = validate_calibration_entry(report, digest)
        self.assertEqual(len(partitions["fit"]), 234)
        self.assertEqual(len(partitions["calibration"]), 54)
        self.assertEqual(report, before)

    def test_quality_and_provenance_failure_cannot_open_development(self):
        for metric, value in (("absolute_peak_error_p95", .02000001), ("global_esr", .05000001),
                              ("quiet_prediction_peak_maximum", .00101), ("spectral_loss_relative_improvement", .29999)):
            report, digest = self.calibration()
            report["calibration"][metric] = value
            with self.assertRaises(ValueError):
                validate_calibration_entry(report, digest)
        report, digest = self.calibration()
        report["quality_policy"]["automatic_normalization"] = True
        with self.assertRaises(ValueError):
            validate_calibration_entry(report, digest)

    def test_missing_entry_stops_before_any_model_or_audio_access(self):
        with patch("remix.evaluate_multirate_fuzz.load_candidate") as load, patch("remix.evaluate_multirate_fuzz.effect_files") as files:
            with self.assertRaisesRegex(ValueError, "no development audio access"):
                evaluate(Path("not-read.pt"), Path("not-read-corpus"), "development")
            load.assert_not_called()
            files.assert_not_called()

    def test_locked_final_is_not_an_option(self):
        with self.assertRaisesRegex(ValueError, "locked-final forbidden"):
            evaluate(Path("not-read.pt"), Path("not-read-corpus"), "locked-final")


if __name__ == "__main__":
    unittest.main()
