import unittest

import torch

from remix.mixed import bucket, decision, objective, split_egfx, widen
from remix.net import SpectralNet


def metrics(esr=.2, mae=.2, sidr=2., tail=.8, peak=.8):
    return {
        "coverage": 1.,
        "aligned_esr_improvement": esr,
        "aligned_mae_improvement": mae,
        "sidr_improvement_db": sidr,
        "baseline_aligned_esr_p95": 1.,
        "restored_aligned_esr_p95": tail,
        "baseline_aligned_peak_error_p95": 1.,
        "restored_aligned_peak_error_p95": peak,
        "source_mutations": 0,
        "nonfinite_outputs": 0,
        "geometry_errors": 0,
    }


class MixedTests(unittest.TestCase):
    def setUp(self):
        self.gates = {
            "asrnn": {"coverage": .75, "aligned_esr_improvement": .1, "aligned_mae_improvement": .05, "sidr_improvement_db": 1., "aligned_p95_ratio_max": 1., "aligned_peak_ratio_max": 1.},
            "egfx": {"coverage": .95, "aligned_esr_improvement": .03, "aligned_mae_improvement": .03, "sidr_improvement_db": .5, "aligned_p95_ratio_max": 1., "aligned_peak_ratio_max": 1.},
        }

    def test_both_domains_must_pass(self):
        passed, failures = decision(metrics(), metrics(esr=.01), self.gates)
        self.assertFalse(passed)
        self.assertEqual(failures["asrnn"], [])
        self.assertIn("aligned_esr_improvement", failures["egfx"])

    def test_objective_is_controlled_by_weakest_gate(self):
        strong = objective(metrics(), metrics(), self.gates)
        weak = objective(metrics(), metrics(sidr=.1), self.gates)
        self.assertGreater(strong, weak)

    def test_widen_preserves_source_function(self):
        torch.manual_seed(4)
        source = SpectralNet(4).eval()
        torch.nn.init.normal_(source.head.weight, std=.01)
        expanded = widen(source, 6).eval()
        audio = torch.randn(1, 4097) * .05
        with torch.inference_mode():
            self.assertLess(float((source(audio) - expanded(audio)).abs().max()), 2e-7)

    def test_declared_group_buckets_do_not_overlap(self):
        rows = [
            {"group": name, "eligible": True, "split": "fit"}
            for name in ("a", "b", "c", "d")
        ]
        groups = {name: [] for name in ("fit", "calibration", "development")}
        for row in rows:
            groups[("calibration", "development", "fit")[bucket(row["group"]) % 3]].append(bucket(row["group"]))
        selected = split_egfx(rows, {"egfx_split": groups})
        observed = [{row["group"] for row in selected[name]} for name in selected]
        self.assertFalse(any(observed[i] & observed[j] for i in range(3) for j in range(i)))


if __name__ == "__main__":
    unittest.main()
