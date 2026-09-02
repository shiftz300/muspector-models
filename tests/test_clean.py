import unittest

import torch

from remix.clean import CleanNet, SpectralNet, accepted


class CleanTests(unittest.TestCase):
    def test_geometry_source_and_exact_bypass(self):
        model = CleanNet(channels=4, dilations=(1, 2)).eval()
        wet = torch.randn(2, 257) * 0.1
        before = wet.clone()
        restored = model(wet)
        self.assertEqual(restored.shape, wet.shape)
        self.assertTrue(torch.isfinite(restored).all())
        self.assertTrue(torch.equal(model(wet, strength=0), wet))
        self.assertTrue(torch.equal(wet, before))

    def test_gate_rejects_average_gain_with_tail_regression(self):
        metrics = {"coverage": .8, "esr_improvement": .5, "mae_improvement": .5, "sidr_improvement_db": 4., "baseline_esr_p95": 1., "restored_esr_p95": 1.01, "baseline_peak_error_p95": .2, "restored_peak_error_p95": .1, "source_mutations": 0, "nonfinite_outputs": 0, "geometry_errors": 0}
        gates = {"coverage": .75, "esr_improvement": .25, "mae_improvement": .20, "sidr_improvement_db": 3., "p95_ratio_max": 1., "peak_ratio_max": 1.}
        self.assertFalse(accepted(metrics, gates)[0])

    def test_spectral_geometry_and_exact_bypass(self):
        model = SpectralNet(channels=4).eval()
        wet = torch.randn(2, 4097) * 0.1
        before = wet.clone()
        restored = model(wet)
        self.assertEqual(restored.shape, wet.shape)
        self.assertTrue(torch.isfinite(restored).all())
        self.assertLess(float((restored - wet).abs().max().detach()), 2e-6)
        self.assertTrue(torch.equal(model(wet, strength=0), wet))
        self.assertTrue(torch.equal(wet, before))


if __name__ == "__main__":
    unittest.main()
