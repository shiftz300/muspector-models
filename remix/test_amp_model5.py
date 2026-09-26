import unittest

import torch

from .amp_model5 import AmpStageSupervisedInverseExpert


class AmpStageSupervisedModelTests(unittest.TestCase):
    def test_identity_stage_boundaries_and_route_contract(self):
        model = AmpStageSupervisedInverseExpert().eval()
        wet = torch.randn(2, 4096) * 0.05
        controls = torch.tensor([[0.2, 0.5, 0.8, 0.3], [0.8, 0.5, 0.2, 0.7]])
        with torch.inference_mode():
            restored, uncertainty, state = model(wet, controls)
            staged, preamp = model.forward_stages(wet, controls)
        torch.testing.assert_close(restored, wet, rtol=1.0e-5, atol=1.0e-6)
        torch.testing.assert_close(staged, wet, rtol=1.0e-5, atol=1.0e-6)
        torch.testing.assert_close(preamp, wet, rtol=1.0e-5, atol=1.0e-6)
        self.assertTrue(torch.all(uncertainty > 0.0))
        self.assertIsNone(state)
        manifest = model.manifest()
        self.assertFalse(manifest["graph_order_input"])
        self.assertFalse(manifest["neighbor_effect_input"])
        self.assertFalse(manifest["recurrent_state_input"])
        self.assertEqual(manifest["measured_intermediate"], "preamp")

    def test_stage_parameter_sets_do_not_overlap(self):
        model = AmpStageSupervisedInverseExpert()
        first = {id(value) for value in model.stage_parameters("power-tone")}
        second = {id(value) for value in model.stage_parameters("preamp")}
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertFalse(first & second)

    def test_external_state_is_rejected(self):
        model = AmpStageSupervisedInverseExpert()
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 4096), torch.zeros(1, 4), torch.zeros(1))


if __name__ == "__main__":
    unittest.main()
