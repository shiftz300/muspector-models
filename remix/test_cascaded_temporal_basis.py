import unittest

import torch

from .cascaded_multirate_fuzz import GEOMETRY, CascadedMultirateFuzz
from .cascaded_temporal_basis import INDICES, attach, materialize


class CascadedBasisTests(unittest.TestCase):
    @torch.inference_mode()
    def test_exact_initial_and_changed_materialized_model(self):
        torch.set_num_threads(1)
        torch.manual_seed(1016)
        source = CascadedMultirateFuzz(**GEOMETRY).eval()
        source.audio.output.weight.normal_(0, .1)
        dry, knobs = torch.randn(2, 6173)*.03, torch.rand(2, 6173, 2)
        original = source(dry, knobs)[0]
        attach(source)
        self.assertTrue(torch.equal(original, source(dry, knobs)[0]))
        for index in INDICES:
            source.audio.convolutions[index].coefficients.normal_(0, 1e-4)
        expected = source(dry, knobs)[0]
        state = materialize(source)
        self.assertFalse(any(any(key in name for key in ("basis", "coefficients", "base_weight")) for name in state))
        actual = CascadedMultirateFuzz(**GEOMETRY).eval()
        actual.load_state_dict(state, strict=True)
        self.assertTrue(torch.equal(expected, actual(dry, knobs)[0]))
        self.assertEqual(float(actual(torch.zeros_like(dry), knobs)[0].abs().max()), 0)
        with self.assertRaises(ValueError):
            attach(source)


if __name__ == "__main__":
    unittest.main()
