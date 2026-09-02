import unittest

import torch

from .audit_multirate_fuzz import CheckedMultirate, validate_call
from .multirate_fuzz import MultirateFuzz


class TrainedAuditBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = MultirateFuzz(audio_width=2, audio_blocks=2, controller_width=2, controller_blocks=2).eval()
        self.dry, self.controls = torch.zeros(1, 17), torch.tensor([[.5, .5]])

    def test_static_and_dynamic(self):
        boundary = CheckedMultirate(self.model, self.model)
        value, state = boundary(self.dry, self.controls)
        self.assertEqual(float(value.detach().abs().max()), 0)
        self.assertEqual(int(state[-1]), 17)
        _, next_state = boundary(self.dry, self.controls[:, None].expand(-1, 17, -1), state)
        self.assertEqual(int(next_state[-1]), 34)

    def test_reject_control_precision_and_shape(self):
        for controls in (self.controls.double(), torch.zeros(1, 16, 2), torch.tensor([[float("nan"), .5]]), torch.ones(1, 2) * 1.1):
            with self.assertRaises(ValueError):
                validate_call(self.model, self.dry, controls, None)

    def test_reject_bad_states(self):
        good = self.model.initial_state(self.dry)
        for index, replacement in ((0, torch.zeros(1, 2, 2)), (0, good[0].double()),
                                   (0, good[0] + float("inf")), (-1, torch.tensor(64)),
                                   (-1, torch.tensor(1.)), (-3, torch.zeros(1, 63))):
            state = list(good)
            state[index] = replacement
            with self.assertRaises(ValueError):
                validate_call(self.model, self.dry, self.controls, state)

    def test_reject_nonfinite_output(self):
        def broken(dry, controls, state):
            state = list(self.model.initial_state(dry))
            state[-1] = torch.tensor(17)
            return dry + float("nan"), tuple(state)
        with self.assertRaises(ValueError):
            CheckedMultirate(broken, self.model)(self.dry, self.controls)

    def test_reject_wrong_backend_clock(self):
        def broken(dry, controls, state):
            return dry, self.model.initial_state(dry)
        with self.assertRaises(ValueError):
            CheckedMultirate(broken, self.model)(self.dry, self.controls)


if __name__ == "__main__":
    unittest.main()
