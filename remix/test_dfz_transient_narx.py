"""CPU contracts for dual-event sampling and same-index transient supervision."""
import unittest

import numpy as np
import torch

from .dfz_dynamic_readout import CausalTapBank
from .train_dfz_long_tcn import context_windows
from .train_dfz_transient_narx import (CornerCycle, event_weights, event_window,
                                       signed_events, transient_loss)


class TransientNARXTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_cycle_covers_every_recording_at_every_corner(self):
        cycle = CornerCycle(np.random.default_rng(988))
        for _ in range(26):
            cycle.next()
        self.assertTrue(np.array_equal(cycle.counts, np.ones((9, 26))))
        for _ in range(26):
            cycle.next()
        self.assertTrue(np.array_equal(cycle.counts, np.full((9, 26), 2)))
        self.assertTrue(all(row["distinct_fit_recordings"] == 26 for row in cycle.coverage().values()))

    def test_both_signed_event_sets_and_clamped_complete_windows(self):
        wet, core = torch.zeros(4_000), torch.zeros(4_000)
        wet[20], wet[1_500], wet[2_000] = 10., .3, -.4
        core[30], core[1_800], core[2_600] = -10., .5, -.2
        self.assertEqual(signed_events(wet), (1_500, 2_000))
        self.assertEqual(signed_events(core), (1_800, 2_600))
        rng = np.random.default_rng(988)
        for event in (1_024, 1_500, 3_999):
            start, offset = event_window(event, len(wet), rng)
            self.assertEqual(start + offset, event)
            self.assertGreaterEqual(start, 1_024)
            self.assertLessEqual(start + 256, len(wet))
            self.assertTrue(0 <= offset < 256)

    def test_event_center_weights_have_exact_17_frame_support(self):
        weights = event_weights(torch.zeros(2, 256), torch.tensor([128, 0]))
        self.assertEqual(int((weights[0] == 16).sum()), 17)
        self.assertEqual(int((weights[1] == 16).sum()), 9)
        self.assertTrue(torch.equal(weights[0, 120:137], torch.full((17,), 16.)))
        self.assertTrue(torch.equal(event_weights(torch.zeros(2, 512)), torch.ones(2, 512)))

    def test_false_core_peak_is_penalized_at_its_actual_time_and_sign(self):
        wet = torch.zeros(1, 256)
        wet[0, 32] = .2
        prediction = torch.zeros_like(wet)
        prediction[0, 128] = .2
        self.assertEqual(float(prediction.abs().max()), float(wet.abs().max()))
        prediction.requires_grad_()
        loss = transient_loss(prediction, wet, torch.tensor([.01]), torch.tensor([.01]), torch.tensor([128])).mean()
        loss.backward()
        self.assertGreater(float(loss.detach()), 0.)
        self.assertGreater(float(prediction.grad[0, 128]), 0.)
        self.assertLess(float(prediction.grad[0, 32]), 0.)
        exact = wet.clone().requires_grad_()
        zero = transient_loss(exact, wet, torch.tensor([.01]), torch.tensor([.01]), torch.tensor([128])).mean()
        zero.backward()
        self.assertEqual(float(zero.detach()), 0.)
        self.assertEqual(int(torch.count_nonzero(exact.grad)), 0)

    def test_file_energy_and_pre_weight_normalization_are_exact(self):
        wet = torch.tensor([[.1, -.3, .2, 0.], [0., 1e-5, 0., -1e-5]], dtype=torch.float64)
        prediction = wet * .8
        energy, pre_energy = torch.tensor([.05, 0.], dtype=torch.float64), torch.tensor([.02, 0.], dtype=torch.float64)
        offsets = torch.tensor([1, 2])
        weights = event_weights(wet, offsets)
        expected = (weights * (prediction - wet).square()).sum(1) / weights.sum(1) / energy.clamp_min(1e-5)
        dp, dt = prediction[:, 1:] - .95 * prediction[:, :-1], wet[:, 1:] - .95 * wet[:, :-1]
        expected += .1 * (weights[:, 1:] * (dp - dt).square()).sum(1) / weights[:, 1:].sum(1) / pre_energy.clamp_min(1e-5)
        torch.testing.assert_close(transient_loss(prediction, wet, energy, pre_energy, offsets), expected, atol=0, rtol=0)

    def test_short_windows_use_exact_real_64_sample_prefix_for_both_inputs(self):
        torch.manual_seed(988)
        dry = torch.randn(4_000) * .03
        core = torch.tanh(dry * 7)
        bank = CausalTapBank()
        with torch.inference_mode():
            complete = bank.cpu_features(dry, core)
            for frames in (256, 512):
                start = 1_507
                x = context_windows([dry], [start], frames, 64)
                p = context_windows([core], [start], frames, 64)
                local, _ = bank(x, p)
                torch.testing.assert_close(local[0, 64:], complete[start:start + frames], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
