"""CPU-only exact-context and state contracts for the long residual."""
import unittest

import torch

from .stable_tcn_residual import ZeroGatedTCN
from .train_dfz_long_tcn import GenericLongTCNResidual, context_windows, receptive_field


class StatelessBase(torch.nn.Module):
    control_count = 2
    export_state_widths = [1]

    def forward(self, dry, controls, state=None):
        hidden = dry.new_zeros(1, dry.shape[0], 1)
        return dry * .4, ((hidden, hidden.clone()),)


class LongTCNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(984)

    def test_receptive_field_is_8191_and_step0_is_exact_source(self):
        tcn = ZeroGatedTCN(8, 12)
        self.assertEqual(receptive_field(tcn), 8_191)
        model = GenericLongTCNResidual(StatelessBase(), tcn)
        dry, controls = torch.randn(2, 311) * .03, torch.rand(2, 2)
        with torch.inference_mode():
            rendered, _ = model(dry, controls)
        torch.testing.assert_close(rendered, dry * .4, atol=0, rtol=0)

    def test_actual_8190_context_matches_complete_recording_after_weight_change(self):
        tcn = ZeroGatedTCN(2, 12)
        with torch.no_grad():
            tcn.output.weight.normal_(std=.1)
        controls = torch.tensor([[0., 1.], [.5, .5], [1., 0.]])
        signals = [torch.randn(13_000) * .03 for _ in range(3)]
        starts = [0, 133, 9_177]
        inputs = context_windows(signals, starts, 512, 8_190)
        before = [value.clone() for value in signals]
        with torch.inference_mode():
            window = tcn(inputs, controls)[0][:, 8_190:]
            complete = torch.cat([tcn(signal[None], control[None])[0][:, start:start + 512]
                                  for signal, control, start in zip(signals, controls, starts)])
        torch.testing.assert_close(window, complete, atol=2e-6, rtol=0)
        for left, right in zip(signals, before):
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_generic_wrapper_streams_and_is_zero_for_dynamic_controls(self):
        tcn = ZeroGatedTCN(3, 12)
        with torch.no_grad():
            tcn.output.weight.normal_(std=.05)
        model = GenericLongTCNResidual(StatelessBase(), tcn).eval()
        dry, controls = torch.randn(2, 10_337) * .03, torch.rand(2, 10_337, 2)
        with torch.inference_mode():
            whole, _ = model(dry, controls)
            state, chunks = None, []
            for start, stop in ((0, 17), (17, 513), (513, 5_121), (5_121, 10_337)):
                value, state = model(dry[:, start:stop], controls[:, start:stop], state)
                chunks.append(value)
            zero, _ = model(torch.zeros_like(dry), controls)
        torch.testing.assert_close(whole, torch.cat(chunks, 1), atol=2e-6, rtol=0)
        self.assertEqual(int(torch.count_nonzero(zero)), 0)
        self.assertEqual(len(state), 13)


if __name__ == "__main__":
    unittest.main()
