import unittest

import torch

from .multirate_fuzz import BLOCK, MultirateFuzz
from .train_dfz_long_tcn import context_windows


class MultirateFuzzTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(989)
        self.model = MultirateFuzz().eval()
        with torch.no_grad():
            self.model.audio.output.weight.normal_(0, .05)

    @torch.inference_mode()
    def test_irregular_stream_dynamic_control_and_clock(self):
        dry, controls = torch.randn(2, 6173) * .03, torch.rand(2, 6173, 2)
        whole, end = self.model(dry, controls)
        state, chunks, start = None, [], 0
        for stop in (1, 17, 63, 64, 65, 126, 129, 512, 1025, 2171, 6173):
            value, state = self.model(dry[:, start:stop], controls[:, start:stop], state)
            self.assertEqual(int(state[-1]), stop % BLOCK)
            chunks.append(value)
            start = stop
        self.assertLessEqual(float((whole - torch.cat(chunks, 1)).abs().max()), 2e-6)
        self.assertTrue(torch.equal(state[-3], end[-3]))
        self.assertLessEqual(max(float((a - b).abs().max()) for a, b in zip(state[:-1], end[:-1])), 2e-6)

    @torch.inference_mode()
    def test_zero_with_nonzero_slow_history_and_dynamic_controls(self):
        dry, controls = torch.zeros(2, 4099), torch.rand(2, 4099, 2)
        state = list(self.model.initial_state(dry))
        for index in range(2 * self.model.fast_count, 2 * len(self.model.state_widths)):
            state[index].normal_(0, .1)
        state[-2].normal_(0, .1)
        value, _ = self.model(dry, controls, tuple(state))
        self.assertEqual(float(value.abs().max()), 0.)

    @torch.inference_mode()
    def test_excitation_tail_expires_after_finite_audio_history(self):
        prefix = torch.randn(2, 1033) * .03
        dry = torch.cat((prefix, torch.zeros(2, 4099)), 1)
        controls = torch.rand(2, dry.shape[1], 2)
        whole, _ = self.model(dry, controls)
        cutoff = prefix.shape[1] + self.model.audio.receptive_field - 1
        self.assertGreater(float(whole[:, :prefix.shape[1]].abs().max()), 0.)
        self.assertEqual(float(whole[:, cutoff:].abs().max()), 0.)

    @torch.inference_mode()
    def test_no_future_within_block_or_control_path(self):
        dry, controls = torch.randn(1, 1033) * .03, torch.rand(1, 1033, 2)
        expected, _ = self.model(dry, controls)
        for cutoff in (1, 17, 63, 64, 65, 527):
            changed, knobs = dry.clone(), controls.clone()
            changed[:, cutoff:] += .4
            knobs[:, cutoff:] = 1 - knobs[:, cutoff:]
            actual, _ = self.model(changed, knobs)
            self.assertTrue(torch.equal(expected[:, :cutoff], actual[:, :cutoff]))

    @torch.inference_mode()
    def test_complete_training_history_matches_stream(self):
        dry, controls = torch.randn(2, 8192) * .03, torch.tensor([[0., .5], [1., 0.]])
        expected, _ = self.model(dry, controls)
        held = self.model.training_held(dry, controls)
        for starts in ([0, 17], [1024, 3141], [5001, 6000]):
            frames, left = 257, self.model.audio.receptive_field - 1
            windows = context_windows(list(dry), starts, frames, left)
            indices = (torch.tensor(starts)[:, None] - left + torch.arange(left + frames)[None]).clamp_min(0) // BLOCK
            value, _ = self.model.audio(windows, controls[:, None].expand(-1, left + frames, -1), held, indices)
            target = torch.stack([expected[index, start:start + frames] for index, start in enumerate(starts)])
            self.assertLessEqual(float((value[:, left:] - target).abs().max()), 2e-6)

    def test_integer_clock_and_input_rejections(self):
        dry, controls = torch.zeros(1, 1), torch.zeros(1, 1, 2)
        state = list(self.model.initial_state(dry))
        state[-1] = torch.tensor(0.)
        with self.assertRaises(ValueError):
            self.model(dry, controls, tuple(state))
        with self.assertRaises(ValueError):
            self.model(dry.double(), controls)
        with self.assertRaises(ValueError):
            self.model(dry, controls + 2)

    def test_training_has_no_future_gradient(self):
        dry = (torch.randn(1, 4096) * .03).requires_grad_()
        controls = torch.tensor([[.2, .8]])
        held = self.model.training_held(dry, controls)
        start, frames = 2147, 73
        left = self.model.audio.receptive_field - 1
        window = dry[:, start - left:start + frames]
        indices = (torch.arange(window.shape[1]) + start - left) // BLOCK
        prediction, _ = self.model.audio(window, controls[:, None].expand(-1, window.shape[1], -1), held, indices)
        prediction[:, left:].square().sum().backward()
        self.assertGreater(float(dry.grad[:, :start + frames].abs().sum()), 0.)
        self.assertEqual(float(dry.grad[:, start + frames:].abs().sum()), 0.)

    @torch.inference_mode()
    def test_inputs_and_weights_unchanged_and_batch_independent(self):
        dry, controls = torch.randn(3, 1091) * .03, torch.rand(3, 1091, 2)
        saved_dry, saved_controls = dry.clone(), controls.clone()
        weights = {key: value.clone() for key, value in self.model.state_dict().items()}
        together, _ = self.model(dry, controls)
        pieces = torch.cat([self.model(dry[index:index + 1], controls[index:index + 1])[0] for index in range(3)], 0)
        self.assertLessEqual(float((together - pieces).abs().max()), 2e-6)
        self.assertTrue(torch.equal(dry, saved_dry))
        self.assertTrue(torch.equal(controls, saved_controls))
        self.assertTrue(all(torch.equal(value, weights[key]) for key, value in self.model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
