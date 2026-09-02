import unittest

import torch

from .cascaded_multirate_fuzz import CascadedMultirateFuzz, GEOMETRY
from .precheck_multirate_fuzz import streamed


class CascadedCoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1010)
        torch.set_num_threads(1)
        self.model = CascadedMultirateFuzz(stage_widths=(4, 2), blocks_per_stage=3, skip_width=3,
                                         controller_width=2, controller_blocks=3).eval()
        with torch.no_grad():
            self.model.audio.output.weight.normal_(0, .1)

    def test_variable_width_history_and_irregular_stream(self):
        dry, controls = torch.randn(2, 1033)*.03, torch.rand(2, 1033, 2)
        with torch.inference_mode():
            whole, state = self.model(dry, controls)
            chunks, streamed_state = streamed(self.model, dry, controls, [1, 17, 63, 64, 257])
        self.assertLess(float((whole-chunks).abs().max()), 2e-6)
        self.assertTrue(torch.equal(state[-1], streamed_state[-1]))
        self.assertEqual(self.model.state_widths[:6], [4, 8, 16, 2, 4, 8])

    def test_zero_with_nonzero_slow_history_and_finite_tail(self):
        dry, controls = torch.zeros(2, 1033), torch.rand(2, 1033, 2)
        state = list(self.model.initial_state(dry))
        for tensor in state[2*self.model.fast_count:-3]:
            tensor.normal_()
        state[-2].normal_()
        with torch.inference_mode():
            zero = self.model(dry, controls, tuple(state))[0]
            excited = dry.clone()
            excited[:, :100].normal_(0, .03)
            rendered = self.model(excited, controls)[0]
        self.assertEqual(float(zero.abs().max()), 0.)
        self.assertEqual(float(rendered[:, 100+self.model.audio.receptive_field-1:].abs().max()), 0.)

    def test_future_audio_and_controls_cannot_change_past(self):
        dry, controls = torch.randn(2, 513)*.03, torch.rand(2, 513, 2)
        with torch.inference_mode():
            before = self.model(dry, controls)[0]
            dry[:, 79:] += .3
            controls[:, 79:] = 1-controls[:, 79:]
            after = self.model(dry, controls)[0]
        self.assertTrue(torch.equal(before[:, :79], after[:, :79]))

    def test_full_training_context_matches_whole_causal_recording(self):
        model = self.model
        dry, knobs = torch.randn(2, 2048)*.03, torch.rand(2, 2)
        starts = torch.tensor([768, 1024])
        left, count = model.audio.receptive_field-1, 257
        times = starts[:, None]-left+torch.arange(left+count)[None]
        window = torch.stack([row[start-left:start+count] for row, start in zip(dry, starts)])
        with torch.inference_mode():
            complete = model(dry, knobs)[0]
            held = model.training_held(dry, knobs)
            predicted = model.audio(window, knobs[:, None].expand(-1, left+count, -1), held, times//64)[0][:, left:]
        expected = torch.stack([row[start:start+count] for row, start in zip(complete, starts)])
        self.assertLess(float((predicted-expected).abs().max()), 2e-6)

    def test_reviewed_geometry_and_initial_zero_projection(self):
        model = CascadedMultirateFuzz(**GEOMETRY)
        self.assertEqual(model.audio.receptive_field, 4093)
        self.assertEqual(model.fast_count, 20)
        self.assertEqual(sum(v.numel()*v.element_size() for v in model.initial_state(torch.zeros(1, 1))), 262184)
        self.assertEqual(int(model.audio.output.weight.count_nonzero()), 0)
        self.assertEqual(len(model.audio.raw_mixins), 20)
        self.assertEqual(len(model.audio.skip_heads), 20)

    def test_full_geometry_training_prefix_including_padded_start(self):
        from .train_dfz_long_tcn import context_windows
        model = CascadedMultirateFuzz(**GEOMETRY).eval()
        dry, knobs = torch.randn(2, 16384)*.03, torch.rand(2, 2)
        starts, count = [1024, 8192], 4096
        left = model.audio.receptive_field-1
        times = torch.tensor(starts)[:, None]-left+torch.arange(left+count)[None]
        window = context_windows(list(dry), starts, count, left)
        with torch.inference_mode():
            model.audio.output.weight.normal_(0, .05)
            complete = model(dry, knobs)[0]
            held = model.training_held(dry, knobs)
            predicted = model.audio(window, knobs[:, None].expand(-1, left+count, -1), held,
                                    times.clamp_min(0)//64)[0][:, left:]
        expected = torch.stack([row[start:start+count] for row, start in zip(complete, starts)])
        self.assertLess(float((predicted-expected).abs().max()), 2e-6)


if __name__ == "__main__":
    unittest.main()
