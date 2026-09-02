import unittest

import torch

from .centered_multirate_fuzz import CenteredMultirateFuzz
from .multirate_fuzz import MultirateFuzz
from .precheck_multirate_fuzz import streamed
from .train_dfz_long_tcn import context_windows


class CenteredMultirateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1000)
        self.base = MultirateFuzz().eval()
        with torch.no_grad():
            self.base.audio.output.weight.normal_(0, .05)
        self.model = CenteredMultirateFuzz.from_base(self.base)

    @torch.inference_mode()
    def test_exact_zero_head_initialization(self):
        dry, knobs = torch.randn(2, 6173) * .03, torch.rand(2, 6173, 2)
        self.assertTrue(torch.equal(self.base(dry, knobs)[0], self.model(dry, knobs)[0]))
        self.assertEqual(self.base.state_widths, self.model.state_widths)

    def nonzero_heads(self):
        with torch.no_grad():
            for layer in (*self.model.audio.activation_current, *self.model.audio.activation_slow):
                layer.weight.normal_(0, .3)
                if layer.bias is not None:
                    layer.bias.normal_(0, .3)

    @torch.inference_mode()
    def test_zero_origin_even_with_nonzero_thresholds_and_slow_history(self):
        self.nonzero_heads()
        dry, knobs = torch.zeros(2, 6173), torch.rand(2, 6173, 2)
        state = list(self.model.initial_state(dry))
        for value in state[2 * self.model.fast_count:-3]:
            value.normal_(0, .1)
        state[-2].normal_(0, .3)
        self.assertEqual(float(self.model(dry, knobs, tuple(state))[0].abs().max()), 0.)

    @torch.inference_mode()
    def test_nonzero_thresholds_stream_and_finite_tail(self):
        self.nonzero_heads()
        dry = torch.cat((torch.randn(2, 1033) * .03, torch.zeros(2, 4099)), 1)
        knobs = torch.rand(2, dry.shape[1], 2)
        whole, _ = self.model(dry, knobs)
        chunks, _ = streamed(self.model, dry, knobs, [1, 17, 63, 64, 257, 1024])
        self.assertLessEqual(float((whole - chunks).abs().max()), 2e-6)
        self.assertEqual(float(whole[:, 1033 + self.model.audio.receptive_field - 1:].abs().max()), 0.)

    @torch.inference_mode()
    def test_full_training_history_and_causality(self):
        self.nonzero_heads()
        dry, controls = torch.randn(2, 8192) * .03, torch.tensor([[0., .5], [1., 0.]])
        whole = self.model(dry, controls)[0]
        held = self.model.training_held(dry, controls)
        starts, frames, left = [0, 3141], 257, self.model.audio.receptive_field - 1
        windows = context_windows(list(dry), starts, frames, left)
        times = (torch.tensor(starts)[:, None] - left + torch.arange(left + frames)[None]).clamp_min(0) // 64
        actual = self.model.audio(windows, controls[:, None].expand(-1, windows.shape[1], -1), held, times)[0][:, left:]
        expected = torch.stack([whole[i, start:start + frames] for i, start in enumerate(starts)])
        self.assertLessEqual(float((actual - expected).abs().max()), 2e-6)
        changed = dry.clone()
        changed[:, 17:] += .3
        self.assertTrue(torch.equal(whole[:, :17], self.model(changed, controls)[0][:, :17]))

    def test_new_threshold_heads_receive_gradient(self):
        dry, controls = torch.randn(2, 1033) * .03, torch.rand(2, 1033, 2)
        output = self.model(dry, controls)[0]
        (output - torch.sin(dry * 7)).square().mean().backward()
        gradients = [layer.weight.grad for layer in self.model.audio.activation_current]
        self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in gradients))
        self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0.)

    def test_head_only_warmup_preserves_every_existing_weight(self):
        from .dfz_temporal_basis import attach_temporal_basis, materialized_state_dict
        from .train_centered_multirate_fuzz import set_head_only
        model = attach_temporal_basis(self.model)
        before = materialized_state_dict(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        set_head_only(model, True)
        dry, controls = torch.randn(2, 1033) * .03, torch.rand(2, 1033, 2)
        (model(dry, controls)[0] - torch.sin(dry * 7)).square().mean().backward()
        optimizer.step()
        after = materialized_state_dict(model)
        existing = [name for name in before if ".activation_current." not in name and ".activation_slow." not in name]
        self.assertTrue(all(torch.equal(before[name], after[name]) for name in existing))
        self.assertTrue(any(not torch.equal(before[name], after[name]) for name in before if name not in existing))
        set_head_only(model, False)
        self.assertTrue(all(value.requires_grad for value in model.parameters()))

    def test_fixed_candidate_learning_rate_schedule(self):
        from .train_centered_multirate_fuzz import learning_rate
        self.assertAlmostEqual(learning_rate(1000), 3e-4)
        self.assertAlmostEqual(learning_rate(1001), 2e-4)
        self.assertAlmostEqual(learning_rate(12000), 1e-5)
        with self.assertRaises(ValueError):
            learning_rate(12001)


if __name__ == "__main__":
    unittest.main()
