"""Small CPU proofs for the frozen-prefix DFZ training experiment."""
from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from .stable_effect import StableEffectRenderer
from .stable_nonlinear_readout import NonlinearReadout, StableNonlinearReadout
from .train_dfz_transients import project, combine_mirrored_gate_gradients
from .train_dfz_last_layer import (early_digest, encode_early, freeze_early,
                                  grouped_paths, last_layer_view, project_last_layer,
                                  render_cached, selection_key, training_windows)


class LastLayerCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(964)
        base = StableEffectRenderer(2, 4, 4)
        project(base, mirror_gates=True)
        readout = NonlinearReadout(4, 3)
        torch.nn.init.normal_(readout.output, std=.05)
        self.model = StableNonlinearReadout(base, readout)
        freeze_early(self.model)

    def test_cached_prefix_matches_full_model_after_last_layer_update(self):
        dry = torch.randn(2, 311)*.03
        controls = torch.tensor([[0., .5], [1., 0.]])
        frozen = early_digest(self.model)
        early, _ = encode_early(self.model, dry, controls)
        full, _ = self.model(dry, controls)
        cached, _ = render_cached(self.model, early, controls)
        torch.testing.assert_close(full, cached, atol=2e-7, rtol=0)
        optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad], lr=.01)
        optimizer.zero_grad(); (cached-.8*full.detach()).square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in self.model.base.rnn_layers[-1].parameters()))
        self.assertTrue(all(p.grad is None for layer in self.model.base.rnn_layers[:-1] for p in layer.parameters()))
        combine_mirrored_gate_gradients(last_layer_view(self.model)); optimizer.step(); project_last_layer(self.model)
        self.assertEqual(frozen, early_digest(self.model))
        updated, _ = self.model(dry, controls)
        still_cached, _ = render_cached(self.model, early, controls)
        self.assertGreater(float((updated-full).abs().max().detach()), 0)
        torch.testing.assert_close(updated, still_cached, atol=2e-7, rtol=0)

    def test_different_full_prefix_lengths_match_unpadded_current_model(self):
        dry = torch.randn(2, 277)*.04
        controls = torch.tensor([[0., 1.], [.5, .5]])
        early, _ = encode_early(self.model, dry, controls)
        # Mutate only the final recurrence after the early cache was made.
        with torch.no_grad():
            self.model.base.rnn_layers[-1].weight_ih_l0.add_(torch.randn_like(self.model.base.rnn_layers[-1].weight_ih_l0)*.002)
        project_last_layer(self.model)
        expected, _ = self.model(dry, controls)
        rows = [{'dry': early[i], 'wet': expected[i].detach(), 'controls': controls[i]} for i in range(2)]
        predicted, target = training_windows(self.model, rows, [0, 1], [17, 133], 101, torch.device('cpu'), chunk_frames=64)
        torch.testing.assert_close(predicted, target, atol=3e-7, rtol=0)
        predicted.square().mean().backward()
        self.assertTrue(all(p.grad is None for layer in self.model.base.rnn_layers[:-1] for p in layer.parameters()))

    def test_projection_never_changes_cached_early_layers(self):
        before = early_digest(self.model)
        last = self.model.base.rnn_layers[-1]
        with torch.no_grad():
            last.weight_hh_l0.mul_(3)
            last.bias_ih_l0.add_(.2)
        norms = project_last_layer(self.model)
        self.assertEqual(before, early_digest(self.model))
        self.assertLessEqual(norms[0], .995001)
        h = self.model.base.hidden_size
        torch.testing.assert_close(last.weight_hh_l0[:h], -last.weight_hh_l0[h:2*h], atol=0, rtol=0)
        self.assertEqual(float(last.bias_ih_l0[2*h:3*h].abs().max().detach()), 0)

    def test_groups_cover_every_fit_example_with_all_nine_controls(self):
        paths = [Path(f'{a},{b},{index}.wav') for a in (0, 50, 100) for b in (0, 50, 100)
                 for index in range(1, 33) if index % 5]
        groups = grouped_paths(paths, np.random.default_rng(1), 18)
        self.assertEqual(len(paths), 234)
        self.assertEqual(len(groups), 13)
        self.assertEqual({p for group in groups for p in group}, set(paths))
        for group in groups:
            self.assertEqual(len(group), 18)
            self.assertEqual(len({tuple(p.stem.split(',')[:2]) for p in group}), 9)

    def test_selection_prefers_complete_gate_not_only_lowest_peak(self):
        failing = {'passes_selection_gate': False, 'absolute_peak_error_p95': .018, 'global_esr': .009}
        passing = {'passes_selection_gate': True, 'absolute_peak_error_p95': .019, 'global_esr': .009}
        self.assertLess(selection_key(passing), selection_key(failing))


if __name__ == '__main__':
    unittest.main()
