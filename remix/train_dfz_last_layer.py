"""Bounded DFZ last-layer adaptation with exact frozen-prefix feature caches.

Only the final LSTM, original output layer, and existing nonlinear readout
change. The first three LSTMs are cached as complete float32 trajectories;
no state of a trainable recurrence is ever cached across optimizer steps.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as functional

from .asrnn_effects import effect_files
from .fit_asrnn_effect_output import _partition
from .stable_effect import load_stable_effect
from .train_asrnn_phase7 import _evaluate, _rows
from .train_dfz_transients import combine_mirrored_gate_gradients, project


def freeze_early(model):
    if len(model.base.rnn_layers) != 4:
        raise ValueError('this experiment requires exactly four recurrent layers')
    for layer in model.base.rnn_layers[:-1]:
        layer.requires_grad_(False)


def early_digest(model):
    digest = hashlib.sha256()
    digest.update(str((model.base.input_coef, model.base.control_count)).encode())
    for index, layer in enumerate(model.base.rnn_layers[:-1]):
        for name, value in layer.state_dict().items():
            digest.update(f'{index}:{name}'.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def last_layer_view(model):
    """Give the existing constraint helpers access to only the mutable layer."""
    return SimpleNamespace(hidden_size=model.base.hidden_size,
                           control_count=model.base.control_count,
                           rnn_layers=[model.base.rnn_layers[-1]])


def project_last_layer(model):
    return project(last_layer_view(model), mirror_gates=True)


@torch.no_grad()
def encode_early(model, dry, controls, state=None):
    base = model.base
    condition = base._official_controls(controls, dry.shape[1])
    value = dry[..., None] * base.input_coef
    previous = [None] * 3 if state is None else state
    following = []
    for layer, old in zip(base.rnn_layers[:-1], previous):
        value, new = layer(torch.cat((value, condition), -1), old)
        following.append(new)
    return value, tuple(following)


def last_hidden(model, features, controls, state=None):
    condition = model.base._official_controls(controls, features.shape[1])
    return model.base.rnn_layers[-1](torch.cat((features, condition), -1), state)


def render_cached(model, features, controls, state=None, *, training_knots=False):
    hidden, state = last_hidden(model, features, controls, state)
    correction = (model.readout.at_training_knots(hidden, controls) if training_knots
                  else model.readout(hidden, controls))
    return model.base.output_layer(hidden).squeeze(-1) + correction, state


class CachedTailRenderer(torch.nn.Module):
    """Adapter to reuse the unchanged complete-clip calibration gate."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, controls, state=None):
        return render_cached(self.model, features, controls, state)


@torch.no_grad()
def cache_early(model, paths, target, *, label, batch_size=4, chunk_frames=2048):
    started = time.perf_counter()
    rows = _rows(paths, 'dfz')
    frames = len(rows[0]['dry'])
    if any(len(row['dry']) != frames for row in rows):
        raise ValueError('cache requires equal-length complete recordings')
    # CPU allocation only; at most four short feature blocks occupy the GPU.
    cache = torch.empty(len(rows), frames, model.base.hidden_size, dtype=torch.float32)
    cached_rows = []
    for offset in range(0, len(rows), batch_size):
        selected = rows[offset:offset+batch_size]
        dry = torch.stack([row['dry'] for row in selected]).to(target)
        controls = torch.stack([row['controls'] for row in selected]).to(target)
        state = None
        for start in range(0, frames, chunk_frames):
            value, state = encode_early(model, dry[:, start:start+chunk_frames], controls, state)
            cache[offset:offset+len(selected), start:start+value.shape[1]].copy_(value.cpu())
        for index, row in enumerate(selected, offset):
            signal = row['dry'].numpy()
            active = np.flatnonzero(np.abs(signal) > max(1e-4, float(np.abs(signal).max())*.01))
            onset = int(active[0]) if len(active) else 1024
            cached_rows.append({**row, 'dry': cache[index], 'onset': onset})
        print(json.dumps({'stage': 'frozen-early-cache', 'partition': label,
                          'completed': len(cached_rows), 'total': len(rows),
                          'cache_bytes': cache.numel()*cache.element_size(),
                          'elapsed_seconds': time.perf_counter()-started}), flush=True)
    return cached_rows, cache.numel()*cache.element_size()


def training_windows(model, cached_rows, indices, starts, frames, target, *, chunk_frames=2048):
    """Recompute every current final-layer state from sample zero, then TBPTT.

    Left-padding aligns different prefix lengths in a batch. The current
    zero-origin constraints make padded zero-feature history exactly inert.
    Only the subsequent gradient window is recorded by autograd.
    """
    selected = [cached_rows[int(index)] for index in indices]
    if len(selected) != len(starts) or any(start < 0 or start+frames > len(row['wet'])
                                         for row, start in zip(selected, starts)):
        raise ValueError('invalid complete-prefix training windows')
    controls = torch.stack([row['controls'] for row in selected]).to(target)
    history_frames = ((max(starts, default=0)+chunk_frames-1)//chunk_frames)*chunk_frames
    state = None
    with torch.no_grad():
        if history_frames:
            history = torch.zeros(len(selected), history_frames, model.base.hidden_size)
            for index, (row, start) in enumerate(zip(selected, starts)):
                if start:
                    history[index, history_frames-start:].copy_(row['dry'][:start])
            for start in range(0, history_frames, chunk_frames):
                _, state = last_hidden(model, history[:, start:start+chunk_frames].to(target), controls, state)
    features = torch.stack([row['dry'][start:start+frames] for row, start in zip(selected, starts)]).to(target)
    expected = torch.stack([row['wet'][start:start+frames] for row, start in zip(selected, starts)]).to(target)
    prediction, _ = render_cached(model, features, controls, state, training_knots=True)
    return prediction, expected


def grouped_paths(paths, rng, group_size=18):
    buckets = {}
    for path in paths:
        buckets.setdefault(tuple(path.stem.split(',')[:2]), []).append(path)
    if len(buckets) != 9 or group_size % 9:
        raise ValueError('groups must contain equal samples from all nine control settings')
    for bucket in buckets.values():
        rng.shuffle(bucket)
    per_control = group_size//9
    groups = []
    for start in range(0, max(map(len, buckets.values())), per_control):
        groups.append([path for key in sorted(buckets) for path in buckets[key][start:start+per_control]])
    rng.shuffle(groups)
    return groups


def candidate_payload(model, original):
    payload = copy.deepcopy(original)
    payload['base']['state_dict'] = {key: value.detach().cpu().clone()
                                     for key, value in model.base.state_dict().items()}
    payload['readout_state_dict'] = {key: value.detach().cpu().clone()
                                      for key, value in model.readout.state_dict().items()}
    return payload


def loss_per_example(prediction, expected):
    error = prediction-expected
    waveform = error.square().mean(1)/expected.square().mean(1).clamp_min(1e-6)
    dp, dt = prediction[:, 1:]-.95*prediction[:, :-1], expected[:, 1:]-.95*expected[:, :-1]
    emphasized = (dp-dt).square().mean(1)/dt.square().mean(1).clamp_min(1e-6)
    envelope = (functional.max_pool1d(prediction.abs()[:, None], 128, 64)
                - functional.max_pool1d(expected.abs()[:, None], 128, 64)).abs().mean((1, 2))
    signed_extrema = .5*((prediction.amax(1)-expected.amax(1)).abs()
                          + (prediction.amin(1)-expected.amin(1)).abs())
    peak = (prediction.abs().amax(1)-expected.abs().amax(1)).abs()
    return waveform + .25*emphasized + 4*envelope + 12*signed_extrema + 4*peak


def selection_key(metrics):
    # A passing worst-group gate must outrank a lower overall P95 that fails it.
    return (not metrics['passes_selection_gate'],
            metrics['absolute_peak_error_p95']+max(0., metrics['global_esr']-.03))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('remix/runs/dfz-nonlinear-readout-phase9/candidate.pt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--group-size', type=int, choices=(18, 27), default=18)
    parser.add_argument('--group-updates', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--frames', type=int, choices=(4096, 6144), default=6144)
    parser.add_argument('--calibration-interval', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--readout-learning-rate', type=float, default=3e-5)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite an existing experiment')
    if min(args.steps, args.group_updates, args.batch_size, args.calibration_interval) <= 0:
        raise ValueError('positive bounded training dimensions required')
    torch.set_num_threads(2); torch.manual_seed(963); rng = np.random.default_rng(963)
    target = torch.device('mps')
    if not torch.backends.mps.is_available():
        raise RuntimeError('this bounded full-prefix experiment requires MPS compute')
    source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    model, original = load_stable_effect(args.source)
    if original.get('schema') != 7 or original.get('device') != 'dfz':
        raise ValueError('requires the existing DFZ nonlinear-readout source')
    freeze_early(model); frozen_sha = early_digest(model); model.to(target)
    optimizer = torch.optim.AdamW([
        {'params': model.base.rnn_layers[-1].parameters(), 'lr': args.learning_rate},
        {'params': model.base.output_layer.parameters(), 'lr': args.learning_rate},
        {'params': model.readout.parameters(), 'lr': args.readout_learning_rate},
    ], weight_decay=1e-6)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    anchors = [parameter.detach().clone() for parameter in trainable]
    fit, cal = _partition(effect_files(Path('data/corpus/asrnn-physical-effects'), 'dfz', 'train'))
    if len(fit) != 234 or len(cal) != 54:
        raise ValueError('frozen DFZ fit/calibration geometry changed')
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    cal_cache, cal_bytes = cache_early(model, cal, target, label='calibration')
    if cal_bytes > 2_100_000_000:
        raise ValueError('calibration cache exceeded the memory budget')
    initial = _evaluate(CachedTailRenderer(model), cal_cache, target, 4)
    history = [{'step': 0, **initial}]
    best_score = selection_key(initial)
    best_step = 0
    torch.save(original, args.output/'candidate.pt')
    print(json.dumps({'calibration': history[0], 'elapsed_seconds': time.perf_counter()-started}), flush=True)
    groups = []; group_index = 0; fit_cache = None; group_bytes = 0; covered = set(); updates = []

    def progress():
        if early_digest(model) != frozen_sha:
            raise RuntimeError('frozen early-layer weights changed; cache is invalid')
        return {
            'schema': 1, 'history': history, 'updates': updates, 'best_step': best_step,
            'source_sha256': source_sha, 'frozen_early_sha256': frozen_sha,
            'fit_examples': len(fit), 'calibration_examples': len(cal),
            'fit_files_covered': sorted(covered), 'fit_coverage': len(covered),
            'calibration_cache_bytes': cal_bytes, 'largest_fit_group_cache_bytes': group_bytes,
            'cached_dtype': 'float32', 'cache_storage': 'RAM only',
            'frozen_early_layers': 3, 'trainable_recurrent_layers': 1,
            'mutable_recurrent_state_cached_between_steps': False,
            'current_weight_full_causal_prefix_replayed': True,
            'learning_rate': args.learning_rate, 'readout_learning_rate': args.readout_learning_rate,
            'gradient_window_frames': args.frames, 'batch_size': args.batch_size,
            'requested_steps': args.steps, 'completed_steps': updates[-1]['step'] if updates else 0,
            'elapsed_seconds': time.perf_counter()-started,
            'admitted_for_official_eval': False, 'official_eval_used_for_selection': False,
            'physical_audio_devices_used': False, 'source_audio_modified': False,
            'automatic_normalization': False, 'automatic_limiting': False,
        }

    (args.output/'training.json').write_text(json.dumps(progress(), indent=2)+'\n')
    for step in range(1, args.steps+1):
        if (step-1) % args.group_updates == 0:
            if fit_cache is not None:
                del fit_cache; fit_cache = None; gc.collect(); torch.mps.empty_cache()
            if group_index == len(groups):
                groups = grouped_paths(fit, rng, args.group_size); group_index = 0
            paths = groups[group_index]; group_index += 1
            fit_cache, size = cache_early(model, paths, target, label=f'fit-step-{step}')
            group_bytes = max(group_bytes, size)
            if size > 1_000_000_000:
                raise ValueError('fit-group cache exceeded the memory budget')
            covered.update(path.name for path in paths)
        indices = rng.integers(0, len(fit_cache), args.batch_size)
        starts = []
        for index in indices:
            row = fit_cache[int(index)]
            if rng.random() < .7:
                peak = int(row['wet'][1024:].abs().argmax())+1024
                start = peak-int(rng.integers(512, args.frames-512))
            else:
                start = int(rng.integers(max(0, row['onset']-256), len(row['wet'])-args.frames+1))
            starts.append(max(0, min(len(row['wet'])-args.frames, start)))
        del row  # Do not retain a view into an old fit-group cache on group changes.
        step_started = time.perf_counter(); model.train()
        prediction, expected = training_windows(model, fit_cache, indices, starts, args.frames, target)
        per_file = loss_per_example(prediction, expected)
        anchor_loss = sum((value-anchor).square().mean() for value, anchor in zip(trainable, anchors))
        loss = .5*per_file.mean()+.5*per_file.topk(max(1, args.batch_size//3)).values.mean()+.01*anchor_loss
        optimizer.zero_grad(set_to_none=True); loss.backward()
        combine_mirrored_gate_gradients(last_layer_view(model))
        torch.nn.utils.clip_grad_norm_(trainable, 5.); optimizer.step(); norms = project_last_layer(model)
        row = {'step': step, 'loss': float(loss.detach()), 'step_seconds': time.perf_counter()-step_started,
               'elapsed_seconds': time.perf_counter()-started}
        updates.append(row)
        if step == 1 or step % 10 == 0:
            print(json.dumps({**row, 'fit_coverage': len(covered)}), flush=True)
            (args.output/'training.json').write_text(json.dumps(progress(), indent=2)+'\n')
        if step % args.calibration_interval == 0 or step == args.steps:
            measured = _evaluate(CachedTailRenderer(model), cal_cache, target, 4)
            measured = {'step': step, **measured}; history.append(measured)
            score = selection_key(measured)
            if score < best_score:
                best_score, best_step = score, step
                torch.save(candidate_payload(model, original), args.output/'candidate.pt')
            record = progress(); record['last_candidate_recurrent_infinity_norm'] = norms[0]
            (args.output/'training.json').write_text(json.dumps(record, indent=2)+'\n')
            print(json.dumps({'calibration': measured, 'best_step': best_step,
                              'elapsed_seconds': time.perf_counter()-started}), flush=True)
    record = progress()
    del cal_cache, fit_cache; gc.collect(); model.cpu(); torch.mps.empty_cache()
    print(json.dumps({'stage': 'gpu-released-cpu-final-audit',
                      'elapsed_seconds': time.perf_counter()-started}), flush=True)
    selected, _ = load_stable_effect(args.output/'candidate.pt')
    if early_digest(selected) != frozen_sha or hashlib.sha256(args.source.read_bytes()).hexdigest() != source_sha:
        raise RuntimeError('source/early weights changed during the experiment')
    verified = _evaluate(selected, _rows(cal, 'dfz'), torch.device('cpu'), 8)
    record.update(calibration=verified, calibration_compute='complete-original-model-cpu',
                  admitted_for_official_eval=verified['passes_selection_gate'],
                  checkpoint_sha256=hashlib.sha256((args.output/'candidate.pt').read_bytes()).hexdigest(),
                  elapsed_seconds=time.perf_counter()-started)
    (args.output/'metrics.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'cpu_calibration': verified, 'best_step': best_step,
                      'elapsed_seconds': time.perf_counter()-started}), flush=True)


if __name__ == '__main__':
    main()
