"""Select a fixed convex DFZ mixture on calibration, never per-clip gain fitting."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .asrnn_effects import effect_files, read_effect_pair
from .fit_asrnn_effect_output import _partition
from .stable_effect import load_stable_effect, stable_effect_from_payload
from .train_asrnn_phase7 import _evaluate, _rows


def simplex(count, denominator):
    for values in itertools.product(range(denominator + 1), repeat=count - 1):
        remaining = denominator - sum(values)
        if remaining >= 0:
            yield (*values, remaining)


@torch.inference_mode()
def predictions(model, pairs, compute):
    result = []
    model = model.to(compute)
    for offset in range(0, len(pairs), 4):
        batch = pairs[offset:offset + 4]
        dry = torch.from_numpy(np.stack([p[0] for p in batch])).to(compute)
        controls = torch.from_numpy(np.stack([p[2] for p in batch])).to(compute)
        chunks, state = [], None
        for start in range(0, dry.shape[1], 2048):
            value, state = model(dry[:, start:start + 2048], controls, state)
            chunks.append(value)
        result.append(torch.cat(chunks, 1)[:, 1024:].cpu())
    model.cpu()
    if compute == 'mps':
        torch.mps.empty_cache()
    return torch.cat(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--members', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--denominator', type=int, default=10)
    parser.add_argument('--compute', choices=('cpu', 'mps'), default='mps')
    args = parser.parse_args()
    if args.output.exists() or not 2 <= len(args.members) <= 5 or not 1 <= args.denominator <= 20:
        raise ValueError('new output and a bounded two-to-five-member pool required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    _, paths = _partition(effect_files(Path('data/corpus/asrnn-physical-effects'), 'dfz', 'train'))
    pairs = [read_effect_pair(path, 'dfz') for path in paths]
    wet = torch.from_numpy(np.stack([p[1][1024:] for p in pairs]))
    outputs, payloads, sources = [], [], []
    for path in args.members:
        model, payload = load_stable_effect(path)
        if payload.get('device') != 'dfz' or payload.get('schema') not in (2, 7, 8):
            raise ValueError('incompatible member')
        payloads.append(payload)
        outputs.append(predictions(model, pairs, args.compute))
        sources.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print(json.dumps({'stage': 'ensemble-calibration', 'members': len(outputs), 'total': len(args.members)}), flush=True)
    outputs = torch.stack(outputs)
    # Peak candidates include each member's high-amplitude samples. Full CPU
    # evaluation below is mandatory: sampled extrema alone cannot admit a model.
    uniform = torch.linspace(0, wet.shape[1] - 1, 2048).long().expand(len(pairs), -1)
    ids = torch.cat((uniform, wet.abs().topk(256, 1).indices, *(v.abs().topk(256, 1).indices for v in outputs)), 1)
    sampled = torch.stack([v.gather(1, ids) for v in outputs])
    target = wet.gather(1, ids)
    target_peak = wet.abs().amax(1)
    energy = wet.square().mean(1).clamp_min(1e-8)
    rows = []
    for integers in simplex(len(outputs), args.denominator):
        weights = torch.tensor(integers) / args.denominator
        prediction = torch.einsum('m,mbt->bt', weights, sampled)
        peak_errors = (prediction.abs().amax(1) - target_peak).abs()
        esr = (prediction[:, :2048] - target[:, :2048]).square().mean(1) / energy
        rows.append({'weights': weights.tolist(), 'sampled_peak_error_p95': float(torch.quantile(peak_errors, .95)), 'sampled_mean_esr': float(esr.mean())})
    selected = min(rows, key=lambda row: row['sampled_peak_error_p95'] + max(0., row['sampled_mean_esr'] - .03))
    positive = [(payload, weight) for payload, weight in zip(payloads, selected['weights']) if weight > 0]
    if len(positive) == 1:
        payload = positive[0][0]
    else:
        payload = {'schema': 9, 'architecture': 'stable-dfz-composite-ensemble', 'device': 'dfz', 'sample_rate': 48000,
                   'members': [p for p, w in positive], 'weights': [w for p, w in positive]}
    complete = _evaluate(stable_effect_from_payload(payload), _rows(paths, 'dfz'), torch.device('cpu'), 8)
    report = {'schema': 1, 'sources': sources, 'selected': selected, 'candidates': rows, 'calibration': complete,
              'admitted_for_official_eval': complete['passes_selection_gate'], 'calibration_examples': len(paths),
              'calibration_compute': 'cpu', 'weights_fixed_before_development': True, 'official_eval_used_for_selection': False,
              'physical_audio_devices_used': False, 'source_audio_modified': False, 'automatic_normalization': False, 'automatic_limiting': False}
    if complete['passes_selection_gate']:
        checkpoint = args.output / 'candidate.pt'
        torch.save(payload, checkpoint)
        report['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'candidates'}), flush=True)


if __name__ == '__main__':
    main()
