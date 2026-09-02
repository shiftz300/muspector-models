"""One preregistered fit-only CPU comparison of odd/even frozen-state features.

No original calibration/eval audio is opened. No training state/audio cache,
gain adjustment, device API, export, or accepted runtime change is made.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from .asrnn_effects import read_effect_pair
from .precheck_multirate_fuzz import sha256
from .stable_effect import load_stable_effect

SOURCE = Path('remix/runs/dfz-nonlinear-readout-phase9/candidate.pt')
MANIFEST = Path('remix/runs/dfz-centered-core-phase11/metrics.json')
SOURCE_HASH = '89640c9ee860544a8f26f408a883311657f47de60b7076d5315d00599890037a'
MODES = ('odd', 'odd_cubic', 'odd_even')
RIDGE, BUDGET_SECONDS = 1e-3, 900


def partition(records):
    """Original fit only; choose internal validation without any measurements."""
    if len(records) != 234 or len({r['path'] for r in records}) != 234:
        raise ValueError('unique original234-fit manifest required')
    train, held = [], []
    for record in records:
        p = Path(record['path'])
        if 'train' not in p.parts or p.suffix != '.wav':
            raise ValueError('only explicit original training WAVs allowed')
        first, second, take = map(int, p.stem.split(','))
        if first not in (0, 50, 100) or second not in (0, 50, 100) or take % 5 == 0:
            raise ValueError('record outside original fit partition')
        (held if take % 4 == 1 else train).append(record)
    for rows, count in ((train, 20), (held, 6)):
        coverage = Counter(tuple(Path(r['path']).stem.split(',')[:2]) for r in rows)
        if len(coverage) != 9 or set(coverage.values()) != {count}:
            raise ValueError('expected9-corner20/6 inner partition')
    return train, held


def feature_bank(hidden, projection):
    odd = torch.cat((hidden * 8, torch.tanh(hidden @ projection * 8)), -1)
    return torch.cat((odd, odd.pow(3), odd.square()), -1)


def columns(mode, width=80):
    if mode == 'odd':
        return np.arange(width)
    if mode == 'odd_cubic':
        return np.arange(width*2)
    if mode == 'odd_even':
        return np.r_[np.arange(width), np.arange(width*2, width*3)]
    raise ValueError('unknown parity arm')


def sampling(wet, original):
    """Same target-derived locations and mass for every arm; fit-only labels."""
    n = len(wet)
    uniform = np.linspace(1024, n-1, 2048).astype(np.int64)
    events = []
    for audio in (wet, original):
        for sign in (-1, 1):
            center = int(np.argmax(sign*audio[1024:]))+1024
            events.append(np.arange(max(1024, center-128), min(n, center+129)))
    events = np.concatenate(events)
    weights = np.r_[np.full(len(uniform), .5/len(uniform)), np.full(len(events), .5/len(events))]
    return np.r_[uniform, events], weights


def solve(xx, xy, mode):
    ids = columns(mode)
    matrix, target = xx[np.ix_(ids, ids)], xy[ids]
    scale = np.sqrt(np.maximum(np.diag(matrix), 1e-16))
    normalized = matrix / scale[:, None] / scale[None, :]
    result = np.linalg.solve(normalized + np.eye(len(ids))*RIDGE, target/scale)/scale
    folded = np.zeros(xx.shape[0], np.float32)
    folded[ids] = result.astype(np.float32)
    if not np.isfinite(folded).all():
        raise ValueError('nonfinite folded coefficients')
    return folded


def summarize(rows):
    return {
        'examples': len(rows),
        'global_esr': sum(r['error_energy'] for r in rows)/sum(r['wet_energy'] for r in rows),
        'peak_p95': float(np.quantile([r['peak_error'] for r in rows], .95)),
        'mean_peak_error': float(np.mean([r['peak_error'] for r in rows])),
        'by_fuzz_blend': {str(c): float(np.quantile([r['peak_error'] for r in rows if r['blend'] == c], .95))
                          for c in (0, 50, 100)},
    }


def decision(evaluations):
    summary = {name: summarize(rows) for name, rows in evaluations.items()}
    baseline, candidate = summary['unchanged'], summary['odd_even']
    delta = np.array([r['peak_error'] for r in evaluations['odd_even']]) - np.array(
        [r['peak_error'] for r in evaluations['unchanged']])
    # File pairs only; not an independent-performer confidence claim.
    rng = np.random.default_rng(1017)
    boot = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)
    interval = np.quantile(boot, [.025, .975]).tolist()
    checks = {
        'peak10pct_vs_each_control': all(candidate['peak_p95'] <= .9*summary[n]['peak_p95']
                                        for n in ('unchanged', 'odd', 'odd_cubic')),
        'global_esr_no_regression': candidate['global_esr'] <= baseline['global_esr'],
        'worst_blend_no_regression': max(candidate['by_fuzz_blend'].values()) <= max(baseline['by_fuzz_blend'].values()),
        'paired_peak_mean_bootstrap_upper_below_zero': interval[1] < 0,
    }
    return summary, {'continue_to_full_fit': all(checks.values()), 'checks': checks,
                     'paired_mean_peak_delta_bootstrap95': interval,
                     'bootstrap_unit': 'file, reused development material; no performer independence claimed'}


@torch.inference_mode()
def render(model, record):
    dry, wet, controls = read_effect_pair(Path(record['path']), 'dfz')
    if len(dry) != 144000 or not all(np.isfinite(a).all() for a in (dry, wet, controls)):
        raise ValueError('finite complete3s original recording required')
    hidden, original, state = [], [], None
    condition = torch.from_numpy(controls)[None]
    for start in range(0, len(dry), 4096):
        h, state = model.base.encode(torch.from_numpy(dry[start:start+4096])[None], condition, state)
        value = model.base.output_layer(h).squeeze(-1) + model.readout(h, condition)
        hidden.append(h[0]); original.append(value[0])
    return torch.cat(hidden), torch.cat(original), wet, controls


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh compact report path required')
    torch.set_num_threads(2)
    started = time.perf_counter()
    records = json.loads(MANIFEST.read_text())['audio_provenance']['fit']
    train, held = partition(records)
    hashes = {str(p): sha256(p) for p in (SOURCE, MANIFEST, Path(__file__),
              Path('remix/stable_effect.py'), Path('remix/stable_nonlinear_readout.py'),
              Path('remix/asrnn_effects.py'), Path('remix/asrnn_data.py'))}
    if hashes[str(SOURCE)] != SOURCE_HASH:
        raise ValueError('frozen best source changed')

    def check():
        if time.perf_counter()-started > BUDGET_SECONDS:
            raise TimeoutError('fixed15-minute budget exhausted; no automatic retry')

    def integrity():
        if any(sha256(path) != digest for path, digest in hashes.items()):
            raise ValueError('source/code/manifest changed')
        if any(sha256(r['path']) != r['sha256'] for r in records):
            raise ValueError('original fit audio changed')

    integrity()
    model, _ = load_stable_effect(SOURCE)
    equations = [(np.zeros((240, 240)), np.zeros(240)) for _ in range(9)]
    for i, record in enumerate(train):
        check()
        hidden, original, wet, control = render(model, record)
        corner = int(control[0]*2)*3+int(control[1]*2)
        ids, weights = sampling(wet, original.numpy())
        x = feature_bank(hidden[ids], model.readout.input[corner]).numpy().astype(np.float64)
        y = (torch.from_numpy(wet[ids])-original[ids]).double().numpy()
        mass = weights/max(float(np.mean(wet[1024:].astype(np.float64)**2)), 1e-8)
        equations[corner][0][:] += x.T @ (x*mass[:, None])
        equations[corner][1][:] += x.T @ (y*mass)
        if (i+1)%9 == 0:
            print(json.dumps({'stage': 'inner-fit-equations', 'completed': i+1, 'total': len(train)}), flush=True)
    weights = {mode: torch.from_numpy(np.stack([solve(xx, xy, mode) for xx, xy in equations])) for mode in MODES}
    evaluations = {name: [] for name in ('unchanged', *MODES)}
    for i, record in enumerate(held):
        check()
        hidden, original, wet, control = render(model, record)
        corner = int(control[0]*2)*3+int(control[1]*2)
        predictions = {name: [] for name in MODES}
        for start in range(0, len(wet), 4096):
            feature = feature_bank(hidden[start:start+4096], model.readout.input[corner])
            for name, weight in weights.items():
                predictions[name].append(original[start:start+4096] + feature @ weight[corner])
        predictions = {name: torch.cat(v).numpy() for name, v in predictions.items()}
        predictions['unchanged'] = original.numpy()
        target = wet[1024:].astype(np.float64)
        for name, audio in predictions.items():
            p = audio[1024:].astype(np.float64)
            evaluations[name].append({'file': Path(record['path']).name, 'blend': int(control[0]*100),
                                     'peak_error': float(abs(abs(p).max()-abs(target).max())),
                                     'error_energy': float(np.sum((p-target)**2)), 'wet_energy': float(np.sum(target**2))})
        if (i+1)%9 == 0:
            print(json.dumps({'stage': 'inner-held-full-cpu', 'completed': i+1, 'total': len(held)}), flush=True)
    integrity()
    summary, verdict = decision(evaluations)
    result = {'schema': 1, 'hypothesis': 'even frozen-state readout features, versus unchanged/odd/matched-cubic controls',
              'source_sha256': hashes, 'fit_only_manifest': records, 'inner_train_files': len(train),
              'inner_held_files': len(held), 'ridge': RIDGE, 'wall_budget_seconds': BUDGET_SECONDS,
              'elapsed_seconds': time.perf_counter()-started, 'summaries': summary, 'decision': verdict,
              'per_file': evaluations, 'coefficient_lists': {n: w.tolist() for n, w in weights.items()} if verdict['continue_to_full_fit'] else None,
              'original_calibration_audio_opened': False, 'official_eval_opened': False,
              'physical_audio_devices_used': False, 'source_audio_modified': False, 'admitted': False,
              'source_audio_and_code_reverified': True, 'audio_or_hidden_cache_saved': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'summaries': summary, 'decision': verdict}), flush=True)


if __name__ == '__main__':
    main()
