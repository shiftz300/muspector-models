"""Bounded convex full-record readout diagnostic, with no runtime limiter.

Keep the old80 odd state features. Fit a quadratic waveform objective subject
to training-only linear peak constraints; verify cuts on every training sample.
Only the preregistered original-fit180/held54 files are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import LinearConstraint, linprog, minimize
import torch

from .diagnose_dfz_readout_parity import MANIFEST, SOURCE, SOURCE_HASH, partition, render, summarize
from .precheck_multirate_fuzz import sha256
from .stable_effect import load_stable_effect

EPSILON, RIDGE, MAX_ROUNDS, BUDGET = .015, 1e-3, 8, 900


def quadratic(xx, xy):
    scale = np.sqrt(np.maximum(np.diag(xx), 1e-16))
    h = xx/scale[:, None]/scale[None, :] + RIDGE*np.eye(len(xy))
    b = xy/scale
    return h, b, scale


def constraints(rows, scale, cuts, epsilon=EPSILON):
    matrix, lower, upper = [], [], []
    for index, row in enumerate(rows):
        x, p, y = row['x'], row['p'], row['y']
        peak_index = int(np.abs(y).argmax())
        peak, sign = abs(y[peak_index]), float(np.sign(y[peak_index]))
        matrix.append(sign*x[peak_index]/scale)
        lower.append(peak-epsilon-sign*p[peak_index]); upper.append(np.inf)
        for frame in sorted(cuts[index]):
            matrix.append(x[frame]/scale)
            lower.append(-peak-epsilon-p[frame]); upper.append(peak+epsilon-p[frame])
    return np.asarray(matrix), np.asarray(lower), np.asarray(upper)


def solve_constrained(rows, xx, xy, deadline, epsilon=EPSILON):
    if not 0 < epsilon <= .02:
        raise ValueError('training tolerance must stay within original0.02 gate')
    h, b, scale = quadratic(xx, xy)
    plain = np.linalg.solve(h, b)
    cuts = [set((int(np.argmax(r['p'])), int(np.argmin(r['p'])), int(np.abs(r['y']).argmax()))) for r in rows]
    history, value = [], np.zeros(len(b))
    for iteration in range(MAX_ROUNDS):
        if time.perf_counter() >= deadline:
            return None, plain/scale, {'status': 'budget-exhausted', 'history': history}
        a, lo, hi = constraints(rows, scale, cuts, epsilon)
        upper_finite = np.isfinite(hi)
        feasibility = linprog(np.zeros(len(b)), A_ub=np.r_[a[upper_finite], -a],
                              b_ub=np.r_[hi[upper_finite], -lo], bounds=[(None, None)]*len(b), method='highs',
                              options={'time_limit': min(30., max(.01, deadline-time.perf_counter()))})
        if not feasibility.success:
            return None, plain/scale, {'status': 'cut-constraints-infeasible' if feasibility.status == 2 else 'feasibility-solver-failed',
                                      'solver_status': int(feasibility.status), 'message': feasibility.message, 'history': history,
                                      'constraints': len(lo), 'round': iteration}
        def objective(v):
            if time.perf_counter() >= deadline:
                raise TimeoutError('fixed total diagnostic budget exhausted')
            return .5*float(v@h@v)-float(b@v), h@v-b
        optimized = minimize(objective, value if iteration else feasibility.x, jac=True, method='SLSQP',
                             constraints=[LinearConstraint(a, lo, hi)],
                             options={'maxiter': 300, 'ftol': 1e-11})
        if not optimized.success:
            return None, plain/scale, {'status': 'quadratic-solver-failed', 'message': optimized.message, 'history': history}
        value = optimized.x
        # This exact float32 coefficient is checked on complete train waveforms.
        weight = (value/scale).astype(np.float32)
        overshoots, deficits, added = [], [], 0
        for index, row in enumerate(rows):
            prediction = row['p']+row['x']@weight
            y = row['y']; k = int(np.abs(y).argmax()); peak = abs(y[k])
            violation = abs(prediction)-(peak+epsilon)
            overshoots.append(float(violation.max()))
            deficits.append(float(peak-epsilon-np.sign(y[k])*prediction[k]))
            ids = np.flatnonzero(violation > 5e-7)
            if len(ids):
                worst = ids[np.argsort(violation[ids])[-16:]]
                before = len(cuts[index]); cuts[index].update(map(int, worst)); added += len(cuts[index])-before
        history.append({'round': iteration, 'constraints': len(lo), 'iterations': int(optimized.nit),
                        'max_upper_violation': max(overshoots), 'max_lower_violation': max(deficits), 'cuts_added': added})
        if max(overshoots+deficits) <= 5e-7:
            return weight, (plain/scale).astype(np.float32), {'status': 'complete-training-waveforms-verified', 'history': history}
        if not added:
            return None, plain/scale, {'status': 'float32-feasibility-failed', 'history': history}
    return None, plain/scale, {'status': 'cut-round-budget-exhausted', 'history': history}


def file_metrics(record, prediction, wet):
    y, p = wet.astype(np.float64), prediction.astype(np.float64)
    return {'file': Path(record['path']).name, 'blend': int(Path(record['path']).stem.split(',')[0]),
            'peak_error': float(abs(abs(p).max()-abs(y).max())),
            'error_energy': float(np.sum((p-y)**2)), 'wet_energy': float(np.sum(y*y))}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh result path required')
    started = time.perf_counter(); deadline = started+BUDGET
    torch.set_num_threads(2)
    records = json.loads(MANIFEST.read_text())['audio_provenance']['fit']
    train, held = partition(records)
    code = [Path(__file__), Path('remix/diagnose_dfz_readout_parity.py'), Path('remix/stable_effect.py'),
            Path('remix/stable_nonlinear_readout.py'), Path('remix/asrnn_effects.py'), Path('remix/asrnn_data.py')]
    hashes = {str(p): sha256(p) for p in (SOURCE, MANIFEST, *code)}
    if hashes[str(SOURCE)] != SOURCE_HASH:
        raise ValueError('immutable source differs')
    def integrity():
        if any(sha256(p) != h for p, h in hashes.items()) or any(sha256(r['path']) != r['sha256'] for r in records):
            raise ValueError('source, code or original audio changed')
    integrity()
    model, _ = load_stable_effect(SOURCE)
    solutions, fits, evaluations = {}, {}, {name: [] for name in ('unchanged', 'dense_ridge', 'constrained')}
    failure = None
    try:
        for a in (0, 50, 100):
            for b in (0, 50, 100):
                label = f'{a},{b}'
                current = [r for r in train if Path(r['path']).stem.rsplit(',', 1)[0] == label]
                rows, xx, xy = [], np.zeros((80, 80)), np.zeros(80)
                for record in current:
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('fixed total diagnostic budget exhausted')
                    h, p, wet, control = render(model, record)
                    corner = int(control[0]*2)*3+int(control[1]*2)
                    x = torch.cat((h[1024:]*8, torch.tanh(h[1024:]@model.readout.input[corner]*8)), -1).numpy()
                    p, y = p[1024:].numpy(), wet[1024:]
                    factor = 1/(len(y)*max(float(np.mean(y.astype(np.float64)**2)), 1e-8))
                    for start in range(0, len(y), 8192):
                        block = x[start:start+8192].astype(np.float64)
                        target = y[start:start+8192].astype(np.float64)-p[start:start+8192]
                        xx += (block.T@block)*factor; xy += (block.T@target)*factor
                    rows.append({'x': x, 'p': p, 'y': y})
                    del h
                print(json.dumps({'stage': 'corner-full-equations', 'corner': label, 'files': len(rows)}), flush=True)
                weight, plain, audit = solve_constrained(rows, xx, xy, deadline)
                fits[label] = audit
                print(json.dumps({'stage': 'corner-constraint-audit', 'corner': label, **audit}), flush=True)
                if weight is None:
                    failure = {'corner': label, 'status': audit['status']}
                    break
                solutions[label] = {'constrained': weight, 'dense_ridge': plain}
                del rows
            if failure:
                break
        if failure is None:
            for i, record in enumerate(held):
                if time.perf_counter() >= deadline:
                    raise TimeoutError('fixed total diagnostic budget exhausted')
                h, p, wet, control = render(model, record)
                corner = int(control[0]*2)*3+int(control[1]*2)
                x = torch.cat((h[1024:]*8, torch.tanh(h[1024:]@model.readout.input[corner]*8)), -1).numpy()
                p, y = p[1024:].numpy(), wet[1024:]
                label = Path(record['path']).stem.rsplit(',', 1)[0]
                evaluations['unchanged'].append(file_metrics(record, p, y))
                for name, w in solutions[label].items():
                    evaluations[name].append(file_metrics(record, p+x@w, y))
                if (i+1)%9 == 0:
                    print(json.dumps({'stage': 'inner-held-full-cpu', 'completed': i+1, 'total': 54}), flush=True)
    except TimeoutError as error:
        failure = {'status': 'budget-exhausted', 'message': str(error)}
    integrity()
    summary = {n: summarize(rows) for n, rows in evaluations.items()} if failure is None else {}
    checks, interval = {}, None
    if failure is None:
        base, candidate = summary['unchanged'], summary['constrained']
        delta = np.array([r['peak_error'] for r in evaluations['constrained']])-np.array([r['peak_error'] for r in evaluations['unchanged']])
        rng = np.random.default_rng(1017)
        interval = np.quantile(delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1), [.025, .975]).tolist()
        checks = {'peak10pct_vs_each_control': all(candidate['peak_p95'] <= .9*summary[n]['peak_p95'] for n in ('unchanged', 'dense_ridge')),
                  'global_esr_no_regression': candidate['global_esr'] <= base['global_esr'],
                  'worst_blend_no_regression': max(candidate['by_fuzz_blend'].values()) <= max(base['by_fuzz_blend'].values()),
                  'paired_mean_peak_bootstrap_upper_below_zero': interval[1] < 0}
    go = failure is None and all(checks.values())
    report = {'schema': 1, 'source_sha256': hashes, 'fit_only_manifest': records, 'epsilon': EPSILON, 'ridge': RIDGE,
              'budget_seconds': BUDGET, 'max_cut_rounds': MAX_ROUNDS, 'elapsed_seconds': time.perf_counter()-started,
              'fit_audits': fits, 'failure': failure, 'summaries': summary, 'per_file': evaluations,
              'decision': {'continue_to_full_fit': go, 'checks': checks, 'paired_mean_peak_delta_bootstrap95': interval},
              'coefficients': {c: {n: w.tolist() for n, w in v.items()} for c, v in solutions.items()} if go else None,
              'original_calibration_audio_opened': False, 'official_eval_opened': False, 'admitted': False,
              'physical_audio_devices_used': False, 'source_audio_modified': False,
              'source_audio_and_code_reverified': True, 'audio_or_hidden_cache_saved': False,
              'runtime_limiter_or_gain_added': False,
              'interpretation': 'Training feasibility only for fixed80 features and same-index peak anchors; not proof of physical impossibility.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'failure': failure, 'summaries': summary, 'decision': report['decision']}), flush=True)


if __name__ == '__main__':
    main()
