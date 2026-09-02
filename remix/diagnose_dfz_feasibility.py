"""Numerical reproducer for the failed100,0 LP; not another model experiment.

Same rows,80 features,quadratic objective and0.015 constraints. Replace only
the zero-objective feasibility LP with a row-scaled, always-feasible Phase I.
Positive slack is reported separately from numerical solver failure, with
primal/dual residuals. This one-corner check cannot admit a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.optimize import OptimizeResult, linprog
import torch

from . import diagnose_dfz_peak_constraints as original
from .diagnose_dfz_readout_parity import MANIFEST, SOURCE, SOURCE_HASH, partition, render
from .precheck_multirate_fuzz import sha256
from .stable_effect import load_stable_effect


def epsilon_lower_bound(a, b, epsilon, time_limit=30.):
    """Auxiliary bound only: every supplied RHS is b0+epsilon.

    Minimize the shared absolute training tolerance on this subset of cuts.
    No solution from this auxiliary problem is installed into a candidate.
    """
    scales = np.maximum(np.linalg.norm(a, axis=1), 1e-12)
    matrix = np.c_[a/scales[:, None], -1/scales]
    rhs = (b-epsilon)/scales
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*')
        result = linprog(np.r_[np.zeros(a.shape[1]), 1.], A_ub=matrix, b_ub=rhs,
                         bounds=[(None, None)]*a.shape[1]+[(0., None)], method='highs-ipm',
                         options={'time_limit': time_limit, 'primal_feasibility_tolerance': 1e-8,
                                  'dual_feasibility_tolerance': 1e-8, 'ipm_optimality_tolerance': 1e-10,
                                  'threads': 2, 'parallel': False})
    if not result.success:
        return {'verified': False, 'solver_status': int(result.status), 'message': result.message}
    dual = -np.asarray(result.ineqlin.marginals)
    lower = float(-rhs@dual)
    residual = float(abs(matrix[:, :-1].T@dual).max())
    budget = float((dual/scales).sum())
    primal_error = float(np.max(matrix@result.x-rhs))
    gap = float(result.fun-lower)
    verified = (dual.min() >= -1e-8 and residual < 1e-7 and budget <= 1+1e-7
                and primal_error < 1e-7 and abs(gap) < 1e-7)
    return {'verified': verified, 'minimum_epsilon_on_cuts': float(result.fun), 'dual_lower_bound': lower,
            'dual_balance_max': residual, 'dual_epsilon_weight': budget, 'primal_max_violation': primal_error,
            'primal_dual_gap': gap, 'applies_to': 'same-index anchors plus global caps in fixed80 feature space only',
            'not_the_original_independent_maximum_metric': True}


def phase_one(a, b, time_limit=30., epsilon=original.EPSILON, auxiliary_bound=True):
    """Invertible positive row scaling; no feature truncation or relaxed gate."""
    row_scale = np.maximum(np.linalg.norm(a, axis=1), 1e-12)
    matrix, rhs = a/row_scale[:, None], b/row_scale
    c = np.r_[np.zeros(a.shape[1]), 1.]
    augmented = np.c_[matrix, -np.ones(len(b))]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*')
        result = linprog(c, A_ub=augmented, b_ub=rhs,
                         bounds=[(None, None)]*a.shape[1]+[(0., None)], method='highs-ipm',
                         options={'time_limit': time_limit, 'primal_feasibility_tolerance': 1e-8,
                                  'dual_feasibility_tolerance': 1e-8, 'ipm_optimality_tolerance': 1e-10,
                                  'threads': 2, 'parallel': False})
    audit = {'rows': len(b), 'columns': a.shape[1], 'solver_status': int(result.status),
             'message': result.message, 'threads': 2}
    if not result.success:
        return OptimizeResult(success=False, status=4, message=result.message), audit
    slack = float(result.x[-1]); multipliers = -np.asarray(result.ineqlin.marginals)
    balance = float(abs(matrix.T@multipliers).max())
    dual_lower = float(-rhs@multipliers)
    audit.update(slack=slack, dual_lower_bound=dual_lower, dual_balance_max=balance,
                 dual_weight_sum=float(multipliers.sum()), min_dual_weight=float(multipliers.min()),
                 primal_augmented_max_violation=float(np.max(augmented@result.x-rhs)),
                 original_max_violation=float(np.max(a@result.x[:-1]-b)),
                 primal_dual_gap=slack-dual_lower)
    if (not np.isfinite(result.x).all() or not np.isfinite(multipliers).all()
            or audit['primal_augmented_max_violation'] > 1e-7 or balance > 1e-7
            or multipliers.min() < -1e-8 or multipliers.sum() > 1+1e-7):
        return OptimizeResult(success=False, status=4, message='PhaseI residual checks failed'), audit
    if slack > 1e-7:
        if dual_lower <= 1e-7 or abs(slack-dual_lower) > max(1e-7, slack*1e-4):
            return OptimizeResult(success=False, status=4, message='Positive slack without verified dual support'), audit
        if auxiliary_bound:
            audit['auxiliary_absolute_epsilon_bound'] = epsilon_lower_bound(a, b, epsilon, time_limit)
        return OptimizeResult(success=False, status=2, message='Positive PhaseI minimum slack; primal/dual checks passed'), audit
    if audit['original_max_violation'] > 5e-7:
        return OptimizeResult(success=False, status=4, message='Original-scale feasibility check failed'), audit
    return OptimizeResult(success=True, status=0, x=result.x[:-1], message='PhaseI zero slack verified'), audit


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--original-gate', action='store_true',
                        help='Same model, original0.02 target minus1e-6 float32 safety margin; not0.015 design target')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh numerical-reproducer result required')
    started = time.perf_counter(); torch.set_num_threads(2)
    epsilon = .02-1e-6 if args.original_gate else original.EPSILON
    names = [Path(__file__), Path(original.__file__), Path('remix/diagnose_dfz_readout_parity.py'),
             Path('remix/stable_effect.py'), Path('remix/stable_nonlinear_readout.py'),
             Path('remix/asrnn_effects.py'), Path('remix/asrnn_data.py'), SOURCE, MANIFEST]
    hashes = {str(p): sha256(p) for p in names}
    assert hashes[str(SOURCE)] == SOURCE_HASH
    all_fit, _ = partition(json.loads(MANIFEST.read_text())['audio_provenance']['fit'])
    records = [r for r in all_fit if Path(r['path']).stem.rsplit(',', 1)[0] == '100,0']
    assert len(records) == 20
    def integrity():
        assert all(sha256(p) == h for p, h in hashes.items())
        assert all(sha256(r['path']) == r['sha256'] for r in records)
    integrity()
    model, _ = load_stable_effect(SOURCE)
    rows, xx, xy = [], np.zeros((80, 80)), np.zeros(80)
    for i, record in enumerate(records):
        h, p, wet, _ = render(model, record)
        x = torch.cat((h[1024:]*8, torch.tanh(h[1024:]@model.readout.input[6]*8)), -1).numpy()
        p, y = p[1024:].numpy(), wet[1024:]
        factor = 1/(len(y)*max(float(np.mean(y.astype(np.float64)**2)), 1e-8))
        for start in range(0, len(y), 8192):
            block = x[start:start+8192].astype(np.float64)
            target = y[start:start+8192].astype(np.float64)-p[start:start+8192]
            xx += (block.T@block)*factor; xy += (block.T@target)*factor
        rows.append({'x': x, 'p': p, 'y': y})
        if (i+1)%5 == 0:
            print(json.dumps({'stage': 'same-corner-reproducer', 'files': i+1, 'total': 20}), flush=True)
        if time.perf_counter()-started > 240:
            raise TimeoutError('four-minute reproducer budget exhausted')
    calls = []
    def replacement(c, *, A_ub, b_ub, bounds, method, options):
        result, audit = phase_one(A_ub, b_ub, time_limit=min(30., options['time_limit']), epsilon=epsilon)
        calls.append(audit)
        print(json.dumps({'stage': 'phase-I-feasibility', **audit}), flush=True)
        return result
    old = original.linprog
    try:
        original.linprog = replacement
        weight, plain, audit = original.solve_constrained(rows, xx, xy, started+240, epsilon=epsilon)
    finally:
        original.linprog = old
    integrity()
    result = {'schema': 1, 'purpose': 'numerical-reproducer-only', 'corner': '100,0', 'files': records,
              'source_sha256': hashes, 'epsilon': epsilon, 'ridge': original.RIDGE,
              'original_gate_mode': args.original_gate,
              'solver_change_only': True, 'row_scaling_is_invertible': True, 'constraints_relaxed': False,
              'phase_I_calls': calls, 'constraint_audit': audit, 'elapsed_seconds': time.perf_counter()-started,
              'model_saved': False, 'coefficients_saved': weight is not None,
              'coefficients': {'constrained': weight.tolist(), 'dense_ridge': plain.tolist()} if weight is not None else None,
              'admitted': False, 'source_and_audio_reverified': True,
              'original_calibration_audio_opened': False, 'official_eval_opened': False,
              'physical_audio_devices_used': False, 'source_audio_modified': False,
              'limitation': 'Numerical feasibility for this fixed feature class and same-index anchors; not physical impossibility or final audio acceptance.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'constraint_audit': audit, 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
