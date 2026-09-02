"""Fixed-basis whole-record CVaR-constrained readout, no inference limiter.

Ephemeral frozen features only. All original-fit records participate; explicit
fit180/inner-held54 split. No original calibration/eval audio is opened.
"""
from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import solve_triangular
import torch

from .asrnn_effects import read_effect_pair
from .diagnose_dfz_readout_parity import MANIFEST, SOURCE, SOURCE_HASH, partition, render, summarize
from .diagnose_dfz_peak_constraints import file_metrics, quadratic
from .precheck_multirate_fuzz import sha256
from .stable_effect import load_stable_effect

PEAK_LIMIT, GROUP_LIMIT = .02-1e-5, .03-1e-5
WALL_SECONDS, ROUNDS = 2700, 10


def tail_mean(values):
    value = np.asarray(values, dtype=float)
    return float(np.sort(value)[-max(1, math.ceil(.05*len(value))):].mean())


def tail_problem(equations, rows, width=80, peak_limit=PEAK_LIMIT, group_limit=GROUP_LIMIT):
    """Convex epigraph of the average of the largest ceil(.05*N) bounds."""
    count, corners = len(rows), len(equations)
    groups = sorted({r['blend'] for r in rows})
    e0 = corners*width; t0 = e0+count; s0 = t0+1
    tg0 = s0+count; sg0 = tg0+len(groups); variables = sg0+count
    h, b, scales = zip(*(quadratic(xx, xy) for xx, xy in equations))
    p = sparse.block_diag([*h, sparse.csc_matrix((variables-e0, variables-e0))], format='csc')
    q = np.r_[np.concatenate([-v for v in b]), np.zeros(variables-e0)]
    rr, cc, vv, lower, upper = [], [], [], [], []
    def add(columns, values, lo=-np.inf, hi=np.inf):
        rr.extend([len(lower)]*len(columns)); cc.extend(columns); vv.extend(values)
        lower.append(lo); upper.append(hi)
    for index in range(e0, variables):
        add([index], [1.], lo=0.)
    k = max(1, math.ceil(.05*count))
    add([t0, *range(s0, s0+count)], [1., *([1/k]*count)], hi=peak_limit)
    for group_index, group in enumerate(groups):
        members = [i for i, row in enumerate(rows) if row['blend'] == group]
        kg = max(1, math.ceil(.05*len(members)))
        add([tg0+group_index, *[sg0+i for i in members]], [1., *([1/kg]*len(members))], hi=group_limit)
    for i, row in enumerate(rows):
        group_index = groups.index(row['blend'])
        add([e0+i, t0, s0+i], [1., -1., -1.], hi=0.)
        add([e0+i, tg0+group_index, sg0+i], [1., -1., -1.], hi=0.)
        columns = list(range(row['corner']*width, (row['corner']+1)*width))+[e0+i]
        scale = scales[row['corner']]
        add(columns, np.r_[-row['sign']*row['anchor_x']/scale, -1.],
            hi=row['sign']*row['anchor_p']-row['peak'])
        for x, original in row['cuts'].values():
            add(columns, np.r_[x/scale, -1.], hi=row['peak']-original)
            add(columns, np.r_[-x/scale, -1.], hi=row['peak']+original)
    a = sparse.csc_matrix((vv, (rr, cc)), shape=(len(lower), variables))
    plain = np.stack([np.linalg.solve(matrix, target)/scale for matrix, target, scale in zip(h, b, scales)]).astype(np.float32)
    return p, q, a, np.asarray(lower), np.asarray(upper), np.stack(scales), plain, e0


def whiten_problem(p, q, a, corners, width):
    """Invertible Cholesky coordinates; no discarded directions or new loss."""
    transforms = []
    for corner in range(corners):
        start = corner*width
        block = p[start:start+width, start:start+width].toarray()
        factor = np.linalg.cholesky(block)
        transforms.append(solve_triangular(factor.T, np.eye(width), lower=False))
    rest = p.shape[0]-corners*width
    transform = sparse.block_diag([*transforms, sparse.eye(rest)], format='csc')
    return transform, (transform.T@p@transform).tocsc(), np.asarray(transform.T@q), (a@transform).tocsc()


def save_replay(path, p, q, a, lo, hi):
    """Compact failing QP only; no full audio or hidden-state cache."""
    path = Path(path)
    if path.exists():
        raise ValueError('never overwrite a previous solver replay')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, p_data=p.data, p_indices=p.indices, p_indptr=p.indptr, p_shape=p.shape,
                        a_data=a.data, a_indices=a.indices, a_indptr=a.indptr, a_shape=a.shape,
                        q=q, lo=lo, hi=hi)
    return {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size,
            'contents': 'sparse original-coordinate QP, no raw audio or full hidden states'}


def solve_problem(p, q, a, lo, hi, corners, width=80, seconds=60., original_warm_start=None):
    transform, pw, qw, aw = whiten_problem(p, q, a, corners, width)
    solver = osqp.OSQP(algebra='builtin')
    solver.setup(P=pw, q=qw, A=aw, l=lo, u=hi, verbose=False, eps_abs=1e-8, eps_rel=1e-8,
                 max_iter=2000000, polishing=True, adaptive_rho_interval=50, time_limit=seconds)
    if original_warm_start is not None:
        solver.warm_start(x=sparse.linalg.spsolve(transform, np.asarray(original_warm_start)))
    result = solver.solve(raise_error=False)
    audit = {'status': result.info.status, 'status_val': result.info.status_val,
             'iterations': result.info.iter, 'primal_residual': result.info.prim_res,
             'dual_residual': result.info.dual_res, 'seconds': result.info.run_time,
             'variables': len(q), 'constraints': len(lo), 'iteration_limit': 2000000,
             'time_limit_seconds': seconds, 'phase_one_primal_warm_start': original_warm_start is not None}
    audit['coordinates'] = 'invertible Cholesky; original objective and constraints unchanged'
    if result.x is None or result.y is None or not np.isfinite(result.x).all() or not np.isfinite(result.y).all():
        return None, audit
    original = np.asarray(transform@result.x)
    ax = a@original
    primal = float(max(np.max(lo-ax), np.max(ax-hi)))
    dual = float(abs(p@original+q+a.T@result.y).max())
    audit.update(independent_primal_residual=primal, independent_dual_residual=dual)
    if result.info.status_val != 1 or primal > 2e-7 or dual > 1e-6:
        if result.info.status_val == 1:
            audit['status'] = 'independent-residual-check-failed'
        return None, audit
    return original, audit


def solve_tail(equations, rows, seconds=60., width=80, peak_limit=PEAK_LIMIT, group_limit=GROUP_LIMIT, replay_path=None):
    p, q, a, lo, hi, scales, plain, e0 = tail_problem(equations, rows, width, peak_limit, group_limit)
    original, audit = solve_problem(p, q, a, lo, hi, len(equations), width, seconds)
    if original is None:
        if replay_path is not None:
            audit['replay'] = save_replay(replay_path, p, q, a, lo, hi)
        return None, None, plain, audit
    weight = (original[:e0].reshape(scales.shape)/scales).astype(np.float32)
    return weight, original[e0:e0+len(rows)], plain, audit


class FrozenFeatures:
    def __init__(self, model, device):
        self.model = model
        self.encoder = copy.deepcopy(model.base).to(device).eval().requires_grad_(False)
        self.device = device

    @torch.inference_mode()
    def __call__(self, record):
        return self.batch([record])[0]

    @torch.inference_mode()
    def batch(self, records):
        pairs = [read_effect_pair(Path(r['path']), 'dfz') for r in records]
        if any(len(dry) != 144000 or not all(np.isfinite(v).all() for v in (dry, wet, control))
               for dry, wet, control in pairs):
            raise ValueError('complete finite3s original audio required')
        dry = torch.from_numpy(np.stack([v[0] for v in pairs]))
        c = torch.from_numpy(np.stack([v[2] for v in pairs])); cd = c.to(self.device)
        corners = (c[:, 0]*2).long()*3+(c[:, 1]*2).long()
        projections = self.model.readout.input[corners]
        state = None
        features = torch.empty(len(records), 142976, 80)
        predictions = torch.empty(len(records), 142976)
        for start in range(0, 144000, 4096):
            h, state = self.encoder.encode(dry[:, start:start+4096].to(self.device), cd, state)
            h = h.cpu()
            p = self.model.base.output_layer(h).squeeze(-1)+self.model.readout(h, c)
            x = torch.cat((h*8, torch.tanh(torch.bmm(h, projections)*8)), -1)
            first = max(0, 1024-start); target_start = start+first-1024
            features[:, target_start:target_start+x.shape[1]-first] = x[:, first:]
            predictions[:, target_start:target_start+x.shape[1]-first] = p[:, first:]
        if self.device.type == 'mps' and torch.mps.driver_allocated_memory() > 4000000000:
            raise MemoryError('fixed4GB MPS allocation budget exceeded')
        return [(features[i].numpy(), predictions[i].numpy(), pairs[i][1][1024:], int(corners[i]))
                for i in range(len(records))]

    def records(self, records):
        for start in range(0, len(records), 9):
            yield from self.batch(records[start:start+9])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--replay-qp', type=Path, help='CPU numerical replay only; opens no audio or checkpoint')
    parser.add_argument('--feasibility-only', action='store_true', help='90-second Phase I of saved cuts, no new optimization candidate')
    parser.add_argument('--phase-one-warm-start', action='store_true', help='use verified Phase-I witness to initialize the same60-second QP')
    args = parser.parse_args()
    if (args.feasibility_only or args.phase_one_warm_start) and args.replay_qp is None:
        parser.error('numerical replay modes require --replay-qp')
    if args.feasibility_only and args.phase_one_warm_start:
        parser.error('choose one numerical replay mode')
    if args.output.exists():
        raise ValueError('fresh report path required')
    if args.replay_qp:
        before = sha256(args.replay_qp)
        with np.load(args.replay_qp, allow_pickle=False) as data:
            def matrix(prefix):
                return sparse.csc_matrix((data[prefix+'_data'], data[prefix+'_indices'], data[prefix+'_indptr']),
                                         shape=tuple(data[prefix+'_shape']))
            if args.feasibility_only or args.phase_one_warm_start:
                from .diagnose_dfz_feasibility import phase_one
                a, lo, hi = matrix('a'), data['lo'], data['hi']
                upper, lower = np.isfinite(hi), np.isfinite(lo)
                tick = time.perf_counter()
                feasibility, audit = phase_one(sparse.vstack([a[upper], -a[lower]]).toarray(),
                                                np.r_[hi[upper], -lo[lower]], time_limit=90., auxiliary_bound=False)
                phase_one_audit = dict(audit)
                phase_one_audit.update(seconds=time.perf_counter()-tick, time_limit_seconds=90.,
                             verified_feasible=bool(feasibility.success), verified_infeasible=feasibility.status == 2,
                             applies_to='saved cuts only; complete-waveform feasibility remains unknown')
                if args.phase_one_warm_start and feasibility.success:
                    solution, audit = solve_problem(matrix('p'), data['q'], a, lo, hi, 9,
                                                      original_warm_start=feasibility.x)
                    audit['phase_one'] = phase_one_audit
                else:
                    solution, audit = None, phase_one_audit  # A feasibility witness is never a fitted model.
            else:
                solution, audit = solve_problem(matrix('p'), data['q'], matrix('a'), data['lo'], data['hi'], 9)
        if sha256(args.replay_qp) != before:
            raise ValueError('QP replay changed')
        result = {'replay_sha256': before, 'source_sha256': sha256(__file__), 'osqp_version': osqp.__version__,
                  'audit': audit, 'numerically_verified': solution is not None, 'audio_opened': False,
                  'checkpoint_opened': False, 'admitted': False}
        result['feasibility_only'] = args.feasibility_only
        result['phase_one_warm_start'] = args.phase_one_warm_start
        if args.feasibility_only or args.phase_one_warm_start:
            result['phase_one_source_sha256'] = sha256('remix/diagnose_dfz_feasibility.py')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(result), flush=True)
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError('offline MPS feature compute required; no audio hardware')
    started = time.perf_counter(); deadline = started+WALL_SECONDS
    torch.set_num_threads(2)
    records = json.loads(MANIFEST.read_text())['audio_provenance']['fit']
    train, held = partition(records)
    source_files = [SOURCE, MANIFEST, Path(__file__), Path('remix/diagnose_dfz_readout_parity.py'),
                    Path('remix/diagnose_dfz_peak_constraints.py'), Path('remix/stable_effect.py'),
                    Path('remix/stable_nonlinear_readout.py'), Path('remix/asrnn_effects.py'), Path('remix/asrnn_data.py')]
    hashes = {str(path): sha256(path) for path in source_files}
    if hashes[str(SOURCE)] != SOURCE_HASH:
        raise ValueError('frozen source changed')
    def integrity():
        if any(sha256(p) != h for p, h in hashes.items()) or any(sha256(r['path']) != r['sha256'] for r in records):
            raise ValueError('source/code/audio changed')
    def budget():
        if time.perf_counter() >= deadline:
            raise TimeoutError('fixed45-minute phase budget exhausted')
    integrity()
    model, _ = load_stable_effect(SOURCE)
    extractor = FrozenFeatures(model, torch.device('mps'))
    if args.smoke_only:
        tick = time.perf_counter(); batch = extractor.batch(train[:9]); cold = time.perf_counter()-tick
        del batch
        tick = time.perf_counter(); batch = extractor.batch(train[:9]); warm = time.perf_counter()-tick
        x, p, y, corner = batch[0]
        cpu = FrozenFeatures(model, torch.device('cpu'))
        xc, pc, _, _ = cpu(train[0])
        result = {'smoke_only': True, 'rows': len(x), 'features': x.shape[1], 'finite': bool(np.isfinite(x).all()),
                  'frozen_mps_source_vs_cpu_max_error': float(abs(p-pc).max()),
                  'mps_feature_vs_cpu_max_error': float(abs(x-xc).max()), 'compute_seconds': time.perf_counter()-started,
                  'batch_size': 9, 'cold_batch_seconds': cold, 'warm_batch_seconds': warm,
                  'mps_driver_bytes': torch.mps.driver_allocated_memory(), 'source_sha256': hashes,
                  'physical_audio_devices_used': False, 'admitted': False}
        integrity(); args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result), flush=True)
        return
    equations = [(np.zeros((80, 80)), np.zeros(80)) for _ in range(9)]
    observations, history, failure = [], [], None
    weight = plain = None
    evaluations = {n: [] for n in ('unchanged', 'dense_ridge', 'constrained')}
    try:
        for i, (record, extracted) in enumerate(zip(train, extractor.records(train))):
            budget(); x, p, y, corner = extracted
            factor = 1/(len(y)*max(float(np.mean(y.astype(np.float64)**2)), 1e-8))
            for start in range(0, len(y), 8192):
                block = x[start:start+8192].astype(np.float64)
                residual = y[start:start+8192].astype(np.float64)-p[start:start+8192]
                equations[corner][0][:] += block.T@block*factor
                equations[corner][1][:] += block.T@residual*factor
            k = int(abs(y).argmax()); frames = {k, int(p.argmax()), int(p.argmin())}
            observations.append({'corner': corner, 'blend': corner//3*50, 'peak': float(abs(y[k])),
                                 'sign': float(np.sign(y[k])), 'anchor_x': x[k].copy(), 'anchor_p': float(p[k]),
                                 'cuts': {int(j): (x[j].copy(), float(p[j])) for j in frames}})
            if (i+1)%18 == 0:
                print(json.dumps({'stage': 'complete-waveform-equations', 'files': i+1, 'total': 180}), flush=True)
        for iteration in range(ROUNDS):
            budget()
            weight, bounds, plain, audit = solve_tail(equations, observations, seconds=min(60., deadline-time.perf_counter()),
                                                     replay_path=args.output.with_suffix('.qp.npz'))
            print(json.dumps({'stage': 'tail-qp', 'round': iteration, **audit}), flush=True)
            if weight is None:
                failure = {'status': 'qp-not-verified', 'audit': audit}; break
            actual, added, max_violation = [], 0, -np.inf
            for i, (record, row, extracted) in enumerate(zip(train, observations, extractor.records(train))):
                budget(); x, p, y, corner = extracted
                prediction = p+x@weight[corner]
                deficit = row['peak']-row['sign']*float(row['anchor_p']+row['anchor_x']@weight[corner])
                overshoot = abs(prediction)-row['peak']
                bound = max(0., float(overshoot.max()), deficit)
                actual.append(bound); max_violation = max(max_violation, bound-bounds[i])
                ids = np.flatnonzero(overshoot > bounds[i]+5e-7)
                if len(ids):
                    ids = ids[np.argsort(overshoot[ids])[-16:]]
                    for j in ids:
                        if int(j) not in row['cuts']:
                            row['cuts'][int(j)] = (x[j].copy(), float(p[j])); added += 1
                if (i+1)%36 == 0:
                    print(json.dumps({'stage': 'complete-fit-scan', 'round': iteration, 'files': i+1, 'total': 180}), flush=True)
            group_tail = {str(g): tail_mean([v for v, r in zip(actual, observations) if r['blend'] == g]) for g in (0, 50, 100)}
            audit.update(round=iteration, actual_tail=tail_mean(actual), group_tail=group_tail,
                         full_scan_max_bound_violation=float(max_violation), cuts_added=added)
            history.append(audit)
            print(json.dumps({'stage': 'verified-fit-tail', **audit}), flush=True)
            if max_violation <= 5e-7 and tail_mean(actual) <= .02 and max(group_tail.values()) <= .03:
                break
            if not added:
                failure = {'status': 'float32-full-scan-failed', 'audit': audit}; break
        else:
            failure = {'status': 'fixed10-cut-rounds-exhausted'}
        if failure is None:
            del extractor
            torch.mps.empty_cache()
            for i, record in enumerate(held):
                budget()
                h, p, wet, c = render(model, record)
                corner = int(c[0]*2)*3+int(c[1]*2)
                x = torch.cat((h[1024:]*8, torch.tanh(h[1024:]@model.readout.input[corner]*8)), -1)
                p, y = p[1024:], wet[1024:]
                evaluations['unchanged'].append(file_metrics(record, p.numpy(), y))
                for name, w in (('dense_ridge', plain), ('constrained', weight)):
                    prediction = (p+x@torch.from_numpy(w[corner])).numpy()
                    evaluations[name].append(file_metrics(record, prediction, y))
                if (i+1)%9 == 0:
                    print(json.dumps({'stage': 'original-fit-inner-held-cpu', 'files': i+1, 'total': 54}), flush=True)
    except (TimeoutError, MemoryError) as error:
        failure = {'status': type(error).__name__, 'message': str(error)}
    integrity()
    summary = {n: summarize(v) for n, v in evaluations.items()} if failure is None else {}
    checks, interval = {}, None
    if failure is None:
        base, candidate = summary['unchanged'], summary['constrained']
        delta = np.array([r['peak_error'] for r in evaluations['constrained']])-np.array([r['peak_error'] for r in evaluations['unchanged']])
        rng = np.random.default_rng(1017)
        interval = np.quantile(delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1), [.025, .975]).tolist()
        checks = {'peak10pct_vs_controls': all(candidate['peak_p95'] <= .9*summary[n]['peak_p95'] for n in ('unchanged', 'dense_ridge')),
                  'global_esr_no_regression': candidate['global_esr'] <= base['global_esr'],
                  'worst_group_no_regression': max(candidate['by_fuzz_blend'].values()) <= max(base['by_fuzz_blend'].values()),
                  'paired_mean_bootstrap_upper_below_zero': interval[1] < 0}
    go = failure is None and all(checks.values())
    report = {'schema': 1, 'source_sha256': hashes, 'original_fit_manifest': records, 'inner_fit_files': 180, 'inner_held_files': 54,
              'source_and_audio_reverified': True, 'fit_compute': 'frozenMPS recurrence, CPU features/QP', 'held_compute': 'cpu-float32',
              'fit_batch_size': 9, 'fit_block_frames': 4096,
              'osqp_version': osqp.__version__, 'osqp_algebra': 'builtin', 'ridge': 1e-3, 'fit_tail_limit': PEAK_LIMIT,
              'fit_group_tail_limit': GROUP_LIMIT, 'failure': failure, 'history': history, 'summaries': summary, 'per_file': evaluations,
              'decision': {'continue_to_full_fit': go, 'checks': checks, 'paired_mean_delta_bootstrap95': interval},
              'coefficients': weight.tolist() if go else None, 'elapsed_seconds': time.perf_counter()-started,
              'physical_audio_devices_used': False, 'original_calibration_audio_opened': False, 'official_eval_opened': False,
              'source_audio_modified': False, 'audio_or_hidden_cache_saved': False, 'runtime_gain_or_limiter_added': False,
              'admitted': False, 'interpretation': 'Conservative aligned peak CVaR is not the original maximum-difference P95; only original complete CPU gates can admit.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'failure': failure, 'summaries': summary, 'decision': report['decision']}), flush=True)


if __name__ == '__main__':
    main()
