"""Fit-only held diagnostic for the manufacturer's two-fuzz-circuit mixture.

No coefficients are fitted. At Fuzz Blend=0.5, compare the retained model's
ordinary conditional render with a fixed average of renders at the two circuit
endpoints. Original calibration/eval audio stays closed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .asrnn_effects import read_effect_pair
from .diagnose_dfz_peak_constraints import file_metrics
from .diagnose_dfz_readout_parity import MANIFEST, SOURCE, SOURCE_HASH, partition, summarize
from .precheck_multirate_fuzz import sha256
from .stable_effect import load_stable_effect

BUDGET_SECONDS = 600
REFERENCE = Path('remix/runs/dfz-readout-parity-phase11/metrics.json')


def fixed_circuit_mix(left: np.ndarray, right: np.ndarray, blend: float) -> np.ndarray:
    if left.shape != right.shape or not 0 <= blend <= 1:
        raise ValueError('matching circuit renders and normalized blend required')
    return (left * (1-blend) + right * blend).astype(np.float32)


@torch.inference_mode()
def render_controls(model, dry: np.ndarray, controls: np.ndarray) -> np.ndarray:
    condition = torch.from_numpy(np.asarray(controls, dtype=np.float32))[None]
    state, pieces = None, []
    for start in range(0, len(dry), 4096):
        value, state = model(torch.from_numpy(dry[start:start+4096])[None], condition, state)
        pieces.append(value[0])
    return torch.cat(pieces).numpy()


def bootstrap_upper(candidate: list[dict], baseline: list[dict]) -> tuple[float, float]:
    delta = np.array([c['peak_error']-b['peak_error'] for c, b in zip(candidate, baseline)])
    rng = np.random.default_rng(1301)
    means = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)
    return tuple(float(v) for v in np.quantile(means, [.025, .975]))


def subset_summary(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('nonempty metric subset required')
    return {'examples': len(rows),
            'global_esr': sum(r['error_energy'] for r in rows)/sum(r['wet_energy'] for r in rows),
            'peak_p95': float(np.quantile([r['peak_error'] for r in rows], .95)),
            'mean_peak_error': float(np.mean([r['peak_error'] for r in rows]))}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh compact report required')
    started = time.perf_counter()
    torch.set_num_threads(2)
    records = json.loads(MANIFEST.read_text())['audio_provenance']['fit']
    _, held = partition(records)
    sources = [SOURCE, MANIFEST, REFERENCE, Path(__file__), Path('remix/stable_effect.py'),
               Path('remix/stable_nonlinear_readout.py'), Path('remix/asrnn_effects.py')]
    hashes = {str(path): sha256(path) for path in sources}
    if hashes[str(SOURCE)] != SOURCE_HASH:
        raise ValueError('retained source changed')
    model, _ = load_stable_effect(SOURCE)
    reference = json.loads(REFERENCE.read_text())
    baseline_by_file = {row['file']: row for row in reference['per_file']['unchanged']}
    if set(baseline_by_file) != {Path(r['path']).name for r in held}:
        raise ValueError('reference does not match fixed inner-held files')
    baseline, mixture, middle_baseline, middle_mixture = [], [], [], []
    for index, record in enumerate(held):
        if time.perf_counter()-started > BUDGET_SECONDS:
            raise TimeoutError('fixed10-minute diagnostic budget exhausted')
        dry, wet, controls = read_effect_pair(Path(record['path']), 'dfz')
        if len(dry) != 144000 or not all(np.isfinite(v).all() for v in (dry, wet, controls)):
            raise ValueError('finite complete original record required')
        ordinary_row = dict(baseline_by_file[Path(record['path']).name])
        mixture_row = ordinary_row
        if float(controls[0]) == .5:
            left = render_controls(model, dry, np.array([0., controls[1]], np.float32))
            right = render_controls(model, dry, np.array([1., controls[1]], np.float32))
            candidate = fixed_circuit_mix(left, right, .5)
            mixture_row = file_metrics(record, candidate[1024:], wet[1024:])
        baseline.append(ordinary_row); mixture.append(mixture_row)
        if float(controls[0]) == .5:
            middle_baseline.append(ordinary_row); middle_mixture.append(mixture_row)
        if (index+1) % 9 == 0:
            print(json.dumps({'stage': 'cpu-held-circuit-diagnostic', 'files': index+1, 'total': len(held)}), flush=True)
    base_mid, candidate_mid = subset_summary(middle_baseline), subset_summary(middle_mixture)
    interval = bootstrap_upper(middle_mixture, middle_baseline)
    checks = {
        'middle_peak_p95_improves_10pct': candidate_mid['peak_p95'] <= .9*base_mid['peak_p95'],
        'middle_global_esr_no_regression': candidate_mid['global_esr'] <= base_mid['global_esr'],
        'middle_paired_mean_bootstrap_upper_below_zero': interval[1] < 0,
        'full_peak_p95_no_regression': summarize(mixture)['peak_p95'] <= summarize(baseline)['peak_p95'],
        'full_global_esr_no_regression': summarize(mixture)['global_esr'] <= summarize(baseline)['global_esr'],
    }
    if any(sha256(path) != digest for path, digest in hashes.items()) or any(
            sha256(r['path']) != r['sha256'] for r in records):
        raise ValueError('source, code, manifest, or original fit audio changed')
    report = {'schema': 1, 'hypothesis': 'fixed endpoint mixture reflects two discrete fuzz circuits',
              'manufacturer_reference': 'https://www.darkglass.com/en-int/pages/duality-fuzz-manual',
              'source_sha256': hashes, 'inner_held_files': len(held), 'middle_files': len(middle_baseline),
              'summaries': {'full_ordinary': summarize(baseline), 'full_fixed_mixture': summarize(mixture),
                            'middle_ordinary': base_mid, 'middle_fixed_mixture': candidate_mid},
              'middle_paired_mean_peak_delta_bootstrap95': interval, 'checks': checks,
              'authorize_dual_expert_training': all(checks.values()), 'per_file': {'ordinary': baseline, 'fixed_mixture': mixture},
              'elapsed_seconds': time.perf_counter()-started, 'weights_saved': False,
              'original_calibration_audio_opened': False, 'official_eval_opened': False,
              'physical_audio_devices_used': False, 'source_audio_modified': False,
              'audio_or_hidden_cache_saved': False, 'admitted': False,
              'limitation': 'Diagnostic of retained endpoint renders; not a trained two-expert model or original quality gate.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'summaries': report['summaries'], 'checks': checks,
                      'authorize_dual_expert_training': report['authorize_dual_expert_training']}), flush=True)


if __name__ == '__main__':
    main()
