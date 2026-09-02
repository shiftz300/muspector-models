"""Fit a compact slow-dynamics correction using training Dry/Wet only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .asrnn_effects import effect_files, read_effect_pair
from .fit_asrnn_effect_output import _partition
from .stable_charge_residual import ChargeBank, ChargeReadout
from .stable_effect import load_stable_effect
from .stable_nonlinear_readout import StableNonlinearReadout
from .train_asrnn_phase7 import _evaluate, _rows


@torch.inference_mode()
def features(base, bank, paths):
    rows = []
    for offset in range(0, len(paths), 4):
        pairs = [read_effect_pair(path, 'dfz') for path in paths[offset:offset+4]]
        dry_cpu = torch.from_numpy(np.stack([p[0] for p in pairs]))
        dry = dry_cpu.to('mps')
        controls = torch.from_numpy(np.stack([p[2] for p in pairs])).to('mps')
        state = charge_state = None
        original, charges = [], []
        for start in range(0, dry.shape[1], 8192):
            if isinstance(base, StableNonlinearReadout):
                hidden,state=base.base.encode(dry[:,start:start+8192],controls,state)
                value=base.base.output_layer(hidden).squeeze(-1)+base.readout.at_training_knots(hidden,controls)
            else:
                value,state=base(dry[:,start:start+8192],controls,state)
            charge, charge_state = bank(dry_cpu[:, start:start+8192], charge_state)
            original.append(value.cpu()); charges.append(charge.cpu())
        predictions = torch.cat(original, 1)[:, 1024:]
        feature = torch.cat(charges, 1)[:, 1024:]
        for prediction, charge, (x, y, c) in zip(predictions, feature, pairs):
            wet = torch.from_numpy(y[1024:]); windows = []
            for audio in (wet, prediction):
                score = audio.abs().clone()
                for _ in range(4):
                    center = int(score.argmax()); start = max(0, min(len(audio)-256, center-128))
                    windows.append(torch.arange(start, start+256))
                    score[max(0, center-256):min(len(audio), center+256)] = -1
            ids = torch.cat((torch.linspace(0, len(wet)-1, 2048).long(), *windows))
            rows.append({'features': charge[ids].clone(), 'wet': wet[ids].clone(), 'original': prediction[ids].clone(),
                         'controls': torch.from_numpy(c), 'peak': wet.abs().max(), 'energy': wet.square().mean().clamp_min(1e-8)})
        if offset % 32 == 0:
            print(json.dumps({'stage': 'charge-features', 'completed': offset+len(pairs), 'total': len(paths)}), flush=True)
    return {key: torch.stack([row[key] for row in rows]).to('mps') for key in rows[0]}


@torch.inference_mode()
def evaluate(readout, data):
    predicted = data['original'] + readout.at_training_knots(data['features'], data['controls'])
    peak = (predicted.abs().amax(1) - data['peak']).abs().cpu().numpy()
    esr = ((predicted[:, :2048] - data['wet'][:, :2048]).square().mean(1) / data['energy']).cpu().numpy()
    return {'sampled_peak_error_p95': float(np.quantile(peak, .95)), 'sampled_mean_esr': float(esr.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('remix/runs/dfz-nonlinear-readout-phase9/candidate.pt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4000)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('output exists')
    args.output.mkdir(parents=True); torch.set_num_threads(2); torch.manual_seed(947)
    rng = np.random.default_rng(947)
    base, payload = load_stable_effect(args.source); base = base.to('mps')
    bank = ChargeBank(); readout = ChargeReadout(bank.width+1).to('mps')
    fit, cal = _partition(effect_files(Path('data/corpus/asrnn-physical-effects'), 'dfz', 'train'))
    train = features(base, bank, fit); validation = features(base, bank, cal)
    base.cpu(); bank.cpu(); torch.mps.empty_cache()
    optimizer = torch.optim.AdamW(readout.parameters(), lr=1e-3, weight_decay=.01)
    history = [{'step': 0, **evaluate(readout, validation)}]
    best = history[0]['sampled_peak_error_p95']; best_step = 0

    def save():
        torch.save({'schema': 10, 'architecture': 'stable-dfz-charge-residual', 'device': 'dfz', 'sample_rate': 48000,
                    'base': payload, 'charge_milliseconds': bank.milliseconds,
                    'readout_state_dict': {key: value.detach().cpu().clone() for key, value in readout.state_dict().items()}}, args.output/'candidate.pt')

    save(); print(json.dumps(history[0]), flush=True)
    for step in range(1, args.steps+1):
        ids = torch.from_numpy(rng.integers(0, len(fit), 32)).to('mps')
        data = {key: value[ids] for key, value in train.items()}
        predicted = data['original'] + readout.at_training_knots(data['features'], data['controls'])
        error = predicted - data['wet']
        waveform = error[:, :2048].square().mean(1) / data['energy']
        neighborhood = error[:, 2048:].square().mean(1) / data['energy']
        peak = (predicted.abs().amax(1) - data['peak']).abs()
        loss_per_file = waveform + neighborhood + 12*peak + peak/data['peak'].clamp_min(.02)
        loss = .5*loss_per_file.mean() + .5*loss_per_file.topk(8).values.mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(readout.parameters(), 2.); optimizer.step()
        if step % 200 == 0:
            row = {'step': step, 'loss': float(loss.detach()), **evaluate(readout, validation)}; history.append(row)
            score = row['sampled_peak_error_p95'] + max(0., row['sampled_mean_esr']-.03)
            if score < best: best = score; best_step = step; save()
            print(json.dumps(row), flush=True)
            (args.output/'training.json').write_text(json.dumps({'history': history, 'best_step': best_step}, indent=2)+'\n')
    model, _ = load_stable_effect(args.output/'candidate.pt')
    complete = _evaluate(model, _rows(cal, 'dfz'), torch.device('cpu'), 8)
    report = {'schema': 1, 'calibration': complete, 'calibration_compute': 'cpu', 'admitted_for_official_eval': complete['passes_selection_gate'],
              'fit_examples': len(fit), 'calibration_examples': len(cal), 'best_step': best_step, 'history': history,
              'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(), 'checkpoint_sha256': hashlib.sha256((args.output/'candidate.pt').read_bytes()).hexdigest(),
              'charge_milliseconds': bank.milliseconds, 'source_audio_modified': False, 'physical_audio_devices_used': False,
              'charge_feature_compute': 'cpu', 'frozen_core_feature_compute': 'mps', 'feature_frames_per_block': 8192,
              'automatic_normalization': False, 'automatic_limiting': False, 'official_eval_used_for_selection': False}
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
