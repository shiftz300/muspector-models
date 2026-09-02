"""Fit signed-waveform charge residuals with a small frozen ridge menu."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from .asrnn_effects import effect_files
from .fit_asrnn_effect_output import _partition
from .stable_charge_residual import ChargeBank, ChargeReadout
from .stable_effect import load_stable_effect
from .train_asrnn_phase7 import _evaluate, _rows
from .train_dfz_charge import features


def signature(source, fit, cal):
    root=Path(__file__).parent
    return {'schema':1,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'feature_function_sha256':hashlib.sha256(inspect.getsource(features).encode()).hexdigest(),
            'code_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('stable_charge_residual.py','stable_effect.py','stable_nonlinear_readout.py','asrnn_effects.py')},
            'partitions':{name:[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths] for name,paths in (('fit',fit),('calibration',cal))}}


@torch.inference_mode()
def quality(weight, data, mask=None):
    ids=(data['controls']*2).round().long();corner=ids[:,0]*3+ids[:,1]
    predicted=data['original']+(data['features']*weight[corner,None]).sum(-1)
    errors=(predicted.abs().amax(1)-data['peak']).abs()
    esr=(predicted[:,:2048]-data['wet'][:,:2048]).square().mean(1)/data['energy']
    if mask is not None:errors,esr=errors[mask],esr[mask]
    return {'peak_error_p95':float(torch.quantile(errors,.95)),'mean_esr':float(esr.mean())}


def objective(row):
    return row['peak_error_p95']+max(0.,row['mean_esr']-.03)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path('remix/runs/dfz-nonlinear-readout-phase9/candidate.pt'))
    args=parser.parse_args();torch.set_num_threads(2)
    if args.output.exists():raise ValueError('output exists')
    args.output.mkdir(parents=True)
    fit,cal=_partition(effect_files(Path('data/corpus/asrnn-physical-effects'),'dfz','train'))
    expected_signature=signature(args.source,fit,cal)
    base,payload=load_stable_effect(args.source);bank=ChargeBank()
    if args.cache.exists():
        cached=torch.load(args.cache,map_location='cpu',weights_only=True)
        if cached['signature']!=expected_signature:raise ValueError('charge feature cache source/protocol changed')
        train,validation=cached['fit'],cached['calibration']
    else:
        base=base.to('mps')
        train={k:v.cpu() for k,v in features(base,bank,fit).items()}
        validation={k:v.cpu() for k,v in features(base,bank,cal).items()}
        base.cpu();torch.mps.empty_cache()
        args.cache.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'signature':expected_signature,'fit':train,'calibration':validation},args.cache)
    if signature(args.source,fit,cal)!=expected_signature:raise ValueError('source changed during feature extraction')
    corners=(train['controls']*2).round().long();corners=corners[:,0]*3+corners[:,1]
    candidates={'unchanged':torch.zeros(9,bank.width+1)}
    for neighborhood_weight in (0.,1.,4.):
        for ridge in (1e-4,1e-2,1.,10.):
            weights=[]
            for corner in range(9):
                mask=corners==corner
                x=train['features'][mask].double();y=(train['wet'][mask]-train['original'][mask]).double()
                scale=train['energy'][mask].double().rsqrt()[:,None]
                time_weight=torch.ones(x.shape[1],dtype=torch.float64);time_weight[2048:]=neighborhood_weight**.5
                factor=scale*time_weight[None]
                matrix=(x*factor[...,None]).reshape(-1,x.shape[-1]);target=(y*factor).flatten()
                xx=matrix.T@matrix;xy=matrix.T@target
                regularization=max(float(xx.trace()/len(xx))*ridge,1e-10)
                weights.append(torch.linalg.solve(xx+torch.eye(len(xx),dtype=torch.float64)*regularization,xy).float())
            candidates[f'neighborhood-{neighborhood_weight:g}-ridge-{ridge:g}']=torch.stack(weights)
    scores={name:quality(weight,validation) for name,weight in candidates.items()}
    global_choice=min(scores,key=lambda name:objective(scores[name]))
    # A second predeclared candidate chooses a ridge setting per known control
    # corner on calibration. This is fixed before development, never chosen from
    # an incoming clip's Wet signal, loudness, filename or target peak.
    cal_corners=(validation['controls']*2).round().long();cal_corners=cal_corners[:,0]*3+cal_corners[:,1]
    corner_choices=[];corner_weights=[]
    for corner in range(9):
        chosen=min(candidates,key=lambda name:objective(quality(candidates[name],validation,cal_corners==corner)))
        corner_choices.append(chosen);corner_weights.append(candidates[chosen][corner])
    candidates['calibrated-corner-ridge']=torch.stack(corner_weights)
    scores['calibrated-corner-ridge']=quality(candidates['calibrated-corner-ridge'],validation)
    selected=min(scores,key=lambda name:objective(scores[name]));readout=ChargeReadout(bank.width+1)
    with torch.no_grad():readout.weight.copy_(candidates[selected])
    output={'schema':10,'architecture':'stable-dfz-charge-residual','device':'dfz','sample_rate':48000,
            'base':payload,'charge_milliseconds':bank.milliseconds,'readout_state_dict':readout.state_dict()}
    torch.save(output,args.output/'candidate.pt')
    model,_=load_stable_effect(args.output/'candidate.pt');full=_evaluate(model,_rows(cal,'dfz'),torch.device('cpu'),8)
    report={'schema':1,'calibration':full,'admitted_for_official_eval':full['passes_selection_gate'],
            'selected':selected,'global_ridge_choice':global_choice,'calibrated_corner_choices':corner_choices,'sampled_calibration':scores,
            'fit_examples':len(fit),'calibration_examples':len(cal),'cache_signature_sha256':hashlib.sha256(json.dumps(expected_signature,sort_keys=True).encode()).hexdigest(),
            'source_sha256':expected_signature['source_sha256'],'checkpoint_sha256':hashlib.sha256((args.output/'candidate.pt').read_bytes()).hexdigest(),
            'official_eval_used_for_selection':False,'physical_audio_devices_used':False,'source_audio_modified':False,
            'automatic_normalization':False,'automatic_limiting':False}
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
