"""Fit conditional DFZ readouts on fit-only audio and select on calibration."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .stable_effect import load_stable_effect
from .asrnn_effects import effect_files
from .evaluate_asrnn_effect import EffectClips
from .fit_asrnn_effect_output import _partition,_hidden,_solve,PEAK_WEIGHTS


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('output already exists')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    model,payload=load_stable_effect(args.checkpoint)
    fit,calibration=_partition(effect_files(Path('data/corpus/asrnn-physical-effects'),'dfz','train'))
    width=model.hidden_size
    xx=torch.zeros(3,3,width,width,dtype=torch.float64);xy=torch.zeros(3,3,width,dtype=torch.float64)
    pxx=torch.zeros_like(xx);pxy=torch.zeros_like(xy)
    for index,batch in enumerate(DataLoader(EffectClips(fit,'dfz'),batch_size=4)):
        hidden=_hidden(model,batch['dry'],batch['controls']).double();wet=batch['wet'][:,1024:].double()
        for h,y,c in zip(hidden,wet,batch['controls']):
            i,j=(c*2).round().int().tolist()
            xx[i,j]+=h.T@h;xy[i,j]+=h.T@y
            dh=h[1:]-.95*h[:-1];dy=y[1:]-.95*y[:-1]
            xx[i,j]+=.25*(dh.T@dh);xy[i,j]+=.25*(dh.T@dy)
            peaks=y.abs().topk(128).indices
            hp,yp=h[peaks],y[peaks]
            pxx[i,j]+=hp.T@hp;pxy[i,j]+=hp.T@yp
        if index%8==0: print(json.dumps({'stage':'conditional-fit','batch':index+1,'total':59}),flush=True)
    candidates={'original':model.output_layer.weight.detach().reshape(1,1,-1).expand(3,3,-1)}
    for peak in PEAK_WEIGHTS:
        candidates[str(peak)]=torch.stack([_solve(xx[i,j],xy[i,j],pxx[i,j],pxy[i,j],peak) for i in range(3) for j in range(3)]).reshape(3,3,-1)
    accumulators={key:{'esr':[],'peak':[],'error':0.,'energy':0.} for key in candidates}
    for batch in DataLoader(EffectClips(calibration,'dfz'),batch_size=4):
        hidden=_hidden(model,batch['dry'],batch['controls']);wet=batch['wet'][:,1024:]
        indices=(batch['controls']*2).round().long()
        for key,weights in candidates.items():
            predicted=(hidden*weights[indices[:,0],indices[:,1]][:,None,:]).sum(-1)
            error=(predicted-wet).square();a=accumulators[key]
            a['esr'].extend((error.mean(1)/wet.square().mean(1).clamp_min(1e-8)).tolist())
            a['peak'].extend((predicted.abs().amax(1)-wet.abs().amax(1)).abs().tolist())
            a['error']+=float(error.sum());a['energy']+=float(wet.square().sum())
    reports={}
    for key,a in accumulators.items():
        r={'global_esr':a['error']/a['energy'],'mean_per_file_esr':float(np.mean(a['esr'])),'p95_per_file_esr':float(np.quantile(a['esr'],.95)),'absolute_peak_error_p95':float(np.quantile(a['peak'],.95))}
        r['passed']=r['global_esr']<=.05 and r['mean_per_file_esr']<=.1 and r['p95_per_file_esr']<=.25 and r['absolute_peak_error_p95']<=.02
        reports[key]=r
    selected=min(reports,key=lambda k:(not reports[k]['passed'],reports[k]['absolute_peak_error_p95'],reports[k]['global_esr']))
    result={'schema':1,'candidates':reports,'selected':selected,'admitted_for_official_eval':reports[selected]['passed'],'fit_examples':len(fit),'calibration_examples':len(calibration),'source_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'physical_audio_devices_used':False,'official_eval_used_for_selection':False}
    if reports[selected]['passed']:
        artifact=args.output/'dfz-stable.pt'
        torch.save({'schema':4,'architecture':'stable-conditioned-readout','device':'dfz','sample_rate':48000,'base':payload,'readout':candidates[selected],'fit_provenance':result},artifact)
        result['checkpoint_sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest()
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
