"""Fit a causal nonlinear FIR residual to recover DFZ fast transients."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from .asrnn_effects import effect_files,read_effect_pair
from .fit_asrnn_effect_output import _partition
from .stable_effect import load_stable_effect,StableEffectFIRResidual
from .train_asrnn_phase7 import _evaluate,_rows


@torch.inference_mode()
def render(model,dry,controls):
    state=None;chunks=[]
    for start in range(0,dry.shape[1],2048):
        y,state=model(dry[:,start:start+2048],controls,state);chunks.append(y)
    return torch.cat(chunks,1)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--taps',type=int,default=33)
    args=p.parse_args()
    if args.output.exists():raise ValueError('output already exists')
    args.output.mkdir(parents=True);torch.set_num_threads(2)
    source=Path('remix/runs/dfz-capacity-phase9/selected.pt');model,payload=load_stable_effect(source);model=model.to('mps')
    fit,cal=_partition(effect_files(Path('data/corpus/asrnn-physical-effects'),'dfz','train'))
    width=6*args.taps
    xx=torch.zeros(9,width,width,dtype=torch.float64);xy=torch.zeros(9,width,dtype=torch.float64)
    pxx=torch.zeros_like(xx);pxy=torch.zeros_like(xy)
    for offset in range(0,len(fit),8):
        pairs=[read_effect_pair(path,'dfz') for path in fit[offset:offset+8]]
        dry=torch.from_numpy(np.stack([v[0] for v in pairs]));wet=torch.from_numpy(np.stack([v[1] for v in pairs]));controls=torch.from_numpy(np.stack([v[2] for v in pairs]))
        predicted=render(model,dry.to('mps'),controls.to('mps')).cpu()
        basis=StableEffectFIRResidual.basis(dry)
        for b,(x,y,c) in enumerate(pairs):
            group=int(round(float(c[0])*2))*3+int(round(float(c[1])*2))
            uniform=torch.linspace(1024,len(x)-1,4096).long()
            peaks=wet[b,1024:].abs().topk(256).indices+1024
            def add(indices,mat,vec):
                lags=indices[:,None]-torch.arange(args.taps)[None,:]
                features=basis[b,:,lags].permute(1,0,2).reshape(len(indices),-1).double()
                target=(wet[b,indices]-predicted[b,indices]).double()
                mat[group]+=features.T@features;vec[group]+=features.T@target
            add(uniform,xx,xy);add(peaks,pxx,pxy)
        if offset%32==0:print(json.dumps({'stage':'FIR-fit','completed':offset+len(pairs),'total':len(fit)}),flush=True)
    candidates={}
    for peak in (0.,16.,64.,256.):
        filters=[]
        for i in range(9):
            matrix=xx[i]+peak*pxx[i];target=xy[i]+peak*pxy[i]
            ridge=float(matrix.trace())/width*1e-4
            filters.append(torch.linalg.solve(matrix+torch.eye(width,dtype=torch.float64)*max(ridge,1e-9),target).float().reshape(6,args.taps))
        candidates[str(peak)]=torch.stack(filters)
    model=model.cpu();reports={}
    # The candidate pool is tiny and evaluated on complete clips, never a
    # selected subset of target peaks or development recordings.
    for key,filters in candidates.items():
        candidate=StableEffectFIRResidual(model,filters)
        reports[key]=_evaluate(candidate,_rows(cal,'dfz'),torch.device('cpu'),8)
        print(json.dumps({'candidate':key,'calibration':reports[key]}),flush=True)
    selected=min(reports,key=lambda k:(not reports[k]['passes_selection_gate'],reports[k]['absolute_peak_error_p95'],reports[k]['global_esr']))
    result={'schema':1,'candidates':reports,'selected':selected,'admitted_for_official_eval':reports[selected]['passes_selection_gate'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'fit_examples':len(fit),'calibration_examples':len(cal),'physical_audio_devices_used':False,'source_audio_modified':False,'official_eval_used_for_selection':False}
    artifact=args.output/'candidate.pt'
    torch.save({'schema':5,'architecture':'stable-causal-fir-residual','device':'dfz','sample_rate':48000,'base':payload,'filters':candidates[selected]},artifact)
    result['checkpoint_sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest()
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
