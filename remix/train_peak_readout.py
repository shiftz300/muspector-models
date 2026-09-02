"""Peak-objective conditional readout training with frozen recurrent features."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .asrnn_effects import effect_files
from .fit_asrnn_effect_output import _partition,_hidden
from .evaluate_asrnn_effect import EffectClips
from .stable_effect import load_stable_effect,ConditionedReadoutEffect
from .train_asrnn_phase7 import _evaluate,_rows


@torch.inference_mode()
def features(model,paths):
    result=[]
    for index,batch in enumerate(DataLoader(EffectClips(paths,'dfz'),batch_size=4)):
        hidden=_hidden(model,batch['dry'],batch['controls']);wet=batch['wet'][:,1024:]
        original=model.output_layer(hidden).squeeze(-1)
        for h,y,p,c in zip(hidden,wet,original,batch['controls']):
            # Uniform samples constrain waveform fidelity; both target and
            # source peak neighborhoods expose the max objective to training.
            uniform=torch.linspace(0,len(y)-1,2048).long()
            indices=torch.cat((uniform,y.abs().topk(1024).indices,p.abs().topk(1024).indices))
            result.append({'hidden':h[indices].clone(),'wet':y[indices].clone(),'peak':y.abs().max(),'energy':y.square().mean().clamp_min(1e-8),'controls':c.clone()})
        if index%8==0: print(json.dumps({'stage':'peak-features','completed':min((index+1)*4,len(paths)),'total':len(paths)}),flush=True)
    return result


@torch.inference_mode()
def evaluate_cache(weights,rows):
    esr=[];peaks=[]
    for batch in DataLoader(rows,batch_size=16):
        ij=(batch['controls']*2).round().long()
        predicted=(batch['hidden']*weights[ij[:,0],ij[:,1]][:,None,:]).sum(-1)
        error=(predicted[:,:2048]-batch['wet'][:,:2048]).square().mean(1)/batch['energy']
        esr.extend(error.tolist());peaks.extend((predicted.abs().amax(1)-batch['peak']).abs().tolist())
    return {'sampled_mean_esr':float(np.mean(esr)),'sampled_p95_esr':float(np.quantile(esr,.95)),'sampled_peak_error_p95':float(np.quantile(peaks,.95))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=180)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('output already exists')
    args.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(915)
    model,payload=load_stable_effect(args.checkpoint)
    paths=effect_files(Path('data/corpus/asrnn-physical-effects'),'dfz','train');fit,calibration=_partition(paths)
    training=features(model,fit);calibrating=features(model,calibration)
    initial=model.output_layer.weight.detach().reshape(1,1,-1).expand(3,3,-1).clone()
    weights=torch.nn.Parameter(initial.clone());optimizer=torch.optim.Adam([weights],lr=3e-4)
    best=None;best_score=float('inf');history=[]
    for epoch in range(args.epochs):
        losses=[]
        for batch in DataLoader(training,batch_size=16,shuffle=True):
            ij=(batch['controls']*2).round().long()
            prediction=(batch['hidden']*weights[ij[:,0],ij[:,1]][:,None,:]).sum(-1)
            waveform=(prediction[:,:2048]-batch['wet'][:,:2048]).square().mean(1)/batch['energy']
            peaks=(prediction.abs().amax(1)-batch['peak']).abs()
            peak_loss=peaks*12+peaks/batch['peak'].clamp_min(.02)
            # CVaR-like emphasis is on bad training clips, never validation.
            per_file=waveform+peak_loss
            loss=.5*per_file.mean()+.5*per_file.topk(max(1,len(per_file)//4)).values.mean()+.1*(weights-initial).square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        if epoch%5==0 or epoch+1==args.epochs:
            values=evaluate_cache(weights,calibrating)
            score=values['sampled_peak_error_p95']+max(0,values['sampled_mean_esr']-.03)
            row={'epoch':epoch+1,'loss':float(np.mean(losses)),**values};history.append(row)
            if score<best_score: best_score=score;best=weights.detach().clone();best_epoch=epoch+1
            print(json.dumps(row),flush=True)
    candidate=ConditionedReadoutEffect(model,best).eval()
    complete=_evaluate(candidate,_rows(calibration,'dfz'),torch.device('cpu'),8)
    result={'schema':1,'history':history,'best_epoch':best_epoch,'calibration':complete,'admitted_for_official_eval':complete['passes_selection_gate'],'fit_examples':len(fit),'calibration_examples':len(calibration),'source_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'selection_approximation':'cached uniform and peak samples, followed by mandatory complete 3s calibration','physical_audio_devices_used':False,'source_audio_modified':False,'official_eval_used_for_selection':False}
    if complete['passes_selection_gate']:
        artifact=args.output/'dfz-stable.pt'
        torch.save({'schema':4,'architecture':'stable-conditioned-readout','device':'dfz','sample_rate':48000,'base':payload,'readout':best,'fit_provenance':result},artifact)
        result['checkpoint_sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest()
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'calibration':complete,'admitted':complete['passes_selection_gate']}),flush=True)


if __name__=='__main__': main()
