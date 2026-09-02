"""Train a causal TCN correction using full-prefix teacher residuals."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .asrnn_effects import effect_files,read_effect_pair
from .fit_asrnn_effect_output import _partition
from .fit_fir_residual import render
from .stable_effect import load_stable_effect
from .stable_tcn_residual import ZeroGatedTCN,StableTCNResidual
from .train_asrnn_phase7 import _evaluate,_rows


@torch.inference_mode()
def cache(model,paths):
    rows=[]
    for offset in range(0,len(paths),8):
        values=[read_effect_pair(p,'dfz') for p in paths[offset:offset+8]]
        dry=torch.from_numpy(np.stack([v[0] for v in values])).to('mps');controls=torch.from_numpy(np.stack([v[2] for v in values])).to('mps')
        predicted=render(model,dry,controls).cpu().numpy()
        rows.extend((x,y,c,p) for (x,y,c),p in zip(values,predicted))
        if offset%32==0:print(json.dumps({'stage':'full-prefix-residual-cache','completed':offset+len(values),'total':len(paths)}),flush=True)
    return rows


@torch.inference_mode()
def calibrate(tcn,rows):
    errors=[];peaks=[];total_error=total_energy=0.
    for offset in range(0,len(rows),8):
        batch=rows[offset:offset+8]
        dry=torch.from_numpy(np.stack([v[0] for v in batch])).to('mps');wet=torch.from_numpy(np.stack([v[1] for v in batch])).to('mps')
        controls=torch.from_numpy(np.stack([v[2] for v in batch])).to('mps');base=torch.from_numpy(np.stack([v[3] for v in batch])).to('mps')
        predicted=(render(tcn,dry,controls)+base)[:,1024:];target=wet[:,1024:]
        error=(predicted-target).square();energy=target.square()
        errors.extend((error.mean(1)/energy.mean(1).clamp_min(1e-8)).cpu().tolist());peaks.extend((predicted.abs().amax(1)-target.abs().amax(1)).abs().cpu().tolist())
        total_error+=float(error.sum());total_energy+=float(energy.sum())
    return {'global_esr':total_error/total_energy,'mean_per_file_esr':float(np.mean(errors)),'p95_per_file_esr':float(np.quantile(errors,.95)),'absolute_peak_error_p95':float(np.quantile(peaks,.95))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=2000)
    args=p.parse_args()
    if args.output.exists():raise ValueError('output already exists')
    args.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(924);rng=np.random.default_rng(924)
    source=Path('remix/runs/dfz-capacity-phase9/selected.pt');base,payload=load_stable_effect(source);base=base.to('mps')
    fit,cal=_partition(effect_files(Path('data/corpus/asrnn-physical-effects'),'dfz','train'))
    training=cache(base,fit);calibration=cache(base,cal);base=base.cpu();torch.mps.empty_cache()
    tcn=ZeroGatedTCN().to('mps');optimizer=torch.optim.AdamW(tcn.parameters(),lr=3e-4,weight_decay=1e-5)
    best=float('inf');history=[];started=time.perf_counter()
    for step in range(args.steps):
        inputs=[];targets=[];controls=[];original=[]
        for i in rng.integers(0,len(training),16):
            x,y,c,pred=training[i]
            if rng.random()<.6:
                peak=int(np.abs(y[1024:]).argmax())+1024;start=int(np.clip(peak-rng.integers(1024,4096),0,len(x)-4096))
            else:start=int(rng.integers(48000-1024,len(x)-4096))
            inputs.append(x[start:start+4096]);targets.append(y[start:start+4096]);controls.append(c);original.append(pred[start:start+4096])
        dry=torch.from_numpy(np.stack(inputs)).to('mps');wet=torch.from_numpy(np.stack(targets)).to('mps');c=torch.from_numpy(np.stack(controls)).to('mps');teacher=torch.from_numpy(np.stack(original)).to('mps')
        prediction=(tcn(dry,c)[0]+teacher)[:,1024:];target=wet[:,1024:]
        error=prediction-target;wave=error.square().mean(1)/target.square().mean(1).clamp_min(1e-6)
        dp=prediction[:,1:]-.95*prediction[:,:-1];dt=target[:,1:]-.95*target[:,:-1]
        pre=(dp-dt).square().mean(1)/dt.square().mean(1).clamp_min(1e-6)
        peak=(prediction.abs().amax(1)-target.abs().amax(1)).abs()
        per_file=wave+.25*pre+12*peak
        loss=.5*per_file.mean()+.5*per_file.topk(4).values.mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(tcn.parameters(),2.);optimizer.step()
        if (step+1)%100==0:print(json.dumps({'step':step+1,'loss':float(loss.detach()),'elapsed_seconds':time.perf_counter()-started}),flush=True)
        if (step+1)%400==0 or step+1==args.steps:
            report=calibrate(tcn,calibration);history.append({'step':step+1,**report})
            score=report['absolute_peak_error_p95']+max(0,report['global_esr']-.03)
            if score<best:
                best=score;best_state={k:v.detach().cpu().clone() for k,v in tcn.state_dict().items()};best_step=step+1
                torch.save({'schema':6,'architecture':'stable-zero-gated-tcn-residual','device':'dfz','sample_rate':48000,'base':payload,'tcn_width':16,'tcn_blocks':8,'tcn_state_dict':best_state},args.output/'candidate.pt')
            print(json.dumps({'calibration':history[-1]}),flush=True)
            (args.output/'training.json').write_text(json.dumps({'history':history,'best_step':best_step,'physical_audio_devices_used':False,'source_audio_modified':False,'official_eval_used_for_selection':False},indent=2)+'\n')
    model,_=load_stable_effect(args.output/'candidate.pt');complete=_evaluate(model,_rows(cal,'dfz'),torch.device('cpu'),8)
    result={'schema':1,'history':history,'best_step':best_step,'calibration':complete,'admitted_for_official_eval':complete['passes_selection_gate'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256((args.output/'candidate.pt').read_bytes()).hexdigest(),'fit_examples':len(fit),'calibration_examples':len(cal),'physical_audio_devices_used':False,'source_audio_modified':False,'official_eval_used_for_selection':False}
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
