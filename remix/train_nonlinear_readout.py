"""Train a zero-preserving nonlinear readout, keeping the source RNN frozen."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from .asrnn_effects import effect_files, read_effect_pair
from .fit_asrnn_effect_output import _partition, _hidden
from .stable_effect import load_stable_effect
from .stable_nonlinear_readout import NonlinearReadout
from .train_asrnn_phase7 import _evaluate, _rows
from .stable_skip_readout import encode_skip


@torch.inference_mode()
def features(base, paths, *, skip_connections=False, peak_neighborhoods=False):
    rows=[]
    for offset in range(0,len(paths),4):
        pairs=[read_effect_pair(p,'dfz') for p in paths[offset:offset+4]]
        dry=torch.from_numpy(np.stack([v[0] for v in pairs])).to('mps')
        control=torch.from_numpy(np.stack([v[2] for v in pairs])).to('mps')
        if skip_connections:
            chunks=[];state=None
            for start in range(0,dry.shape[1],2048):
                value,state=encode_skip(base,dry[:,start:start+2048],control,state);chunks.append(value)
            hidden=torch.cat(chunks,1)[:,1024:].cpu()
        else:hidden=_hidden(base,dry,control).cpu()
        weight=base.output_layer.weight.detach().cpu().flatten()
        for h,(x,y,c) in zip(hidden,pairs):
            target=torch.from_numpy(y[1024:]);original=h[:,-base.hidden_size:]@weight
            uniform=torch.linspace(0,len(target)-1,2048).long()
            if peak_neighborhoods:
                windows=[]
                for audio in (target,original):
                    score=audio.abs().clone()
                    for _ in range(4):
                        center=int(score.argmax());start=max(0,min(len(audio)-256,center-128))
                        windows.append(torch.arange(start,start+256));score[max(0,center-256):min(len(audio),center+256)]=-1
                ids=torch.cat((uniform,*windows))
            else:ids=torch.cat((uniform,target.abs().topk(1024).indices,original.abs().topk(1024).indices))
            rows.append({'hidden':h[ids].clone(),'wet':target[ids].clone(),'original':original[ids].clone(),'peak':target.abs().max(),'energy':target.square().mean().clamp_min(1e-8),'controls':torch.from_numpy(c)})
        if offset%32==0:print(json.dumps({'stage':'nonlinear-readout-features','completed':offset+len(pairs),'total':len(paths)}),flush=True)
    return {key:torch.stack([r[key] for r in rows]).to('mps') for key in rows[0]}


@torch.inference_mode()
def evaluate(readout, data):
    peaks=[];errors=[]
    for start in range(0,len(data['wet']),8):
        d={k:v[start:start+8] for k,v in data.items()}
        prediction=d['original']+readout.at_training_knots(d['hidden'],d['controls'])
        peaks.extend((prediction.abs().amax(1)-d['peak']).abs().cpu().tolist())
        errors.extend(((prediction[:,:2048]-d['wet'][:,:2048]).square().mean(1)/d['energy']).cpu().tolist())
    return {'sampled_peak_error_p95':float(np.quantile(peaks,.95)),'sampled_mean_esr':float(np.mean(errors))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--skip-connections',action='store_true');p.add_argument('--peak-neighborhoods',action='store_true')
    p.add_argument('--readout-width',type=int,default=16)
    p.add_argument('--initialize',type=Path,help='Continue a matching frozen-core readout without resetting learned corrections')
    p.add_argument('--learning-rate',type=float,default=1e-4)
    args=p.parse_args()
    if args.output.exists():raise ValueError('output exists')
    args.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(927);rng=np.random.default_rng(927)
    source=Path('remix/runs/dfz-capacity-phase9/selected.pt');base,payload=load_stable_effect(source);base=base.to('mps')
    fit,cal=_partition(effect_files(Path('data/corpus/asrnn-physical-effects'),'dfz','train'))
    train=features(base,fit,skip_connections=args.skip_connections,peak_neighborhoods=args.peak_neighborhoods)
    validation=features(base,cal,skip_connections=args.skip_connections,peak_neighborhoods=args.peak_neighborhoods);base=base.cpu();torch.mps.empty_cache()
    readout=NonlinearReadout(train['hidden'].shape[-1],args.readout_width).to('mps')
    if args.initialize:
        _,initial=load_stable_effect(args.initialize)
        if initial.get('schema') != (8 if args.skip_connections else 7) or initial['readout_width'] != args.readout_width:
            raise ValueError('initializer readout architecture differs')
        if initial['base'].keys()!=payload.keys() or any(not torch.equal(initial['base']['state_dict'][key],value) for key,value in payload['state_dict'].items()):
            raise ValueError('initializer does not share the exact frozen core')
        for key in payload:
            if key!='state_dict' and initial['base'][key]!=payload[key]:raise ValueError('initializer core metadata differs')
        readout.load_state_dict(initial['readout_state_dict'],strict=True)
    optimizer=torch.optim.AdamW(readout.parameters(),lr=args.learning_rate,weight_decay=1e-3)
    best=float('inf');history=[]
    if args.initialize:
        baseline={'step':0,**evaluate(readout,validation)};history.append(baseline)
        best=baseline['sampled_peak_error_p95']+max(0,baseline['sampled_mean_esr']-.03);best_step=0
        torch.save(initial,args.output/'candidate.pt')
        print(json.dumps(baseline),flush=True)
    for step in range(args.steps):
        ids=torch.from_numpy(rng.integers(0,len(fit),16)).to('mps');d={k:v[ids] for k,v in train.items()}
        prediction=d['original']+readout.at_training_knots(d['hidden'],d['controls'])
        waveform=(prediction[:,:2048]-d['wet'][:,:2048]).square().mean(1)/d['energy']
        peak=(prediction.abs().amax(1)-d['peak']).abs()
        per_file=waveform+12*peak+peak/d['peak'].clamp_min(.02)
        if args.peak_neighborhoods:
            # Contiguous windows, unlike top-k amplitude-sorted points, support
            # true signed waveform and local temporal-difference supervision.
            predicted=prediction[:,2048:].reshape(-1,8,256);expected=d['wet'][:,2048:].reshape(-1,8,256)
            neighborhood=(predicted-expected).square().mean((1,2))/d['energy']
            dp=predicted[:,:,1:]-.95*predicted[:,:,:-1];dt=expected[:,:,1:]-.95*expected[:,:,:-1]
            emphasized=(dp-dt).square().mean((1,2))/dt.square().mean((1,2)).clamp_min(1e-7)
            per_file=per_file+neighborhood+.25*emphasized
        loss=.5*per_file.mean()+.5*per_file.topk(4).values.mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(readout.parameters(),2.);optimizer.step()
        if (step+1)%200==0:
            row={'step':step+1,'loss':float(loss.detach()),**evaluate(readout,validation)};history.append(row)
            score=row['sampled_peak_error_p95']+max(0,row['sampled_mean_esr']-.03)
            if score<best:
                best=score;best_step=step+1
                torch.save({'schema':8 if args.skip_connections else 7,'architecture':'stable-skip-nonlinear-readout' if args.skip_connections else 'stable-nonlinear-readout','device':'dfz','sample_rate':48000,'base':payload,'readout_width':args.readout_width,'readout_state_dict':{k:v.detach().cpu().clone() for k,v in readout.state_dict().items()}},args.output/'candidate.pt')
            print(json.dumps(row),flush=True)
            (args.output/'training.json').write_text(json.dumps({'history':history,'best_step':best_step},indent=2)+'\n')
    model,_=load_stable_effect(args.output/'candidate.pt');complete=_evaluate(model,_rows(cal,'dfz'),torch.device('cpu'),8)
    report={'schema':1,'calibration':complete,'admitted_for_official_eval':complete['passes_selection_gate'],'best_step':best_step,'fit_examples':len(fit),'calibration_examples':len(cal),'history':history,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256((args.output/'candidate.pt').read_bytes()).hexdigest(),'physical_audio_devices_used':False,'source_audio_modified':False,'official_eval_used_for_selection':False}
    report.update(skip_connections=args.skip_connections,peak_neighborhood_supervision=args.peak_neighborhoods)
    report.update(initializer_sha256=hashlib.sha256(args.initialize.read_bytes()).hexdigest() if args.initialize else None,learning_rate=args.learning_rate)
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
