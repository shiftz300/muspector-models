"""Fine-tune stable DFZ recurrent dynamics on fit-only transient windows."""
from __future__ import annotations
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .asrnn_effects import effect_files,read_effect_pair
from .fit_asrnn_effect_output import _partition
from .stable_effect import load_stable_effect
from .train_asrnn_phase7 import _evaluate,_rows


@torch.no_grad()
def project(model, mirror_gates=False):
    h=model.hidden_size
    norms=[]
    for layer in model.rnn_layers:
        if mirror_gates:
            layer.weight_hh_l0[:h].copy_(-layer.weight_hh_l0[h:2*h])
            layer.weight_ih_l0[:h,-model.control_count:].copy_(-layer.weight_ih_l0[h:2*h,-model.control_count:])
            layer.bias_ih_l0[:h].copy_(-layer.bias_ih_l0[h:2*h])
            layer.bias_hh_l0[:h].copy_(-layer.bias_hh_l0[h:2*h])
        layer.weight_ih_l0[2*h:3*h,-model.control_count:].zero_()
        layer.bias_ih_l0[2*h:3*h].zero_();layer.bias_hh_l0[2*h:3*h].zero_()
        candidate=layer.weight_hh_l0[2*h:3*h]
        norm=float(candidate.abs().sum(1).max())
        if norm>.995:candidate.mul_(.995/norm)
        norms.append(float(candidate.abs().sum(1).max()))
    return norms


@torch.no_grad()
def combine_mirrored_gate_gradients(model):
    """Chain rule for the free forget-gate parameter when input gate = -forget."""
    h=model.hidden_size
    for layer in model.rnn_layers:
        for parameter in (layer.weight_hh_l0,layer.bias_ih_l0,layer.bias_hh_l0):
            if parameter.grad is not None:
                parameter.grad[h:2*h].sub_(parameter.grad[:h]);parameter.grad[:h].zero_()
        gradient=layer.weight_ih_l0.grad
        if gradient is not None:
            gradient[h:2*h,-model.control_count:].sub_(gradient[:h,-model.control_count:])
            gradient[:h,-model.control_count:].zero_()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=600)
    p.add_argument('--base',type=Path,default=Path('remix/runs/dfz-capacity-phase9/selected.pt'))
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--train-layers',type=int,default=2)
    p.add_argument('--calibration-interval',type=int,default=100)
    p.add_argument('--full-prefix',action='store_true',help='Replay the complete causal history before each gradient window')
    p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args()
    if args.output.exists():raise ValueError('output already exists')
    args.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(917)
    rng=np.random.default_rng(917);device=torch.device('mps')
    source=args.base
    model,payload=load_stable_effect(source);model=model.to(device)
    # Keep the early representation intact; train the last two causal layers.
    if not 1<=args.train_layers<=model.layers:raise ValueError('invalid number of trainable recurrent layers')
    for layer in model.rnn_layers[:model.layers-args.train_layers]:
        for parameter in layer.parameters():parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=1e-6)
    anchor={k:v.detach().clone() for k,v in model.named_parameters() if v.requires_grad}
    fit,cal=_partition(effect_files(Path('data/corpus/asrnn-physical-effects'),'dfz','train'))
    pairs=[read_effect_pair(path,'dfz') for path in fit];calrows=_rows(cal,'dfz')
    history=[];best=float('inf');best_state=None;best_metrics=None;started=time.perf_counter()
    for step in range(args.steps):
        xx,yy,cc=[],[],[];prefixes=[]
        for index in rng.integers(0,len(pairs),size=args.batch_size):
            x,y,c=pairs[index]
            active=np.flatnonzero(np.abs(x)>max(1e-4,float(np.abs(x).max())*.01));onset=int(active[0]) if len(active) else 2048
            if rng.random()<.65:
                peak=int(np.abs(y[1024:]).argmax())+1024
                start=max(0,min(len(x)-6144,peak-int(rng.integers(2048,6144))))
            else:start=int(rng.integers(max(0,onset-2048),max(onset-2048+1,len(x)-6144)))
            xx.append(x[start:start+6144]);yy.append(y[start:start+6144]);cc.append(c)
            if args.full_prefix:prefixes.append(x[:start+2048])
        dry=torch.from_numpy(np.stack(xx)).to(device);wet=torch.from_numpy(np.stack(yy)).to(device);controls=torch.from_numpy(np.stack(cc)).to(device)
        model.train()
        with torch.no_grad():
            if args.full_prefix:
                length=((max(map(len,prefixes))+2047)//2048)*2048
                history_audio=torch.from_numpy(np.stack([np.pad(x,(length-len(x),0)) for x in prefixes])).to(device)
                state=None
                for offset in range(0,length,2048):_,state=model(history_audio[:,offset:offset+2048],controls,state)
            else:_,state=model(dry[:,:2048],controls)
        prediction,_=model(dry[:,2048:],controls,state);target=wet[:,2048:]
        error=prediction-target
        waveform=error.square().mean(1)/target.square().mean(1).clamp_min(1e-6)
        dp=prediction[:,1:]-.95*prediction[:,:-1];dt=target[:,1:]-.95*target[:,:-1]
        emphasized=(dp-dt).square().mean(1)/dt.square().mean(1).clamp_min(1e-6)
        peak=(prediction.abs().amax(1)-target.abs().amax(1)).abs()
        envelope=(torch.nn.functional.max_pool1d(prediction.abs()[:,None,:],128,64)-torch.nn.functional.max_pool1d(target.abs()[:,None,:],128,64)).abs().mean((1,2))
        per_file=waveform+.25*emphasized+12*peak+2*envelope
        loss=.5*per_file.mean()+.5*per_file.topk(max(1,args.batch_size//4)).values.mean()+.01*sum((v-anchor[k]).square().mean() for k,v in model.named_parameters() if v.requires_grad)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if args.full_prefix:combine_mirrored_gate_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();norms=project(model,mirror_gates=args.full_prefix)
        if (step+1)%25==0:print(json.dumps({'step':step+1,'loss':float(loss.detach()),'elapsed_seconds':time.perf_counter()-started}),flush=True)
        if (step+1)%args.calibration_interval==0 or step+1==args.steps:
            report=_evaluate(model,calrows,device,8);row={'step':step+1,**report};history.append(row)
            key=report['absolute_peak_error_p95']+max(0,report['global_esr']-.03)
            if key<best:
                best=key;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_metrics=report;best_step=step+1
                candidate=dict(payload,state_dict=best_state)
                torch.save(candidate,args.output/'candidate.pt')
            progress={'schema':1,'admitted_for_official_eval':False,'history':history,'best_step':best_step,'best_mps_calibration':best_metrics,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'physical_audio_devices_used':False,'source_audio_modified':False,'official_eval_used_for_selection':False,'training_partition':{'fit':len(fit),'calibration':len(cal)},'candidate_recurrent_infinity_norms':norms}
            progress.update(full_causal_prefix_replayed=args.full_prefix,zero_input_gates_mirrored=args.full_prefix,batch_size=args.batch_size)
            (args.output/'training.json').write_text(json.dumps(progress,indent=2)+'\n')
            print(json.dumps({'calibration':row}),flush=True)
    model.load_state_dict(best_state);cpu=model.cpu().eval()
    verified=_evaluate(cpu,calrows,torch.device('cpu'),8)
    progress['calibration']=verified;progress['admitted_for_official_eval']=verified['passes_selection_gate']
    progress['checkpoint_sha256']=hashlib.sha256((args.output/'candidate.pt').read_bytes()).hexdigest()
    (args.output/'metrics.json').write_text(json.dumps(progress,indent=2)+'\n')
    print(json.dumps({'cpu_calibration':verified,'admitted':verified['passes_selection_gate']}),flush=True)


if __name__=='__main__':main()
