"""Label-free control search using the causal forward waveform model.

All input before the analysis endpoint is processed; the recurrent state is
never reset at a transient. The search grid is independent of filename labels.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from .asrnn_effects import read_effect_pair
from .stable_effect import load_stable_effect
from .train_effect_inverse_features import partitions,metrics


@torch.inference_mode()
def search_pair(model,dry,wet,control_count,step=.05,analysis_frames=16384):
    active=np.flatnonzero(np.abs(dry)>max(1.e-4,float(np.abs(dry).max())*.01))
    end=min(len(dry),int(active[0])+analysis_frames) if len(active) else len(dry)
    grid=np.array(list(itertools.product(np.arange(0,1+step*.1,step),repeat=control_count)),dtype=np.float32)
    target=torch.from_numpy(wet[:end])
    scores=[]
    for offset in range(0,len(grid),24):
        controls=torch.from_numpy(grid[offset:offset+24])
        clean=torch.from_numpy(dry[:end]).expand(len(controls),-1)
        state=None
        squared=torch.zeros(len(controls))
        absolute=torch.zeros(len(controls))
        for start in range(0,end,2048):
            rendered,state=model(clean[:,start:start+2048],controls,state)
            skip=max(0,1024-start)
            difference=rendered[:,skip:]-target[start+skip:start+rendered.shape[1]]
            squared+=difference.square().sum(1)
            absolute+=difference.abs().sum(1)
        scores.append(torch.stack((squared,absolute),1).numpy())
    values=np.concatenate(scores)
    return grid[values[:,0].argmin()],grid[values[:,1].argmin()],end


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('cs3','dfz'),required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--grid-step',type=float,default=.05)
    p.add_argument('--analysis-frames',type=int,default=16384)
    p.add_argument('--limit',type=int,default=0,help='debug calibration only; never admitted')
    args=p.parse_args()
    if args.output.exists(): raise ValueError('output already exists')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    model,payload=load_stable_effect(args.checkpoint)
    paths=partitions(Path('data/corpus/asrnn-physical-effects'),args.device)
    report={'schema':1,'device':args.device,'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'grid_step':args.grid_step,'analysis_frames':args.analysis_frames,'accepted':False,'closed_loop_evaluated':False,'physical_audio_devices_used':False,'source_audio_modified':False,'new_locked_final_audio_opened':False}
    selected=None
    for split in ('calibration','development'):
        if split=='development' and (args.limit or not report['calibration']['semantic_passed']): break
        chosen=paths[split][:args.limit] if args.limit else paths[split]
        predictions=[[],[]]; targets=[]; audible=[]; rows=[]
        for index,path in enumerate(chosen):
            dry,wet,truth=read_effect_pair(path,args.device)
            mse,mae,end=search_pair(model,dry,wet,len(truth),args.grid_step,args.analysis_frames)
            predictions[0].append(mse); predictions[1].append(mae)
            targets.append(truth); audible.append(np.abs(wet).max()>=.001)
            rows.append({'file':path.name,'mse_prediction':mse.tolist(),'mae_prediction':mae.tolist(),'truth':truth.tolist(),'analysis_end':end,'audible':bool(audible[-1])})
            if (index+1)%8==0: print(json.dumps({'split':split,'completed':index+1,'total':len(chosen)}),flush=True)
        target=np.stack(targets); mask=np.array(audible)
        candidates=[metrics(np.stack(v),target,mask,args.device) for v in predictions]
        if selected is None: selected=min(range(2),key=lambda i:(not candidates[i]['semantic_passed'],candidates[i]['macro_mae']+.25*candidates[i]['macro_p95']))
        report[split]=candidates[selected]
        report[split+'_candidates']=candidates
        report[split+'_predictions']=[dict(row,prediction=row['mse_prediction' if selected==0 else 'mae_prediction']) for row in rows]
        report['selected_objective']=('mse','mae')[selected]
        (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'split':split,'metrics':report[split],'selected_objective':report['selected_objective']}),flush=True)


if __name__=='__main__': main()
