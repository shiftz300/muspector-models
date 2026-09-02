"""Batched float32 GPU replay of the frozen CS-3 causal search, no audio I/O."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .asrnn_effects import read_effect_pair
from .stable_effect import load_stable_effect
from .train_effect_inverse_features import partitions,metrics


@torch.inference_mode()
def search_batch(model,pairs,frames=16384):
    ends=[]
    for dry,wet in pairs:
        active=np.flatnonzero(np.abs(dry)>max(.0001,float(np.abs(dry).max())*.01))
        ends.append(min(len(dry),int(active[0])+frames) if len(active) else len(dry))
    n=((max(ends)+2047)//2048)*2048;device=next(model.parameters()).device
    dry=np.stack([np.pad(x[:e],(0,n-e)) for (x,y),e in zip(pairs,ends)])
    wet=np.stack([np.pad(y[:e],(0,n-e)) for (x,y),e in zip(pairs,ends)])
    grid=torch.arange(21,device=device,dtype=torch.float32)/20
    x=torch.from_numpy(dry).to(device)[:,None,:].expand(-1,21,-1).reshape(-1,n)
    y=torch.from_numpy(wet).to(device)[:,None,:].expand(-1,21,-1).reshape(-1,n)
    controls=grid.repeat(len(pairs)).unsqueeze(1)
    endings=torch.tensor(ends,device=device).repeat_interleave(21)
    squares=torch.zeros(len(x),device=device);absolute=torch.zeros_like(squares);state=None
    for start in range(0,n,2048):
        output,state=model(x[:,start:start+2048],controls,state)
        positions=torch.arange(start,start+output.shape[1],device=device)
        mask=(positions[None,:]<endings[:,None]) & (positions[None,:]>=1024)
        residual=(output-y[:,start:start+2048])*mask
        squares+=residual.square().sum(1);absolute+=residual.abs().sum(1)
    mse=grid[squares.reshape(-1,21).argmin(1)].cpu().numpy()[:,None]
    mae=grid[absolute.reshape(-1,21).argmin(1)].cpu().numpy()[:,None]
    return mse,mae,ends


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=4)
    args=p.parse_args()
    if args.output.exists():raise ValueError('output already exists')
    args.output.mkdir(parents=True);torch.set_num_threads(2)
    checkpoint=Path('remix/runs/asrnn-cs3-ensemble-pilot/cs3-stable.pt')
    cpu_report=Path('remix/runs/cs3-search-phase10/metrics.json')
    previous=json.loads(cpu_report.read_text())
    if previous['selected_objective']!='mse' or not previous['calibration']['semantic_passed']:raise ValueError('CPU search calibration is not frozen/passed')
    model,_=load_stable_effect(checkpoint);model=model.to('mps')
    paths=partitions(Path('data/corpus/asrnn-physical-effects'),'cs3')
    report={'schema':1,'device':'cs3','compute':'mps-float32','checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'selected_objective':'mse','accepted':False,'closed_loop_evaluated':False,'grid_step':.05,'analysis_frames':16384,'physical_audio_devices_used':False,'source_audio_modified':False,'new_locked_final_audio_opened':False}
    for split in ('calibration','development'):
        rows=[];started=time.perf_counter()
        for offset in range(0,len(paths[split]),args.batch_size):
            chosen=paths[split][offset:offset+args.batch_size]
            values=[read_effect_pair(path,'cs3') for path in chosen]
            mse,mae,ends=search_batch(model,[(x,y) for x,y,c in values])
            rows.extend({'file':path.name,'prediction':prediction.tolist(),'truth':c.tolist(),'analysis_end':end,'audible':bool(np.abs(y).max()>=.001)} for path,prediction,(x,y,c),end in zip(chosen,mse,values,ends))
            report[split+'_predictions']=rows
            (args.output/'progress.json').write_text(json.dumps({'split':split,'completed':len(rows),'total':len(paths[split]),'elapsed_seconds':time.perf_counter()-started})+'\n')
            print(json.dumps({'split':split,'completed':len(rows),'total':len(paths[split]),'elapsed_seconds':time.perf_counter()-started}),flush=True)
        prediction=np.array([r['prediction'] for r in rows]);target=np.array([r['truth'] for r in rows]);audible=np.array([r['audible'] for r in rows])
        report[split]=metrics(prediction,target,audible,'cs3')
        if split=='calibration':
            expected=np.array([r['prediction'] for r in previous['calibration_predictions']])
            difference=float(np.abs(prediction-expected).max());report['cpu_calibration_max_control_difference']=difference
            if difference>1e-6:raise ValueError(f'GPU search changed CPU calibration decisions: {difference}')
        (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'split':split,'metrics':report[split]}),flush=True)


if __name__=='__main__':main()
