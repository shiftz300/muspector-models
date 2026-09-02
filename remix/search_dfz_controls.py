"""Continuous coarse-to-fine DFZ control recovery via causal waveform search."""
from __future__ import annotations
import argparse,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import torch
from .asrnn_effects import read_effect_pair
from .stable_effect import load_stable_effect
from .train_effect_inverse_features import partitions,metrics


@torch.inference_mode()
def score_proposals(model,pairs,proposals):
    ends=[]
    for dry,wet in pairs:
        active=np.flatnonzero(np.abs(dry)>max(.0001,float(np.abs(dry).max())*.01))
        ends.append(min(len(dry),int(active[0])+16384) if len(active) else len(dry))
    # Keep MPS LSTM graph shapes fixed: variable tail lengths otherwise create
    # one compiled graph per clip. Padding is causal and never enters the loss.
    n=((max(ends)+2047)//2048)*2048;device=next(model.parameters()).device;g=proposals.shape[1]
    dry=np.stack([np.pad(x[:e],(0,n-e)) for (x,y),e in zip(pairs,ends)])
    wet=np.stack([np.pad(y[:e],(0,n-e)) for (x,y),e in zip(pairs,ends)])
    x=torch.from_numpy(dry).to(device)[:,None,:].expand(-1,g,-1).reshape(-1,n)
    y=torch.from_numpy(wet).to(device)[:,None,:].expand(-1,g,-1).reshape(-1,n)
    controls=torch.from_numpy(proposals.reshape(-1,2)).to(device)
    endings=torch.tensor(ends,device=device).repeat_interleave(g)
    squares=torch.zeros(len(x),device=device);state=None
    for start in range(0,n,2048):
        output,state=model(x[:,start:start+2048],controls,state)
        positions=torch.arange(start,start+output.shape[1],device=device)
        mask=(positions[None,:]<endings[:,None]) & (positions[None,:]>=1024)
        residual=(output-y[:,start:start+2048])*mask
        squares+=residual.square().sum(1)
    return squares.reshape(-1,g).cpu().numpy()


def search(model,pairs):
    grid=np.array(list(itertools.product(np.linspace(0,1,5),repeat=2)),dtype=np.float32)
    proposals=np.broadcast_to(grid,(len(pairs),*grid.shape)).copy()
    values=score_proposals(model,pairs,proposals)
    best=proposals[np.arange(len(pairs)),values.argmin(1)]
    for step in (.125,.0625,.03125):
        offsets=np.array(list(itertools.product((-step,0.,step),repeat=2)),dtype=np.float32)
        proposals=(best[:,None,:]+offsets[None,:,:]).clip(0,1)
        values=score_proposals(model,pairs,proposals)
        best=proposals[np.arange(len(pairs)),values.argmin(1)]
    return best


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.output.exists() and not args.resume:raise ValueError('output already exists')
    args.output.mkdir(parents=True,exist_ok=args.resume);torch.set_num_threads(2)
    checkpoint=Path('remix/runs/dfz-capacity-phase9/selected.pt')
    model,_=load_stable_effect(checkpoint);model=model.to('mps')
    paths=partitions(Path('data/corpus/asrnn-physical-effects'),'dfz')
    report={'schema':1,'device':'dfz','compute':'mps-float32','checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'selected_objective':'mse','accepted':False,'closed_loop_evaluated':False,'search_steps':[.25,.125,.0625,.03125],'analysis_frames':16384,'forward_teacher_peak_gate_passed':False,'physical_audio_devices_used':False,'source_audio_modified':False,'new_locked_final_audio_opened':False}
    if args.resume:
        previous=json.loads((args.output/'metrics.json').read_text())
        if previous['checkpoint_sha256']!=report['checkpoint_sha256']:raise ValueError('resume teacher changed')
        report=previous
    for split in ('calibration','development'):
        if split in report:continue
        progress=args.output/f'{split}-progress.json'
        rows=json.loads(progress.read_text()) if args.resume and progress.exists() else []
        if [r['file'] for r in rows]!=[p.name for p in paths[split][:len(rows)]]:raise ValueError('resume row order changed')
        started=time.perf_counter()
        for offset in range(len(rows),len(paths[split]),4):
            chosen=paths[split][offset:offset+4];values=[read_effect_pair(path,'dfz') for path in chosen]
            estimate=search(model,[(x,y) for x,y,c in values])
            rows.extend({'file':path.name,'prediction':prediction.tolist(),'truth':c.tolist(),'audible':bool(np.abs(y).max()>=.001)} for path,prediction,(x,y,c) in zip(chosen,estimate,values))
            progress.write_text(json.dumps(rows)+'\n')
            torch.mps.empty_cache()
            print(json.dumps({'split':split,'completed':len(rows),'total':len(paths[split]),'elapsed_seconds':time.perf_counter()-started}),flush=True)
        prediction=np.array([r['prediction'] for r in rows]);target=np.array([r['truth'] for r in rows]);audible=np.array([r['audible'] for r in rows])
        report[split]=metrics(prediction,target,audible,'dfz');report[split+'_predictions']=rows
        (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'split':split,'metrics':report[split]}),flush=True)
        if not report[split]['semantic_passed']:break


if __name__=='__main__':main()
