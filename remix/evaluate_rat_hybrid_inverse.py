"""Calibrate a fixed blend of supervised and forward-refined RAT controls."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .drive_model import DriveControlEstimator
from .refine_asrnn_rat_inverse import refine
from .train_asrnn_rat_inverse import feature_matrix, predict, closed_loop
from .train_effect_inverse_features import partitions, metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('output already exists')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    inverse=Path('remix/runs/asrnn-rat-inverse-pilot/rat-inverse.pt')
    renderer=Path('remix/runs/asrnn-rat-stable-pilot/rat-stable.pt')
    semantic=Path('remix/runs/rat-semantic-phase10/metrics.json')
    source=json.loads(semantic.read_text())
    model=DriveControlEstimator()
    model.load_state_dict(torch.load(inverse,map_location='cpu',weights_only=True)['state_dict'])
    model.eval()
    paths=partitions(Path('data/corpus/asrnn-physical-effects'),'rat')
    results={}
    choices=None
    for split in ('calibration','development'):
        features=feature_matrix(paths[split])
        initial=predict(model,features,torch.device('cpu'))
        refined=refine(paths[split],initial,renderer,steps=60,learning_rate=.05,frames=4096,batch_size=16).numpy()
        rows=source[split+'_predictions']
        if [r['file'] for r in rows]!=[x.name for x in paths[split]]: raise ValueError('partition order mismatch')
        supervised=np.array([r['prediction'] for r in rows])
        target=np.array([r['truth'] for r in rows])
        audible=np.array([r['audible'] for r in rows])
        # Weights are global per knob, never per-example truth-dependent.
        candidates={str(w):(1-w)*refined+w*supervised for w in (0,.25,.5,.75,1)}
        reports={k:metrics(v,target,audible,'rat') for k,v in candidates.items()}
        if choices is None:
            names=('distortion','tone','volume')
            choices=[min(reports,key=lambda k:(not reports[k]['per_control'][n]['passed'],reports[k]['per_control'][n]['mae']+.25*reports[k]['per_control'][n]['p95'])) for n in names]
        estimate=np.stack([candidates[c][:,i] for i,c in enumerate(choices)],1)
        results[split]=metrics(estimate,target,audible,'rat')
        if split=='calibration': results['calibration_candidates']=reports
        results[split+'_predictions']=[{'file':path.name,'prediction':e.tolist(),'truth':t.tolist(),'audible':bool(a)} for path,e,t,a in zip(paths[split],estimate,target,audible)]
        print(json.dumps({'split':split,'choices':choices,'metrics':results[split]}),flush=True)
        if split=='development':
            results['closed_loop']=closed_loop(paths[split],torch.tensor(estimate,dtype=torch.float32),renderer,torch.device('cpu'))
    closure=results['closed_loop']
    results.update(schema=1,device='rat',choices=choices,accepted=bool(results['calibration']['semantic_passed'] and results['development']['semantic_passed'] and closure['recovered_vs_bypass_esr_improvement']>=.9 and closure['recovered_vs_bypass_mae_improvement']>=.8),physical_audio_devices_used=False,new_locked_final_audio_opened=False,source_audio_modified=False,source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (inverse,renderer,semantic)})
    (args.output/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'accepted':results['accepted'],'closed_loop':closure}),flush=True)


if __name__=='__main__': main()
