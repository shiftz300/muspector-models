"""Calibration-selected convex consensus of retained semantic inverse heads."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib
import numpy as np
from .train_effect_inverse_features import metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('output already exists')
    args.output.mkdir(parents=True)
    payload=joblib.load(args.source/'inverse.joblib')
    evidence=json.loads((args.source/'metrics.json').read_text())
    arrays=np.load(args.source/'features.npz')
    names=sorted(payload['models'])
    if len(names)!=2: raise ValueError('consensus currently needs exactly two retained models')
    selected=None;report={'schema':1,'device':payload['device'],'accepted':False,'closed_loop_evaluated':False,'physical_audio_devices_used':False,'source_audio_modified':False,'new_locked_final_audio_opened':False,'source_model_sha256':hashlib.sha256((args.source/'inverse.joblib').read_bytes()).hexdigest()}
    for split in ('calibration','development'):
        X,Y,mask=(arrays[f'{split}_{k}'] for k in ('X','Y','audible'))
        predictions=[np.asarray(payload['models'][n].predict(X)).reshape(len(X),-1).clip(0,1) for n in names]
        candidates={str(w):predictions[0]*(1-w)+predictions[1]*w for w in (0,.125,.25,.375,.5,.625,.75,.875,1)}
        scores={k:metrics(v,Y,mask,payload['device']) for k,v in candidates.items()}
        if selected is None:
            controls=list(scores['0']['per_control'])
            selected=[min(scores,key=lambda k:(not scores[k]['per_control'][n]['passed'],scores[k]['per_control'][n]['mae']+.25*scores[k]['per_control'][n]['p95'])) for n in controls]
            report['calibration_candidates']=scores
        prediction=np.stack([candidates[w][:,i] for i,w in enumerate(selected)],1)
        report[split]=metrics(prediction,Y,mask,payload['device'])
        report[split+'_predictions']=[{'file':name,'truth':truth.tolist(),'prediction':value.tolist(),'audible':bool(a)} for name,truth,value,a in zip(evidence['partitions'][split],Y,prediction,mask)]
        print(json.dumps({'split':split,'choices':selected,'metrics':report[split]}),flush=True)
    report['model_names']=names;report['weights_for_second_model']=selected
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
