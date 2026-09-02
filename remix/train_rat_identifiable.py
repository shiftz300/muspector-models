"""Fit RAT Tone on audible training pairs; keep quiet cases in reported metrics."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from .train_effect_inverse_features import metrics,partitions
from .train_asrnn_rat_inverse import closed_loop


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('output already exists')
    args.output.mkdir(parents=True);torch.set_num_threads(2)
    source=Path('remix/runs/rat-semantic-phase10')
    arrays=np.load(source/'features.npz')
    hybrid=json.loads(Path('remix/runs/rat-hybrid-phase10/metrics.json').read_text())
    previous=json.loads((source/'metrics.json').read_text())
    def estimates(split):
        rows=hybrid[split+'_predictions'];other=previous[split+'_predictions']
        base=np.array([r['prediction'] for r in rows]);new=np.array([r['prediction'] for r in other])
        w=np.array(list(map(float,hybrid['choices'])))
        refined=(base-w*new)/(1-w)
        return base,refined
    base,refined=estimates('calibration')
    mask=arrays['fit_audible'];X=arrays['fit_X'][mask];Y=arrays['fit_Y'][mask,1]
    models={
        'extra-trees':ExtraTreesRegressor(n_estimators=384,max_features=.8,min_samples_leaf=1,n_jobs=2,random_state=914),
        'hist-15':HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=1,early_stopping=False,random_state=914),
        'hist-7':HistGradientBoostingRegressor(max_iter=300,max_leaf_nodes=7,min_samples_leaf=12,l2_regularization=2,early_stopping=False,random_state=914),
    }
    candidates={};predictions={}
    for name,model in models.items():
        print(json.dumps({'fitting':name,'audible_training_examples':int(mask.sum()),'quiet_training_excluded_for_tone_only':int((~mask).sum())}),flush=True)
        model.fit(X,Y)
        predicted=model.predict(arrays['calibration_X']).clip(0,1)
        for weight in (0.25,.5,.75,1.):
            value=base.copy();value[:,1]=(1-weight)*refined[:,1]+weight*predicted
            key=f'{name}:{weight}';predictions[key]=value
            candidates[key]=metrics(value,arrays['calibration_Y'],arrays['calibration_audible'],'rat')
    selected=min(candidates,key=lambda k:(not candidates[k]['semantic_passed'],candidates[k]['per_control']['tone']['mae']+.25*candidates[k]['per_control']['tone']['p95']))
    name,weight=selected.split(':');weight=float(weight)
    report={'schema':1,'device':'rat','selected':selected,'calibration_candidates':candidates,'calibration':candidates[selected],'accepted':False,'physical_audio_devices_used':False,'source_audio_modified':False,'new_locked_final_audio_opened':False,'calibration_predictions':[dict(r,prediction=e.tolist()) for r,e in zip(hybrid['calibration_predictions'],predictions[selected])]}
    base,refined=estimates('development')
    base[:,1]=(1-weight)*refined[:,1]+weight*models[name].predict(arrays['development_X']).clip(0,1)
    report['development']=metrics(base,arrays['development_Y'],arrays['development_audible'],'rat')
    report['development_predictions']=[dict(r,prediction=e.tolist()) for r,e in zip(hybrid['development_predictions'],base)]
    print(json.dumps({'selected':selected,'calibration':report['calibration'],'development':report['development']}),flush=True)
    model_path=args.output/'tone.joblib';joblib.dump({'model':models[name],'weight':weight,'feature_version':'paired-transfer-v1','upstream_quiet_threshold':.001},model_path,compress=3)
    report['model_sha256']=hashlib.sha256(model_path.read_bytes()).hexdigest()
    if report['calibration']['semantic_passed'] and report['development']['semantic_passed']:
        paths=partitions(Path('data/corpus/asrnn-physical-effects'),'rat')['development']
        report['closed_loop']=closed_loop(paths,torch.tensor(base,dtype=torch.float32),Path('remix/runs/asrnn-rat-stable-pilot/rat-stable.pt'),torch.device('cpu'))
        r=report['closed_loop'];report['accepted']=r['recovered_vs_bypass_esr_improvement']>=.9 and r['recovered_vs_bypass_mae_improvement']>=.8
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'accepted':report['accepted'],'closed_loop':report.get('closed_loop')}),flush=True)


if __name__=='__main__': main()
