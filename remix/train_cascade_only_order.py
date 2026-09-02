"""Physics-only nonlinear placement ablation using existing feature caches."""
from __future__ import annotations
import argparse,hashlib,itertools,json
from pathlib import Path
import joblib,numpy as np,torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('train','calibrate','valid','all'),default='all')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    source=Path('remix/runs/order-cascade-refined-phase11');base=Path('remix/runs/order-temporal-phase11')
    fit=Path('remix/runs/order-cascade-phase11/cascade-fit.npz')
    artifact=args.output/'physics-heads.joblib'
    if artifact.exists():heads=joblib.load(artifact)
    else:
        X=np.load(fit)['X'];arrays=np.load('remix/runs/order-transfer-phase11/fit-features.npz');Y,M=arrays['Y'],arrays['M']
        if X.shape!=(len(Y),416):raise ValueError('physics-only training geometry differs')
        heads={}
        for kind in ('hist','rbf'):
            heads[kind]=[]
            for relation in (0,1):
                use=M[:,relation]>.5
                model=(HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=15,min_samples_leaf=12,l2_regularization=2,early_stopping=False,random_state=963+relation) if kind=='hist' else
                       make_pipeline(StandardScaler(),SVC(C=10,probability=True,cache_size=256,random_state=963+relation)))
                model.fit(X[use],Y[use,relation].astype(int));heads[kind].append(model)
                print(json.dumps({'stage':'physics-only-fit','kind':kind,'relation':relation}),flush=True)
        joblib.dump(heads,artifact,compress=3)
    note=args.output/'training.json';previous=json.loads(note.read_text()) if note.exists() else {}
    report={'model_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'physics_features_sha256':hashlib.sha256(fit.read_bytes()).hexdigest(),'feature_dimensions':416,'training_guitars':['les','prs'],'synthetic_seed':20260921,'source_run':str(source),'retained_relation':2,'new_locked_final_audio_opened':False,'physical_audio_devices_used':False,'source_audio_modified':False,'new_audio_read_for_this_ablation':False,'completed_splits':previous.get('completed_splits',[])}
    note.write_text(json.dumps(report,indent=2)+'\n')
    if args.stage=='train':return
    selection=base/'blend-calibration.json';frozen=json.loads(selection.read_text())
    base_name,weight,margin=frozen['selected']
    if weight!=1 or margin!=0:raise ValueError('unexpected temporal base')
    splits=('calibrate','valid') if args.stage=='all' else (args.stage,)
    for split in splits:
        for domain in ('real','reference','alternate','stress','pedalboard'):
            target=args.output/f'blend-{split}-{domain}.json'
            if target.exists():continue
            X=np.load(source/f'analysis-{split}-{domain}.npz')['X'][:,-416:]
            rows=json.loads((source/target.name).read_text());oldrows=json.loads((base/target.name).read_text())
            if len(X)!=len(rows) or len(rows)!=len(oldrows):raise ValueError('audit geometry differs')
            for kind,models in heads.items():
                probs=np.stack([model.predict_proba(X)[:,1] for model in models],1)
                for row,old,values in zip(rows,oldrows,probs):
                    if any(row[k]!=old[k] for k in ('active','baseline','baseline_error','truth','mask')):raise ValueError('audit alignment differs')
                    original=np.asarray(old['probabilities'][base_name])
                    for first,second in itertools.product((0.,.5,1.),repeat=2):
                        if first==second==0:continue
                        value=original.copy();value[0]=(1-first)*value[0]+first*values[0];value[1]=(1-second)*value[1]+second*values[1]
                        row['probabilities'][f'cascade-drive-physics-{kind}-{first}-{second}']=value.tolist()
            target.write_text(json.dumps(rows)+'\n')
            print(json.dumps({'stage':'physics-only-audit','split':split,'domain':domain}),flush=True)
        report['completed_splits']=sorted(set(report['completed_splits'])|{split});report['base_calibration_sha256']=hashlib.sha256(selection.read_bytes()).hexdigest();note.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
