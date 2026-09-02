"""A small train-only regularization grid for nonlinear-placement heads.

Audio-derived features are cached once for this grid and may be deleted after
selection. C=10 is replayed against the prior audit as a numerical contract;
no development labels participate in training or hyperparameter selection.
"""
from __future__ import annotations
import argparse,hashlib,itertools,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import joblib,numpy as np,torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from .evaluate_order_search import make_domains
from .model import PairedEstimator
from .order_cascade_features import cascade_order_features
from .order_invariant_features import invariant_order_features
from .train_order_transfer import transfer_batch
from .train import CORPUS,RUN


def pair_features(pair):
    return cascade_order_features(*pair)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    original=Path('remix/runs/order-cascade-phase11');base=Path('remix/runs/order-temporal-phase11')
    artifact=args.output/'refined-heads.joblib';source_heads=joblib.load(original/'cascade-heads.joblib')
    if artifact.exists():heads=joblib.load(artifact)
    else:
        arrays=np.load(Path('remix/runs/order-transfer-phase11')/'fit-features.npz')
        X=np.concatenate((invariant_order_features(arrays['X'],True),np.load(original/'cascade-fit.npz')['X']),1)
        Y,M=arrays['Y'],arrays['M'];heads={'rbf-10.0':source_heads['cascade-rbf']}
        for C in (1.,100.):
            key=f'rbf-{C}';heads[key]=[]
            for relation in range(3):
                use=M[:,relation]>.5
                model=make_pipeline(StandardScaler(),SVC(C=C,probability=True,cache_size=256,random_state=950+relation))
                model.fit(X[use],Y[use,relation].astype(int));heads[key].append(model)
                print(json.dumps({'stage':'cascade-regularization-fit','C':C,'relation':relation}),flush=True)
        joblib.dump(heads,artifact,compress=3)
    base_selection=base/'blend-calibration.json';frozen=json.loads(base_selection.read_text())
    base_name,weight,margin=frozen['selected']
    if weight!=1 or margin!=0:raise ValueError('unexpected temporal base selection')
    encoder=PairedEstimator();encoder.load_state_dict(torch.load(RUN/'paired-estimator.pt',map_location='cpu',weights_only=True));encoder.eval()
    progress=args.output/'replay-progress.json'
    parity=json.loads(progress.read_text()) if progress.exists() else {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for split in ('calibrate','valid'):
            for domain,dataset in make_domains(CORPUS,split,320,-30.).items():
                target=args.output/f'blend-{split}-{domain}.json'
                if target.exists() and f'{split}-{domain}' in parity:continue
                cache=args.output/f'analysis-{split}-{domain}.npz'
                if cache.exists():X=np.load(cache)['X']
                else:
                    features=[]
                    for batch in DataLoader(dataset,batch_size=8):
                        old=invariant_order_features(transfer_batch(encoder,batch),True)
                        pairs=[(x.numpy(),y.numpy()) for x,y in zip(batch['dry'],batch['wet'])]
                        extra=np.stack(list(pool.map(pair_features,pairs)))
                        features.append(np.concatenate((old,extra),1))
                    X=np.concatenate(features);np.savez_compressed(cache,X=X)
                rows=json.loads((original/target.name).read_text());oldrows=json.loads((base/target.name).read_text())
                if len(rows)!=len(X) or len(rows)!=len(oldrows):raise ValueError('audit geometry differs')
                for row,old in zip(rows,oldrows):
                    if any(row[k]!=old[k] for k in ('active','baseline','baseline_error','truth','mask')):raise ValueError('audit row alignment differs')
                variants={key:np.stack([h.predict_proba(X)[:,1] for h in models],1) for key,models in heads.items()}
                difference=float(np.max(abs(variants['rbf-10.0']-np.asarray([r['probabilities']['cascade-rbf'] for r in rows]))))
                if difference>1e-9:raise ValueError(f'C=10 feature replay differs: {difference}')
                parity[f'{split}-{domain}']=difference
                variants.update({kind:np.asarray([r['probabilities'][f'cascade-{kind}'] for r in rows]) for kind in ('hist','trees')})
                for key,probabilities in variants.items():
                    for row,old,values in zip(rows,oldrows,probabilities):
                        initial=np.asarray(old['probabilities'][base_name])
                        for first,second in itertools.product((0.,.5,1.),repeat=2):
                            if first==second==0:continue
                            combined=initial.copy()
                            combined[0]=(1-first)*initial[0]+first*values[0]
                            combined[1]=(1-second)*initial[1]+second*values[1]
                            row['probabilities'][f'cascade-drive-{key}-{first}-{second}']=combined.tolist()
                target.write_text(json.dumps(rows)+'\n')
                progress.write_text(json.dumps(parity,indent=2)+'\n')
                print(json.dumps({'stage':'cascade-grid-audit','split':split,'domain':domain,'prior_probability_replay_max':difference}),flush=True)
    report={'model_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'base_calibration_sha256':hashlib.sha256(base_selection.read_bytes()).hexdigest(),'prior_probability_replay':parity,'fit_guitars':['les','prs'],'synthetic_seed':20260921,'C':[1.,10.,100.],'retained_relation':2,'updated_relations':[0,1],'audio_feature_workers':2,'new_locked_final_audio_opened':False,'source_audio_modified':False,'physical_audio_devices_used':False,'pedalboard_training':False}
    (args.output/'training.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
