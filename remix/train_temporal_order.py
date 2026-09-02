"""Fit nonstationary-transfer order heads on the existing train guitars only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Subset, ConcatDataset

from .data import DAFxOrderDataset, SyntheticControlDataset, dry_sources
from .evaluate_order_search import make_domains
from .model import PairedEstimator
from .order_invariant_features import invariant_order_features
from .order_temporal_features import temporal_order_features
from .order_cascade_features import cascade_order_features
from .train_order_transfer import transfer_batch
from .train import CORPUS, RUN


def extra_batch(batch,feature_kind='temporal'):
    function=temporal_order_features if feature_kind=='temporal' else cascade_order_features
    return np.stack([function(x.numpy(),y.numpy())
                     for x,y in zip(batch['dry'],batch['wet'])])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--feature-kind',choices=('temporal','cascade'),default='temporal')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    original=Path('remix/runs/order-transfer-phase11')
    arrays=np.load(original/'fit-features.npz');old,Y,M=(arrays[k] for k in ('X','Y','M'))
    cache=args.output/f'{args.feature_kind}-fit.npz';artifact=args.output/f'{args.feature_kind}-heads.joblib'
    if artifact.exists():heads=joblib.load(artifact)
    else:
        if cache.exists():extras=np.load(cache)['X']
        else:
            real=DAFxOrderDataset(CORPUS,'train')
            real=Subset(real,[i for i,r in enumerate(real.records) if len(r.order)>=2])
            synth=SyntheticControlDataset(dry_sources(CORPUS,'train'),3072,seed=20260921,order_equivalence_db=-30.)
            xx,yy=[],[]
            for index,batch in enumerate(DataLoader(ConcatDataset((real,synth)),batch_size=8)):
                keep=batch['order_mask'].any(1)
                if keep.any():
                    batch={k:v[keep] for k,v in batch.items()};xx.append(extra_batch(batch,args.feature_kind));yy.append(batch['order'].numpy())
                if (index+1)%32==0: print(json.dumps({'stage':f'{args.feature_kind}-train-features','batches':index+1}),flush=True)
            if not np.array_equal(np.concatenate(yy),Y):raise ValueError('training alignment differs')
            extras=np.concatenate(xx);np.savez_compressed(cache,X=extras)
        X=np.concatenate((invariant_order_features(old,True),extras),1)
        heads={}
        for kind in ('hist','trees','rbf'):
            name=f'{args.feature_kind}-{kind}';heads[name]=[]
            for relation in range(3):
                use=M[:,relation]>.5
                model=(HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=15,min_samples_leaf=12,l2_regularization=2,early_stopping=False,random_state=950+relation) if kind=='hist' else
                       ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features=.7,n_jobs=2,random_state=950+relation) if kind=='trees' else
                       make_pipeline(StandardScaler(),SVC(C=10,probability=True,cache_size=256,random_state=950+relation)))
                model.fit(X[use],Y[use,relation].astype(int));heads[name].append(model)
                print(json.dumps({'stage':'temporal-fit','candidate':name,'relation':relation}),flush=True)
        joblib.dump(heads,artifact,compress=3)
    encoder=PairedEstimator();encoder.load_state_dict(torch.load(RUN/'paired-estimator.pt',map_location='cpu',weights_only=True));encoder.eval()
    for split in ('calibrate','valid'):
        for domain,dataset in make_domains(CORPUS,split,320,-30.).items():
            target=args.output/f'blend-{split}-{domain}.json'
            if target.exists():continue
            rows=json.loads((Path('remix/runs/order-invariant-phase11')/target.name).read_text())
            X=np.concatenate([np.concatenate((invariant_order_features(transfer_batch(encoder,batch),True),extra_batch(batch,args.feature_kind)),1) for batch in DataLoader(dataset,batch_size=8)])
            if len(X)!=len(rows):raise ValueError('audit row count differs')
            for name,models in heads.items():
                probs=np.stack([model.predict_proba(X)[:,1] for model in models],1)
                for row,values in zip(rows,probs):
                    row['probabilities'][name]=values.tolist()
                    for base in ('hist-boost','interaction-hist','rbf-1.0+hist-boost'):
                        original_prob=np.asarray(row['probabilities'][base])
                        row['probabilities'][name+'+'+base]=(.5*values+.5*original_prob).tolist()
                        for weight in (.5,1.):
                            merged=original_prob.copy();merged[2]=(1-weight)*merged[2]+weight*values[2]
                            row['probabilities'][f'{name}-relation2-{weight}+{base}']=merged.tolist()
            target.write_text(json.dumps(rows)+'\n');print(json.dumps({'stage':'temporal-audit','split':split,'domain':domain}),flush=True)
    (args.output/'training.json').write_text(json.dumps({'model_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'feature_kind':args.feature_kind,'fit_cache_sha256':hashlib.sha256((original/'fit-features.npz').read_bytes()).hexdigest(),'fit_guitars':['les','prs'],'synthetic_seed':20260921,'pedalboard_training':False,'new_locked_final_audio_opened':False,'source_audio_modified':False,'physical_audio_devices_used':False,'identifiability':'D/R ideal LTI commute; mask unchanged, features only seek nonstationary departures'},indent=2)+'\n')


if __name__=='__main__':main()
