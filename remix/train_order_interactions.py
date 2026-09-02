"""Fit nonlinear interaction heads using the unchanged order train split."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np,torch
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from torch.utils.data import DataLoader,Subset,ConcatDataset
from .data import DAFxOrderDataset,SyntheticControlDataset,dry_sources
from .evaluate_order_search import make_domains
from .train_order_transfer import transfer_batch
from .order_interaction_features import order_interaction_features
from .model import PairedEstimator
from .train import CORPUS,RUN


def interactions(batch):
    return np.stack([order_interaction_features(x.numpy(),y.numpy()) for x,y in zip(batch['dry'],batch['wet'])])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    source=Path('remix/runs/order-transfer-phase11');artifact=args.output/'interaction-heads.joblib'
    if artifact.exists():payload=joblib.load(artifact)
    else:
        arrays=np.load(source/'fit-features.npz');old,Y,M=(arrays[k] for k in ('X','Y','M'))
        real=DAFxOrderDataset(CORPUS,'train');real=Subset(real,[i for i,r in enumerate(real.records) if len(r.order)>=2])
        synth=SyntheticControlDataset(dry_sources(CORPUS,'train'),3072,seed=20260921,order_equivalence_db=-30.)
        dataset=ConcatDataset([real,synth]);extra=[];truth=[]
        for index,batch in enumerate(DataLoader(dataset,batch_size=8)):
            keep=batch['order_mask'].any(1)
            if keep.any():
                batch={k:v[keep] for k,v in batch.items()};extra.append(interactions(batch));truth.append(batch['order'].numpy())
            if (index+1)%64==0:print(json.dumps({'stage':'interaction-fit-features','batch':index+1,'batches':len(dataset)//8}),flush=True)
        if not np.array_equal(np.concatenate(truth),Y):raise ValueError('fit cache order differs')
        X=np.concatenate((old,np.concatenate(extra)),1);heads={}
        for name in ('interaction-hist','interaction-extra'):
            heads[name]=[]
            for i in range(3):
                use=M[:,i]>.5
                model=(HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=15,min_samples_leaf=12,l2_regularization=2,early_stopping=False,random_state=920+i) if name.endswith('hist') else ExtraTreesClassifier(n_estimators=320,max_features=.7,min_samples_leaf=2,n_jobs=2,random_state=920+i))
                model.fit(X[use],Y[use,i].astype(int));heads[name].append(model)
                print(json.dumps({'stage':'interaction-fit','candidate':name,'relation':i}),flush=True)
        payload={'candidates':heads,'feature_dimensions':X.shape[1],'fit_cache_sha256':hashlib.sha256((source/'fit-features.npz').read_bytes()).hexdigest()}
        joblib.dump(payload,artifact,compress=3)
    encoder=PairedEstimator();encoder.load_state_dict(torch.load(RUN/'paired-estimator.pt',map_location='cpu',weights_only=True));encoder.eval()
    for split in ('calibrate','valid'):
        for domain,dataset in make_domains(CORPUS,split,320,-30.).items():
            target=args.output/f'blend-{split}-{domain}.json'
            if target.exists():continue
            original=source/f'blend-{split}-{domain}.json'
            if not original.exists():raise ValueError(f'baseline blend audit not finished: {original}')
            rows=json.loads(original.read_text());features=[]
            for batch in DataLoader(dataset,batch_size=8):features.append(np.concatenate((transfer_batch(encoder,batch),interactions(batch)),1))
            X=np.concatenate(features)
            if X.shape[1]!=payload['feature_dimensions'] or len(X)!=len(rows):raise ValueError('audit feature geometry changed')
            for name,heads in payload['candidates'].items():
                probs=np.stack([h.predict_proba(X)[:,1] for h in heads],1)
                for row,values in zip(rows,probs):row['probabilities'][name]=values.tolist()
            target.write_text(json.dumps(rows)+'\n')
            print(json.dumps({'stage':'interaction-audit','split':split,'domain':domain}),flush=True)
    (args.output/'training.json').write_text(json.dumps({'model_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'training_guitars':['les','prs'],'synthetic_seed':20260921,'pedalboard_training':False,'new_locked_final_audio_opened':False,'physical_audio_devices_used':False,'source_audio_modified':False},indent=2)+'\n')


if __name__=='__main__':main()
