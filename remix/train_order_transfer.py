"""Train masked order heads on frozen audio features; no Pedalboard training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from torch.utils.data import DataLoader,Subset,ConcatDataset
from .data import DAFxOrderDataset,SyntheticControlDataset,dry_sources
from .model import PairedEstimator
from .paired_transfer_features import paired_transfer_features
from .train import CORPUS,RUN


@torch.inference_mode()
def transfer_batch(model,batch):
    dry,wet=batch['dry'],batch['wet']
    encoded=model.encode(dry,wet)
    logits=torch.cat([head(encoded) for head in model.order],1)
    # No filenames, domain identifiers, order truth or mask enters the features.
    acoustic=np.stack([paired_transfer_features(x.numpy(),y.numpy(),44100) for x,y in zip(dry,wet)])
    return np.concatenate((acoustic,encoded.numpy(),logits.numpy()),axis=1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('output already exists')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    model=PairedEstimator()
    checkpoint=RUN/'paired-estimator.pt'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
    real=DAFxOrderDataset(CORPUS,'train')
    real=Subset(real,[i for i,r in enumerate(real.records) if len(r.order)>=2])
    synth=SyntheticControlDataset(dry_sources(CORPUS,'train'),3072,seed=20260921,order_equivalence_db=-30.)
    dataset=ConcatDataset([real,synth])
    xx,yy,mm=[],[],[]
    for index,batch in enumerate(DataLoader(dataset,batch_size=8)):
        keep=batch['order_mask'].any(1)
        if keep.any():
            batch={k:v[keep] for k,v in batch.items()}
            xx.append(transfer_batch(model,batch));yy.append(batch['order'].numpy());mm.append(batch['order_mask'].numpy())
        if (index+1)%32==0: print(json.dumps({'stage':'order-fit-features','batches':index+1,'total':len(dataset)//8}),flush=True)
    X,Y,M=np.concatenate(xx),np.concatenate(yy),np.concatenate(mm)
    np.savez_compressed(args.output/'fit-features.npz',X=X,Y=Y,M=M)
    candidates={}
    for kind in ('extra-trees','hist-boost'):
        heads=[]
        for relation in range(3):
            use=M[:,relation]>.5
            estimator=(ExtraTreesClassifier(n_estimators=300,max_features=.7,min_samples_leaf=2,n_jobs=2,random_state=913+relation)
                       if kind=='extra-trees' else HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=15,min_samples_leaf=12,l2_regularization=1,early_stopping=False,random_state=913+relation))
            print(json.dumps({'stage':'order-fit-head','candidate':kind,'relation':relation,'examples':int(use.sum())}),flush=True)
            estimator.fit(X[use],Y[use,relation].astype(int));heads.append(estimator)
        candidates[kind]=heads
    artifact=args.output/'candidates.joblib'
    joblib.dump({'schema':1,'feature_version':'paired-transfer-and-frozen-embedding-v1','candidates':candidates,'encoder_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()},artifact,compress=3)
    report={'schema':1,'examples':len(X),'relations':M.sum(0).tolist(),'feature_dimensions':X.shape[1],'pedalboard_training':False,'training_guitars':['les','prs'],'new_locked_final_audio_opened':False,'accepted':False,'physical_audio_devices_used':False,'model_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}
    (args.output/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
