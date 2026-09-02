"""Condition order classifiers on the known unordered active-family set."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np,torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from .data import DAFxOrderDataset,SyntheticControlDataset,dry_sources
from .evaluate_order_search import make_domains
from .train_order_transfer import transfer_batch
from .model import PairedEstimator
from .spec import order_targets
from .train import CORPUS,RUN


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    original=Path('remix/runs/order-transfer-phase11');artifact=args.output/'conditional-heads.joblib'
    if artifact.exists():heads=joblib.load(artifact)
    else:
        arrays=np.load(original/'fit-features.npz');X,Y,M=(arrays[k] for k in ('X','Y','M'))
        records=[r for r in DAFxOrderDataset(CORPUS,'train').records if len(r.order)>=2]
        context=[len(r.order)==3 for r in records];truth=[order_targets(r.order)[0] for r in records]
        synth=SyntheticControlDataset(dry_sources(CORPUS,'train'),3072,seed=20260921,order_equivalence_db=-30.)
        for index,batch in enumerate(DataLoader(synth,batch_size=8)):
            keep=batch['order_mask'].any(1)
            truth.extend(batch['order'][keep].tolist());context.extend((batch['topology'][keep]>=0).sum(1).eq(3).tolist())
            if (index+1)%64==0:print(json.dumps({'stage':'audit-unordered-training-context','batch':index+1}),flush=True)
        if not np.array_equal(np.asarray(truth),Y):raise ValueError('training context row order differs')
        context=np.asarray(context);heads={}
        for name in ('context-hist','context-rbf'):
            heads[name]={}
            for third in (False,True):
                heads[name][third]=[]
                for relation in range(3):
                    use=(M[:,relation]>.5)&(context==third)
                    model=(HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=2,early_stopping=False,random_state=930+relation) if name=='context-hist' else make_pipeline(StandardScaler(),SVC(C=10.,gamma='scale',probability=True,cache_size=256,random_state=930+relation)))
                    model.fit(X[use],Y[use,relation].astype(int));heads[name][third].append(model)
                    print(json.dumps({'stage':'conditional-order-fit','candidate':name,'third_stage_present':third,'relation':relation,'examples':int(use.sum())}),flush=True)
        joblib.dump(heads,artifact,compress=3)
    encoder=PairedEstimator();encoder.load_state_dict(torch.load(RUN/'paired-estimator.pt',map_location='cpu',weights_only=True));encoder.eval()
    for split in ('calibrate','valid'):
        for domain,dataset in make_domains(CORPUS,split,320,-30.).items():
            target=args.output/f'blend-{split}-{domain}.json'
            if target.exists():continue
            rows=json.loads((Path('remix/runs/order-kernel-phase11')/target.name).read_text())
            X=np.concatenate([transfer_batch(encoder,batch) for batch in DataLoader(dataset,batch_size=8)])
            context=np.array([len(row['active'])==3 for row in rows])
            if len(X)!=len(rows):raise ValueError('audit geometry differs')
            for name,variants in heads.items():
                predictions=np.zeros((len(X),3))
                for third,models in variants.items():
                    use=context==third
                    if use.any():predictions[use]=np.stack([model.predict_proba(X[use])[:,1] for model in models],1)
                for row,probability in zip(rows,predictions):
                    row['probabilities'][name]=probability.tolist()
                    for blend in ('hist-boost','interaction-hist','rbf-1.0+hist-boost'):
                        row['probabilities'][f'{name}+{blend}']=(.5*probability+.5*np.asarray(row['probabilities'][blend])).tolist()
            target.write_text(json.dumps(rows)+'\n');print(json.dumps({'stage':'conditional-order-audit','split':split,'domain':domain}),flush=True)
    report={'model_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'fit_cache_sha256':hashlib.sha256((original/'fit-features.npz').read_bytes()).hexdigest(),'fit_guitars':['les','prs'],'synthetic_seed':20260921,'context':'unordered active family set already supplied to frozen baseline; no truth order or evaluation mask enters inference','pedalboard_training':False,'new_locked_final_audio_opened':False,'physical_audio_devices_used':False,'source_audio_modified':False}
    (args.output/'training.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
