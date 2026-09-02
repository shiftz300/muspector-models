"""Smooth kernel order classifiers on the frozen training audio features."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import joblib, numpy as np, torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from .evaluate_order_search import make_domains
from .train_order_transfer import transfer_batch
from .model import PairedEstimator
from .train import CORPUS, RUN


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    original=Path('remix/runs/order-transfer-phase11');artifact=args.output/'kernel-heads.joblib'
    if artifact.exists():heads=joblib.load(artifact)
    else:
        arrays=np.load(original/'fit-features.npz');X,Y,M=(arrays[k] for k in ('X','Y','M'));heads={}
        for C in (1.,10.,100.):
            name=f'rbf-{C}';heads[name]=[]
            for relation in range(3):
                use=M[:,relation]>.5
                # Probability calibration uses internal training folds only.
                model=make_pipeline(StandardScaler(),SVC(C=C,kernel='rbf',gamma='scale',probability=True,cache_size=256,random_state=928+relation))
                model.fit(X[use],Y[use,relation].astype(int));heads[name].append(model)
                print(json.dumps({'stage':'kernel-fit','candidate':name,'relation':relation}),flush=True)
        joblib.dump(heads,artifact,compress=3)
    encoder=PairedEstimator();encoder.load_state_dict(torch.load(RUN/'paired-estimator.pt',map_location='cpu',weights_only=True));encoder.eval()
    for split in ('calibrate','valid'):
        for domain,dataset in make_domains(CORPUS,split,320,-30.).items():
            target=args.output/f'blend-{split}-{domain}.json'
            if target.exists():continue
            rows=json.loads((Path('remix/runs/order-specialist-phase11')/target.name).read_text())
            X=np.concatenate([transfer_batch(encoder,batch) for batch in DataLoader(dataset,batch_size=8)])
            if len(X)!=len(rows):raise ValueError('feature and audit row count differs')
            for name,models in heads.items():
                values=np.stack([model.predict_proba(X)[:,1] for model in models],1)
                for row,probability in zip(rows,values):
                    row['probabilities'][name]=probability.tolist()
                    for blend in ('hist-boost','interaction-hist'):
                        row['probabilities'][f'{name}+{blend}']=(.5*probability+.5*np.asarray(row['probabilities'][blend])).tolist()
            target.write_text(json.dumps(rows)+'\n')
            print(json.dumps({'stage':'kernel-audit','split':split,'domain':domain}),flush=True)
    report={'model_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'fit_cache_sha256':hashlib.sha256((original/'fit-features.npz').read_bytes()).hexdigest(),'fit_guitars':['les','prs'],'synthetic_seed':20260921,'pedalboard_training':False,'new_locked_final_audio_opened':False,'physical_audio_devices_used':False,'source_audio_modified':False}
    (args.output/'training.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
