"""Fit a larger order head on the unchanged fit cache and audit fresh features."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np,torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch.utils.data import DataLoader
from .evaluate_order_search import make_domains
from .train_order_transfer import transfer_batch
from .model import PairedEstimator
from .train import CORPUS,RUN


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2)
    path=args.run/'hist31.joblib'
    if path.exists():raise ValueError('additional candidate already exists')
    arrays=np.load(args.run/'fit-features.npz');X,Y,M=(arrays[k] for k in ('X','Y','M'))
    heads=[]
    for i in range(3):
        selected=M[:,i]>.5
        head=HistGradientBoostingClassifier(max_iter=350,max_leaf_nodes=31,min_samples_leaf=12,l2_regularization=2,early_stopping=False,random_state=919+i)
        head.fit(X[selected],Y[selected,i].astype(int));heads.append(head)
        print(json.dumps({'stage':'hist31-fit','relation':i}),flush=True)
    joblib.dump(heads,path,compress=3)
    encoder=PairedEstimator();encoder.load_state_dict(torch.load(RUN/'paired-estimator.pt',map_location='cpu',weights_only=True));encoder.eval()
    for split in ('calibrate','valid'):
        for domain,dataset in make_domains(CORPUS,split,320,-30.).items():
            features=np.concatenate([transfer_batch(encoder,batch) for batch in DataLoader(dataset,batch_size=8)])
            probabilities=np.stack([head.predict_proba(features)[:,1] for head in heads],1)
            for filename in (f'{split}-{domain}-audit.json',f'blend-{split}-{domain}.json'):
                file=args.run/filename
                if not file.exists():continue
                rows=json.loads(file.read_text())
                if len(rows)!=len(probabilities):raise ValueError('audit row alignment changed')
                for row,values in zip(rows,probabilities):row['probabilities']['hist31']=values.tolist()
                file.write_text(json.dumps(rows)+'\n')
            print(json.dumps({'stage':'hist31-audit','split':split,'domain':domain}),flush=True)
    (args.run/'hist31-provenance.json').write_text(json.dumps({'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'training_cache_sha256':hashlib.sha256((args.run/'fit-features.npz').read_bytes()).hexdigest(),'fit_guitars':['les','prs'],'pedalboard_training':False,'new_locked_final_audio_opened':False,'physical_audio_devices_used':False},indent=2)+'\n')


if __name__=='__main__':main()
