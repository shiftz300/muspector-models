"""Add real-corpus specialists; route only by audio residual, never filenames."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np,torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch.utils.data import DataLoader
from .data import DAFxOrderDataset
from .evaluate_order_search import make_domains
from .train_order_transfer import transfer_batch
from .model import PairedEstimator
from .spec import order_targets
from .train import CORPUS,RUN


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    source=Path('remix/runs/order-interactions-phase11');old=Path('remix/runs/order-transfer-phase11')
    artifact=args.output/'real-heads.joblib'
    if artifact.exists():heads=joblib.load(artifact)
    else:
        records=[r for r in DAFxOrderDataset(CORPUS,'train').records if len(r.order)>=2]
        arrays=np.load(old/'fit-features.npz');X,Y,M=(arrays[k][:len(records)] for k in ('X','Y','M'))
        if not np.array_equal(Y,np.array([order_targets(r.order)[0] for r in records])):raise ValueError('real training prefix does not match labels')
        heads={}
        for leaves in (15,31):
            name=f'real-hist{leaves}';heads[name]=[]
            for relation in range(3):
                use=M[:,relation]>.5
                model=HistGradientBoostingClassifier(max_iter=350,max_leaf_nodes=leaves,min_samples_leaf=10,l2_regularization=2,early_stopping=False,random_state=923+relation)
                model.fit(X[use],Y[use,relation].astype(int));heads[name].append(model)
                print(json.dumps({'stage':'real-specialist-fit','candidate':name,'relation':relation}),flush=True)
        joblib.dump(heads,artifact,compress=3)
    encoder=PairedEstimator();encoder.load_state_dict(torch.load(RUN/'paired-estimator.pt',map_location='cpu',weights_only=True));encoder.eval()
    for split in ('calibrate','valid'):
        for domain,dataset in make_domains(CORPUS,split,320,-30.).items():
            target=args.output/f'blend-{split}-{domain}.json'
            if target.exists():continue
            rows=json.loads((source/f'blend-{split}-{domain}.json').read_text())
            features=np.concatenate([transfer_batch(encoder,batch) for batch in DataLoader(dataset,batch_size=8)])
            for name,models in heads.items():
                probs=np.stack([h.predict_proba(features)[:,1] for h in models],1)
                for row,values in zip(rows,probs):
                    row['probabilities'][name]=values.tolist()
                    general=np.array(row['probabilities']['interaction-hist'])
                    for weight in (.25,.5,.75):row['probabilities'][f'mix-{name}-{weight}']=((1-weight)*general+weight*values).tolist()
                    for threshold in (.4,.5,.6,.7,.8):
                        row['probabilities'][f'route-{name}-{threshold}']=(values if row['baseline_error']>=threshold else general).tolist()
            target.write_text(json.dumps(rows)+'\n')
            print(json.dumps({'stage':'real-specialist-audit','split':split,'domain':domain}),flush=True)
    (args.output/'training.json').write_text(json.dumps({'model_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'fit_guitars':['les','prs'],'routing_signal':'gain-aligned residual of the frozen baseline; no dataset or source filename enters inference','pedalboard_training':False,'physical_audio_devices_used':False,'new_locked_final_audio_opened':False},indent=2)+'\n')


if __name__=='__main__':main()
