"""Regularized residual MLP order heads over immutable paired audio features."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from .evaluate_order_search import make_domains
from .train_order_transfer import transfer_batch
from .model import PairedEstimator
from .train import CORPUS,RUN


class OrderFeatureMLP(torch.nn.Module):
    def __init__(self,width):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(width,128),torch.nn.LayerNorm(128),torch.nn.GELU(),torch.nn.Dropout(.25),torch.nn.Linear(128,64),torch.nn.GELU(),torch.nn.Dropout(.15),torch.nn.Linear(64,3))
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self,features,base_logits):return self.net(features)+base_logits


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.manual_seed(931)
    source=Path('remix/runs/order-transfer-phase11');artifact=args.output/'neural-heads.pt'
    if artifact.exists():payload=torch.load(artifact,map_location='cpu',weights_only=True)
    else:
        arrays=np.load(source/'fit-features.npz');X,Y,M=(arrays[k] for k in ('X','Y','M'))
        scaler=StandardScaler().fit(X);features=torch.from_numpy(scaler.transform(X).clip(-8,8).astype(np.float32))
        labels=torch.from_numpy(Y.astype(np.float32));mask=torch.from_numpy(M.astype(np.float32));logits=torch.from_numpy(X[:,-3:].copy())
        model=OrderFeatureMLP(X.shape[1]);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.03)
        states={};history=[]
        for epoch in range(160):
            model.train();losses=[]
            for ids in torch.randperm(len(X)).split(128):
                prediction=model(features[ids]+torch.randn_like(features[ids])*.025,logits[ids])
                values=torch.nn.functional.binary_cross_entropy_with_logits(prediction,labels[ids],reduction='none')
                loss=(values*mask[ids]).sum()/mask[ids].sum().clamp_min(1)
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step();losses.append(float(loss.detach()))
            if (epoch+1)%20==0:
                history.append({'epoch':epoch+1,'training_loss':float(np.mean(losses))});print(json.dumps(history[-1]),flush=True)
            if epoch+1 in (40,80,160):states[f'mlp-{epoch+1}']={k:v.detach().clone() for k,v in model.state_dict().items()}
        payload={'schema':1,'width':X.shape[1],'mean':torch.tensor(scaler.mean_,dtype=torch.float32),'scale':torch.tensor(scaler.scale_,dtype=torch.float32),'states':states,'history':history,'fit_cache_sha256':hashlib.sha256((source/'fit-features.npz').read_bytes()).hexdigest()}
        torch.save(payload,artifact)
    encoder=PairedEstimator();encoder.load_state_dict(torch.load(RUN/'paired-estimator.pt',map_location='cpu',weights_only=True));encoder.eval()
    for split in ('calibrate','valid'):
        for domain,dataset in make_domains(CORPUS,split,320,-30.).items():
            target=args.output/f'blend-{split}-{domain}.json'
            if target.exists():continue
            rows=json.loads((Path('remix/runs/order-conditional-phase11')/target.name).read_text())
            X=torch.from_numpy(np.concatenate([transfer_batch(encoder,batch) for batch in DataLoader(dataset,batch_size=8)]));features=((X-payload['mean'])/payload['scale']).clamp(-8,8)
            for name,state in payload['states'].items():
                model=OrderFeatureMLP(payload['width']);model.load_state_dict(state);model.eval()
                with torch.inference_mode():probabilities=model(features,X[:,-3:]).sigmoid().numpy()
                for row,probability in zip(rows,probabilities):
                    row['probabilities'][name]=probability.tolist()
                    for other in ('hist-boost','interaction-hist','rbf-1.0+hist-boost'):
                        row['probabilities'][f'{name}+{other}']=(.5*probability+.5*np.array(row['probabilities'][other])).tolist()
            target.write_text(json.dumps(rows)+'\n');print(json.dumps({'stage':'neural-order-audit','split':split,'domain':domain}),flush=True)
    (args.output/'training.json').write_text(json.dumps({'model_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'fit_guitars':['les','prs'],'synthetic_seed':20260921,'pedalboard_training':False,'new_locked_final_audio_opened':False,'source_audio_modified':False,'physical_audio_devices_used':False},indent=2)+'\n')


if __name__=='__main__':main()
