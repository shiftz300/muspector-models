#!/usr/bin/env python3
"""Train and gate a Wet-only Chorus delay-trajectory estimator."""
from __future__ import annotations
import argparse,hashlib,json,random,time
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .chorus3 import ChorusPairsV3,restore_chorus
from .chorus_model3 import ChorusTrajectoryV3
from .quality2 import summarize
from .train_modulation3 import _summarize_rows

SEED=20260926
def collate(rows):
 return {"wet":torch.stack([r["wet"] for r in rows]),"controls":torch.stack([r["controls"] for r in rows]),"lfo":torch.stack([r["lfo"] for r in rows])}
def loss_fn(pred,unc,target):
 target=F.interpolate(target[:,None],size=pred.shape[1],mode="linear",align_corners=False).squeeze(1)
 mse=F.mse_loss(pred,target); diff=F.l1_loss(torch.diff(pred,dim=1),torch.diff(target,dim=1))
 a=pred-pred.mean(1,keepdim=True);b=target-target.mean(1,keepdim=True)
 denominator=((a.square().sum(1)+1e-6)*(b.square().sum(1)+1e-6)).sqrt()
 corr=(a*b).sum(1)/denominator
 ul=F.l1_loss(unc,(pred-target).abs().detach()); total=mse+.5*(1-corr.mean())+.2*diff+.05*ul
 return total,{"mse":float(mse.detach()),"correlation":float(corr.mean().detach()),"difference":float(diff.detach()),"uncertainty":float(ul.detach())}
def mean_loss(model,data,batch,device):
 total=n=0;model.eval()
 with torch.inference_mode():
  for r in DataLoader(data,batch_size=batch,collate_fn=collate):
   p,u=model(r["wet"].to(device),r["controls"].to(device));l,_=loss_fn(p,u,r["lfo"].to(device));total+=float(l)*len(r["wet"]);n+=len(r["wet"])
 return total/n
def trajrow(p,t):
 a=p-p.mean();b=t-t.mean();corr=float(np.sum(a*b)/max(np.sqrt(np.sum(a*a)*np.sum(b*b)),1e-10));return {"normalized_mae":float(np.mean(abs(p-t))/2.0),"correlation":corr}
def quality(model,data):
 agg=([],[],[]);tr=[];sources=defaultdict(lambda:(([],[],[]),[]));strata=defaultdict(lambda:(([],[],[]),[]));model.eval()
 with torch.inference_mode():
  for i in range(len(data)):
   r=data[i];p,_=model(r["wet"][None],r["controls"][None]);full=F.interpolate(p[:,None],size=len(r["wet"]),mode="linear",align_corners=False)[0,0].numpy();rest=restore_chorus(r["wet"].numpy(),r["control_values"],full);s,e=r["target_start"],r["target_end"]
   vals=(r["wet"][s:e].numpy(),rest[s:e],r["clean"][s:e].numpy());row=trajrow(full[s:e],r["lfo"][s:e].numpy());key="slow-rate" if r["control_values"]["rate_hz"]<2.2 else "fast-rate"
   for ar,tt in ((agg,tr),sources[r["source_id"]],strata[key]):
    for q,v in zip(ar,vals):q.append(v)
    tt.append(row)
 rep=_summarize_rows(agg,tr);rep["sources"]={k:_summarize_rows(*v) for k,v in sorted(sources.items())};rep["strata"]={k:_summarize_rows(*v) for k,v in sorted(strata.items())};rep["all_sources_accepted"]=all(x["accepted"] for x in rep["sources"].values());rep["all_strata_accepted"]=all(x["accepted"] for x in rep["strata"].values());rep["accepted"]=bool(rep["accepted"] and rep["all_sources_accepted"] and rep["all_strata_accepted"]);return rep
def runtime(model,data):
 r=data[0];started=time.perf_counter()
 with torch.inference_mode():p,_=model(r["wet"][None],r["controls"][None]);full=F.interpolate(p[:,None],size=len(r["wet"]),mode="linear",align_corners=False)[0,0].numpy();restore_chorus(r["wet"].numpy(),r["control_values"],full)
 sec=time.perf_counter()-started;return {"frames":len(r["wet"]),"mean_seconds":sec,"realtime_factor":sec/(len(r["wet"])/48000),"ordinary_cpu":True,"python_reference_inverse":True,"audio_callback":False}
def train(a):
 w=a.workspace.resolve();fit=ChorusPairsV3(w,"fit",a.train_samples,a.target_frames,SEED+1);cal=ChorusPairsV3(w,"calibration",a.calibration_samples,a.target_frames,SEED+2);dev=ChorusPairsV3(w,"development",a.development_samples,a.target_frames,SEED+3);device=torch.device(a.device);m=ChorusTrajectoryV3(a.channels,a.depth).to(device);opt=torch.optim.AdamW(m.parameters(),lr=a.learning_rate,weight_decay=1e-5);loader=DataLoader(fit,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(SEED),collate_fn=collate)
 initial=mean_loss(m,cal,a.batch_size,device);hist=[{"epoch":0,"calibration_loss":initial}];best=initial;epoch_best=0;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};print(json.dumps({"family":"chorus","epoch":0,"calibration_loss":initial}),flush=True)
 for epoch in range(a.epochs):
  m.train();tot=defaultdict(float);n=0
  for r in loader:
   wet,con,tgt=r["wet"].to(device),r["controls"].to(device),r["lfo"].to(device);opt.zero_grad(set_to_none=True);p,u=m(wet,con);l,parts=loss_fn(p,u,tgt);l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step();c=len(wet);n+=c;tot["loss"]+=float(l.detach())*c
   for k,v in parts.items():tot[k]+=v*c
  cl=mean_loss(m,cal,a.batch_size,device);hist.append({"epoch":epoch+1,"train":{k:v/n for k,v in tot.items()},"calibration_loss":cl});print(json.dumps({"family":"chorus","epoch":epoch+1,"calibration_loss":cl}),flush=True)
  if cl<best:best=cl;epoch_best=epoch+1;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
 m=m.cpu();m.load_state_dict(state);cr=quality(m,cal);dr=quality(m,dev);rt=runtime(m,dev);accepted=bool(not a.quick and cr["accepted"] and dr["accepted"] and rt["realtime_factor"]<=.5);target=a.output.resolve()/"chorus";target.mkdir(parents=True,exist_ok=True);cp=target/"model.pt";torch.save({"schema":1,"sample_rate":48000,"architecture":m.manifest(),"state_dict":m.state_dict()},cp);sha=hashlib.sha256(cp.read_bytes()).hexdigest();report={"schema":1,"status":"accepted" if accepted else "diagnostic-not-promoted","accepted":accepted,"quick":a.quick,"mechanism":"modulation","family":"chorus","model":{**m.manifest(),"checkpoint":str(cp),"sha256":sha},"training":{"seed":SEED,"epochs":a.epochs,"selected_epoch":epoch_best,"selected_calibration_loss":best,"accelerator":device.type,"history":hist,"fit_samples_per_epoch":a.train_samples,"calibration_samples":a.calibration_samples,"development_samples":a.development_samples},"calibration":cr,"development":dr,"runtime":rt,"provenance":{"product_sources_only":True,"generic_repository_owned_dsp":True,"named_physical_device_claim":False,"authorized_source_ids":fit.authorization["sources"],"required_attribution":fit.authorization["required_attribution"],"research_source_ids":[],"generated_audio_written":False,"physical_audio_devices_used":False,"locked_final_audio_opened":False}}
 (target/"metrics.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");return report
def main():
 p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,default=Path.cwd());p.add_argument("--output",type=Path,default=Path("runs/foundation/product3-chorus"));p.add_argument("--device",choices=("cpu","mps"),default="cpu");p.add_argument("--quick",action="store_true");p.add_argument("--epochs",type=int,default=10);p.add_argument("--train-samples",type=int,default=144);p.add_argument("--calibration-samples",type=int,default=48);p.add_argument("--development-samples",type=int,default=72);p.add_argument("--target-frames",type=int,default=96000);p.add_argument("--batch-size",type=int,default=2);p.add_argument("--channels",type=int,default=16);p.add_argument("--depth",type=int,default=8);p.add_argument("--learning-rate",type=float,default=4e-4);a=p.parse_args()
 if a.quick:a.epochs=3;a.train_samples=36;a.calibration_samples=12;a.development_samples=16;a.channels=8;a.depth=6
 if a.device=="mps" and not torch.backends.mps.is_available():raise RuntimeError("MPS unavailable")
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);r=train(a);print(json.dumps({"family":"chorus","status":r["status"],"selected_epoch":r["training"]["selected_epoch"],"checkpoint":r["model"]["checkpoint"]},indent=2))
if __name__=="__main__":main()
