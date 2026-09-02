#!/usr/bin/env python3
"""Evaluate and package the conservative generic Flanger subdomain."""
from __future__ import annotations
import argparse,hashlib,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from .flanger3 import FlangerPairsV3,admitted,manifest,restore_flanger
from .train_chorus3 import SEED,trajrow
from .train_modulation3 import _summarize_rows

def quality(data):
 agg=([],[],[]);tr=[];sources=defaultdict(lambda:(([],[],[]),[]));abst=[]
 for i in range(len(data)):
  r=data[i]
  if not admitted(r["control_values"]):abst.append(i);continue
  restored,ok,lfo=restore_flanger(r["wet"].numpy(),r["control_values"]);assert ok;s,e=r["target_start"],r["target_end"];vals=(r["wet"][s:e].numpy(),restored[s:e],r["clean"][s:e].numpy());row=trajrow(lfo[s:e],r["lfo"][s:e].numpy())
  for ar,tt in ((agg,tr),sources[r["source_id"]]):
   for a,v in zip(ar,vals):a.append(v)
   tt.append(row)
 rep=_summarize_rows(agg,tr);rep["attempted_examples"]=len(data);rep["accepted_examples"]=len(tr);rep["coverage"]=len(tr)/len(data);rep["abstained_indices"]=abst;rep["sources"]={k:_summarize_rows(*v) for k,v in sorted(sources.items())};rep["all_sources_accepted"]=all(v["accepted"] for v in rep["sources"].values());rep["accepted"]=bool(rep["accepted"] and rep["all_sources_accepted"] and rep["coverage"]>=.25);return rep
def main():
 p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,default=Path.cwd());p.add_argument("--output",type=Path,default=Path("runs/foundation/product3-flanger-safe"));a=p.parse_args();w=a.workspace.resolve();cal=FlangerPairsV3(w,"calibration",48,96000,SEED+2);challenge=FlangerPairsV3(w,"development",120,96000,SEED+1003);cr=quality(cal);dr=quality(challenge);sample=challenge[next(i for i in range(len(challenge)) if admitted(challenge[i]["control_values"]))];started=time.perf_counter();restore_flanger(sample["wet"].numpy(),sample["control_values"]);sec=time.perf_counter()-started;runtime={"frames":len(sample["wet"]),"mean_seconds":sec,"realtime_factor":sec/(len(sample["wet"])/48000),"ordinary_cpu":True,"bounded_phase_window":True,"python_reference_inverse":True,"audio_callback":False};accepted=bool(cr["accepted"] and dr["accepted"] and runtime["realtime_factor"]<=.5);target=a.output.resolve()/"flanger";target.mkdir(parents=True,exist_ok=True);cp=target/"model.pt";torch.save({"schema":1,"sample_rate":48000,"architecture":manifest()},cp);sha=hashlib.sha256(cp.read_bytes()).hexdigest();report={"schema":1,"status":"accepted-safe-subdomain" if accepted else "diagnostic-not-promoted","accepted":accepted,"mechanism":"modulation","family":"flanger","model":{**manifest(),"checkpoint":str(cp),"sha256":sha},"calibration":{"report":cr,"revision_note":"initial development exposed fast-rate interaction; final conservative domain was frozen before the fresh control challenge"},"development_control_challenge":{"seed":SEED+1003,"same_held_out_source_groups_new_controls_and_crops":True,"report":dr},"runtime":runtime,"provenance":{"product_sources_only":True,"generic_repository_owned_dsp":True,"named_physical_device_claim":False,"authorized_source_ids":cal.authorization["sources"],"required_attribution":cal.authorization["required_attribution"],"research_source_ids":[],"physical_audio_devices_used":False,"locked_final_audio_opened":False,"demo_generated":False}};(target/"metrics.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps({"status":report["status"],"coverage":dr["coverage"],"checkpoint":str(cp),"metrics":str(target/"metrics.json")},indent=2))
if __name__=="__main__":main()
