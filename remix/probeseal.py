#!/usr/bin/env python3
"""One-shot independent seal for the software-model fixed-probe classifier."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path

from .a2 import A2Runtime, loader
from .packages import digest
from .probe import ProbeRuntime, hashes


NAM = {"repository":"https://github.com/sdatkinson/neural-amp-modeler","version":"0.13.0",
       "commit":"f26112906de06ec6b796ad6d1982e29eed83144e","tree":"f8cd0987ec2cff11d4b886b794f6672575dfeda8","license":"MIT"}
MODELS = (
    ("rat","turbo1","seal_turbo1.nam","https://api.tone3000.com/storage/v1/object/public/models/ouynw7vtqeg.nam","https://www.tone3000.com/tones/proco-turbo-rat-74392","T3K"),
    ("rat","turbo2","seal_turbo2.nam","https://api.tone3000.com/storage/v1/object/public/models/6rdsaf7j77.nam","https://www.tone3000.com/tones/proco-turbo-rat-74392","T3K"),
    ("rat","boardrat1","seal_boardrat1.nam","https://api.tone3000.com/storage/v1/object/public/models/gw1100os347_a2.nam","https://www.tone3000.com/tones/pedals-from-my-pedal-board-51809","T3K"),
    ("rat","boardrat2","seal_boardrat2.nam","https://api.tone3000.com/storage/v1/object/public/models/pxsh0dplmb_a2.nam","https://www.tone3000.com/tones/pedals-from-my-pedal-board-51809","T3K"),
    ("other","hyper1","seal_hyper1.nam","https://api.tone3000.com/storage/v1/object/public/models/w26x22fxuem_a2.nam","https://www.tone3000.com/tones/pedals-from-my-pedal-board-51809","T3K"),
    ("other","hyper2","seal_hyper2.nam","https://api.tone3000.com/storage/v1/object/public/models/6hu6vgp8waa_a2.nam","https://www.tone3000.com/tones/pedals-from-my-pedal-board-51809","T3K"),
    ("other","mosky","seal_mosky.nam","https://api.tone3000.com/storage/v1/object/public/models/nnxgwn9xepk_a2.nam","https://www.tone3000.com/tones/pedals-from-my-pedal-board-51809","T3K"),
    ("other","guvnor","seal_guvnor.nam","https://api.tone3000.com/storage/v1/object/public/models/6d7cc2bf7af32012_a2.nam","https://www.tone3000.com/tones/marshall-the-guvnor-distortion-pedal-1726","T3K"),
)
GATES={"rat_recall":.75,"other_false_route":.10,"rat_models":4,"other_models":4,
       "automatic_deliveries":0,"source_mutations":0,"nonfinite_scores":0}


def gate(rows):
    rat=[row for row in rows if row["scope"]=="rat"];other=[row for row in rows if row["scope"]=="other"]
    metrics={"rat_examples":len(rat),"other_examples":len(other),"rat_models":len({r['model'] for r in rat}),
             "other_models":len({r['model'] for r in other}),"rat_recall":sum(r['routed'] for r in rat)/max(len(rat),1),
             "other_false_route":sum(r['routed'] for r in other)/max(len(other),1),
             "automatic_deliveries":sum(r['automatic_delivery'] for r in rows),
             "source_mutations":sum(not r['inputs_unchanged'] for r in rows),
             "nonfinite_scores":sum(not r['finite'] for r in rows)}
    failures=[]
    if metrics["rat_models"]<GATES["rat_models"]:failures.append("rat_models")
    if metrics["other_models"]<GATES["other_models"]:failures.append("other_models")
    if metrics["rat_recall"]<GATES["rat_recall"]:failures.append("rat_recall")
    if metrics["other_false_route"]>GATES["other_false_route"]:failures.append("other_false_route")
    for name in ("automatic_deliveries","source_mutations","nonfinite_scores"):
        if metrics[name]!=GATES[name]:failures.append(name)
    return not failures,failures,metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--models",type=Path,required=True)
    parser.add_argument("--probe",type=Path,required=True);parser.add_argument("--nam",type=Path,required=True)
    parser.add_argument("--deps",type=Path,required=True);parser.add_argument("--lock",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(f"probe seal already exists: {args.output}")
    head=subprocess.check_output(["git","-C",str(args.nam),"rev-parse","HEAD"],text=True).strip()
    tree=subprocess.check_output(["git","-C",str(args.nam),"rev-parse","HEAD:nam"],text=True).strip()
    if head!=NAM["commit"] or tree!=NAM["tree"]:raise ValueError("official NAM revision differs")
    records=[]
    for scope,name,filename,url,page,license_id in MODELS:
        path=args.models/filename;document=json.loads(path.read_text())
        records.append({"scope":scope,"id":name,"filename":filename,"url":url,"page":page,"license":license_id,
                        "bytes":path.stat().st_size,"sha256":digest(path),"metadata":document.get("metadata") or {}})
    lock={"schema":1,"kind":"independent-software-model-probe-seal","gates":GATES,
          "probe_artifact_sha256":digest(args.probe),"probe_sha256":hashes(),
          "probe_runtime_sha256":digest(Path(__file__).with_name("probe.py")),
          "signature_runtime_sha256":digest(Path(__file__).with_name("signature.py")),
          "seal_sha256":digest(Path(__file__)),"nam":NAM,"models":records,
          "environment":{name:importlib.metadata.version(name) for name in ("numpy","torch","scipy","joblib","scikit-learn")},
          "model_files_redistributed":False,"physical_audio_devices_used":False,"retuning_after_lock":False}
    if args.lock.exists():
        if json.loads(args.lock.read_text())!=lock:raise ValueError("existing probe seal lock differs")
    else:
        args.lock.parent.mkdir(parents=True,exist_ok=True);args.lock.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n")
    runtime=ProbeRuntime(args.probe);init=loader(args.nam,args.deps);rows=[]
    for record in records:
        model=A2Runtime(args.models/record["filename"],init);before=digest(args.models/record["filename"])
        result=runtime.infer_model(model);model.assert_unchanged()
        row={"scope":record["scope"],"model":record["id"],"routed":result["decision"]=="candidate",
             "score":result["score"],"probe_scores":result["probe_scores"],"threshold":result["threshold"],
             "automatic_delivery":result["automatic_delivery"],"inputs_unchanged":before==digest(args.models/record["filename"]),
             "finite":all(float('-inf')<value<float('inf') for value in result["probe_scores"])}
        rows.append(row);print(json.dumps(row),flush=True)
    passed,failures,metrics=gate(rows);runtime.assert_artifacts_unchanged()
    models_unchanged=all(digest(args.models/r["filename"])==r["sha256"] for r in records)
    if not models_unchanged:passed=False;failures.append("model-file-mutation")
    report={"schema":1,"status":"accepted-software-model-seal" if passed else "rejected","accepted":passed,
            "scope":"untouched multi-author NAM A2 software models; no claim for arbitrary audio or physical devices",
            "lock_sha256":digest(args.lock),"metrics":metrics,"failures":failures,"rows":rows,
            "artifacts_unchanged":True,"model_files_unchanged":models_unchanged,"source_audio_modified":False,
            "physical_audio_devices_used":False,"automatic_normalization":False,"automatic_limiting":False,
            "lossy_reencoding":False,"intermediate_audio_retained":False,"retuning_after_lock":False}
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"accepted":passed,"metrics":metrics,"failures":failures}))
    if not passed:raise SystemExit(2)


if __name__=="__main__":main()
