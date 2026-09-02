#!/usr/bin/env python3
"""Select semantic inverse regressors without fitting on development labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge

from .asrnn_effects import effect_files, effect_spec, read_effect_pair
from .fit_asrnn_effect_output import _partition as effect_partition
from .paired_transfer_features import paired_transfer_features
from .train_asrnn_rat_adapter import _partition as rat_partition
from .train_asrnn_phase7 import _partition as cs3_partition


def partitions(corpus: Path, device: str):
    paths = effect_files(corpus, device, "train")
    if device == "cs3":
        audit = json.loads(Path("remix/runs/asrnn-cs3-phase7-group-audit.json").read_text())
        fit, calibration = cs3_partition(paths,audit)
    else:
        fit, calibration = (rat_partition if device=="rat" else effect_partition)(paths)
    return {"fit":fit,"calibration":calibration,"development":effect_files(corpus,device,"eval")}


def features(paths, device):
    X,Y,audible = [],[],[]
    for index,path in enumerate(paths):
        dry,wet,controls = read_effect_pair(path,device)
        X.append(paired_transfer_features(dry,wet))
        Y.append(controls)
        audible.append(float(np.max(np.abs(wet))) >= 0.001)
        if (index+1)%128==0:
            print(json.dumps({"features_completed":index+1,"total":len(paths),"device":device}),flush=True)
    return np.stack(X),np.stack(Y),np.asarray(audible)


def metrics(prediction,target,audible,device):
    # Normalized labels and search grids arrive as float32: 0.15 can be
    # represented as 0.15000000596. This tolerance covers representation only.
    tolerance=5e-8
    names = effect_spec(device).control_names
    if device=="rat": names=("distortion","tone","volume")
    rows = {}
    for index,name in enumerate(names):
        mask = audible if device=="rat" and name!="volume" else np.ones(len(target),dtype=bool)
        error = np.abs(prediction[:,index]-target[:,index])
        selected = error[mask]
        if not len(selected):raise ValueError(f'no identifiable examples for {name}')
        mae,p95 = (0.08,0.22) if device=="rat" and name!="volume" else (0.06,0.18)
        rows[name] = {"examples":int(mask.sum()),"mae":float(selected.mean()),"p95":float(np.quantile(selected,.95)),
                      "all_sample_mae":float(error.mean()),"all_sample_p95":float(np.quantile(error,.95)),
                      "gate_mae":mae,"gate_p95":p95,"passed":bool(selected.mean()<=mae+tolerance and np.quantile(selected,.95)<=p95+tolerance)}
    macro_mae=float(np.mean([row["mae"] for row in rows.values()]))
    macro_p95=float(np.mean([row["p95"] for row in rows.values()]))
    return {"examples":len(target),"audible":int(audible.sum()),"quiet":int((~audible).sum()),"per_control":rows,
            "macro_mae":macro_mae,"macro_p95":macro_p95,
            "comparison_absolute_tolerance":tolerance,
            "semantic_passed":all(row["passed"] for row in rows.values()) and (device=="rat" or (macro_mae<=.05+tolerance and macro_p95<=.15+tolerance))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=("rat","cs3","dfz"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--corpus",type=Path,default=Path("data/corpus/asrnn-physical-effects"))
    args=parser.parse_args()
    if args.output.exists(): raise ValueError("inverse output already exists")
    args.output.mkdir(parents=True)
    paths=partitions(args.corpus,args.device)
    matrices={split:features(items,args.device) for split,items in paths.items()}
    np.savez_compressed(args.output/"features.npz",**{f"{split}_{name}":array for split,values in matrices.items() for name,array in zip(("X","Y","audible"),values)})
    # ExtraTrees/RF are label-supervised models; no model receives a filename.
    candidates={
        "extra-trees":ExtraTreesRegressor(n_estimators=256,min_samples_leaf=1,max_features=.8,n_jobs=2,random_state=912),
        "random-forest":RandomForestRegressor(n_estimators=256,min_samples_leaf=1,max_features=.7,n_jobs=2,random_state=912),
        "hist-boost":MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=1,early_stopping=False,random_state=912)),
        "rbf-ridge":make_pipeline(StandardScaler(),KernelRidge(alpha=.01,kernel="rbf",gamma=1/matrices["fit"][0].shape[1])),
    }
    rows,models,predictions={},{},{}
    for name,model in candidates.items():
        print(json.dumps({"fitting":name,"device":args.device}),flush=True)
        model.fit(matrices["fit"][0],matrices["fit"][1])
        values=np.asarray(model.predict(matrices["calibration"][0])).reshape(len(paths["calibration"]),-1).clip(0,1)
        rows[name]=metrics(values,*matrices["calibration"][1:],args.device)
        models[name]=model
        predictions[name]=values
        print(json.dumps({"candidate":name,"calibration":rows[name]}),flush=True)
    # Select per-control heads only using calibration semantic errors.
    choices=[]
    for index in range(matrices["fit"][1].shape[1]):
        control=list(rows[next(iter(rows))]["per_control"])[index]
        choices.append(min(rows,key=lambda name:(not rows[name]["per_control"][control]["passed"],rows[name]["per_control"][control]["mae"]+.25*rows[name]["per_control"][control]["p95"])))
    retained={name:models[name] for name in set(choices)}
    payload={"schema":1,"device":args.device,"choices":choices,"models":retained,"feature_version":"paired-transfer-v1","source_license":"CC-BY-NC-4.0"}
    model_path=args.output/"inverse.joblib"
    joblib.dump(payload,model_path,compress=3)
    report={"schema":1,"device":args.device,"model_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),"choices":choices,
            "calibration_candidates":rows,"feature_dimensions":matrices["fit"][0].shape[1],
            "physical_audio_devices_used":False,"source_audio_modified":False,"new_locked_final_audio_opened":False,
            "accepted":False,"closed_loop_evaluated":False,"partitions":{k:[p.name for p in v] for k,v in paths.items()}}
    for split in ("calibration","development"):
        X,Y,audible=matrices[split]
        predicted={name:np.asarray(model.predict(X)).reshape(len(X),-1).clip(0,1) for name,model in retained.items()}
        prediction=np.stack([predicted[name][:,index] for index,name in enumerate(choices)],1)
        report[split]=metrics(prediction,Y,audible,args.device)
        report[split+"_predictions"]=[{"file":p.name,"truth":t.tolist(),"prediction":e.tolist(),"audible":bool(a)} for p,t,e,a in zip(paths[split],Y,prediction,audible)]
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"choices":choices,"calibration":report["calibration"],"development":report["development"]}),flush=True)


if __name__=="__main__": main()
