#!/usr/bin/env python3
"""Train a software-model RAT fingerprint with leave-one-device-group-out validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from remix.a2 import A2Runtime, loader
from remix.packages import digest
from remix.probe import FEATURES, hashes, probes
from remix.signature import encode
from train.signature import GATES, choose, models, weights


SEED = 20260901


def candidate(name: str):
    if name == "trees": return ExtraTreesClassifier(n_estimators=480,max_features="sqrt",min_samples_leaf=1,class_weight="balanced",random_state=SEED,n_jobs=2)
    if name == "forest": return RandomForestClassifier(n_estimators=480,max_features="sqrt",min_samples_leaf=1,class_weight="balanced",random_state=SEED,n_jobs=2)
    if name == "hist": return HistGradientBoostingClassifier(learning_rate=.05,max_iter=300,max_leaf_nodes=8,min_samples_leaf=4,l2_regularization=.2,random_state=SEED)
    if name == "rbf": return make_pipeline(StandardScaler(),SVC(C=2.0,gamma="scale",class_weight="balanced",probability=True,random_state=SEED))
    raise ValueError(name)


def fit(model, x, y, group):
    sample_weight = weights(group)
    if hasattr(model, "named_steps"):
        model.fit(x,y,svc__sample_weight=sample_weight)
    else:
        model.fit(x,y,sample_weight=sample_weight)


def aggregate(scores: np.ndarray, truth: np.ndarray, groups: np.ndarray, names: np.ndarray):
    unique = list(dict.fromkeys(names.tolist()))
    return (np.asarray([np.median(scores[names == name]) for name in unique]),
            np.asarray([truth[names == name][0] for name in unique]),
            np.asarray([groups[names == name][0] for name in unique]), np.asarray(unique))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",type=Path,required=True);parser.add_argument("--nam",type=Path,required=True)
    parser.add_argument("--deps",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"probe run already exists: {args.output}")
    init=loader(args.nam,args.deps); dry_set=probes(); values=[];truth=[];groups=[];names=[];records=[]
    for positive,name,filename,page,license_id in models():
        path=args.models/filename; runtime=A2Runtime(path,init); group=page.rsplit("/",1)[-1]
        for dry in dry_set:
            before=hashlib.sha256(dry.tobytes()).hexdigest();wet=runtime.render(dry)
            values.append(encode(dry,wet,48_000));truth.append(positive);groups.append(group);names.append(name)
            if before!=hashlib.sha256(dry.tobytes()).hexdigest(): raise ValueError("probe source mutation")
        runtime.assert_unchanged();records.append({"scope":"rat" if positive else "other","model":name,"group":group,"path":path.as_posix(),"sha256":digest(path),"page":page,"license":license_id})
        print(json.dumps({"stage":"render","model":name}),flush=True)
    x=np.stack(values);y=np.asarray(truth,bool);g=np.asarray(groups);n=np.asarray(names)
    screens={}
    for kind in ("trees","forest","hist","rbf"):
        oof=np.zeros(len(y))
        for held in sorted(set(groups)):
            valid=g==held;train=~valid;model=candidate(kind);fit(model,x[train],y[train],g[train]);oof[valid]=model.predict_proba(x[valid])[:,1]
        model_score,model_y,model_g,model_names=aggregate(oof,y,g,n)
        threshold,report,failures=choose(model_score,model_y,model_g)
        screens[kind]={"threshold":threshold,"metrics":report,"failures":failures,"oof":model_score,"names":model_names}
        print(json.dumps({"stage":"screen","candidate":kind,"metrics":report,"failures":failures}),flush=True)
    selected_name,selected=max(screens.items(),key=lambda row:(not row[1]["failures"],row[1]["metrics"]["balanced_accuracy"],row[1]["metrics"]["worst_positive_group_recall"],row[0]))
    final=candidate(selected_name);fit(final,x,y,g);args.output.mkdir(parents=True)
    artifact=args.output/"probe.joblib";joblib.dump({"schema":1,"device":"rat","features":FEATURES,"probe_sha256":hashes(),"threshold":selected["threshold"],"model":final},artifact)
    np.savez_compressed(args.output/"data.npz",x=x,y=y,groups=g,names=n)
    (args.output/"data.json").write_text(json.dumps({"schema":1,"status":"complete","models":records,"probes":[{"sha256":value,"frames":240000,"rate":48000} for value in hashes()],"rendered_audio_retained":False,"physical_audio_devices_used":False},indent=2,sort_keys=True)+"\n")
    report={"schema":1,"status":"accepted-development" if not selected["failures"] else "rejected","accepted":not selected["failures"],"architecture":"fixed-probe software-model fingerprint","selection":selected_name,"threshold":selected["threshold"],"gates":GATES,"metrics":selected["metrics"],"failures":selected["failures"],"screens":{name:{"threshold":row["threshold"],"metrics":row["metrics"],"failures":row["failures"]} for name,row in screens.items()},"artifact":{"path":"probe.joblib","bytes":artifact.stat().st_size,"sha256":digest(artifact)},"scope":"software capture models only; requires a new untouched multi-author seal","physical_audio_devices_used":False,"automatic_delivery":False,"source_audio_modified":False}
    (args.output/"valid.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"accepted":report["accepted"],"selection":selected_name,"metrics":selected["metrics"],"failures":report["failures"]}))
    if not report["accepted"]: raise SystemExit(2)


if __name__=="__main__":main()
