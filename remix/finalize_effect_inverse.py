"""Validate semantic decisions and complete causal reconstruction separately."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from .asrnn_effects import effect_files,read_effect_pair
from .stable_effect import load_stable_effect
from .train_effect_inverse_features import metrics
from .effect_admission import validate_inverse_teacher,forward_teacher_evidence


@torch.inference_mode()
def reconstruct(paths,estimates,model,device_key,target):
    recovered=[];oracle=[];bypass=[];rec_abs=oracle_abs=bypass_abs=0.;total=0
    for offset in range(0,len(paths),8):
        pairs=[read_effect_pair(p,device_key) for p in paths[offset:offset+8]]
        dry=torch.from_numpy(np.stack([p[0] for p in pairs])).to(target)
        wet=torch.from_numpy(np.stack([p[1] for p in pairs])).to(target)
        truth=torch.from_numpy(np.stack([p[2] for p in pairs])).to(target)
        estimated=torch.tensor(estimates[offset:offset+len(pairs)],dtype=torch.float32,device=target)
        def render(c):
            state=None;chunks=[]
            for start in range(0,dry.shape[1],2048):
                value,state=model(dry[:,start:start+2048],c,state);chunks.append(value)
            return torch.cat(chunks,1)[:,1024:]
        predicted=render(estimated);expected=render(truth);y=wet[:,1024:];x=dry[:,1024:]
        energy=y.square().mean(1).clamp_min(1e-8)
        recovered.extend(((predicted-y).square().mean(1)/energy).cpu().tolist())
        oracle.extend(((expected-y).square().mean(1)/energy).cpu().tolist())
        bypass.extend(((x-y).square().mean(1)/energy).cpu().tolist())
        rec_abs+=float((predicted-y).abs().sum());oracle_abs+=float((expected-y).abs().sum());bypass_abs+=float((x-y).abs().sum());total+=y.numel()
        if offset%32==0:print(json.dumps({'stage':'closed-loop','completed':offset+len(pairs),'total':len(paths)}),flush=True)
    return {'examples':len(paths),'mean_recovered_control_esr':float(np.mean(recovered)),'mean_oracle_control_esr':float(np.mean(oracle)),'mean_bypass_esr':float(np.mean(bypass)),
            'recovered_vs_bypass_esr_improvement':1-float(np.mean(recovered))/max(float(np.mean(bypass)),1e-12),'recovered_vs_bypass_mae_improvement':1-rec_abs/max(bypass_abs,1e-12),
            'recovered_global_mae':rec_abs/total,'oracle_global_mae':oracle_abs/total,'bypass_global_mae':bypass_abs/total}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--compute',choices=('cpu','mps'),default='cpu')
    args=p.parse_args();torch.set_num_threads(2)
    output=args.run/'acceptance.json'
    if output.exists():raise ValueError('inverse acceptance already exists')
    source=args.run/'metrics.json';report=json.loads(source.read_text());kind=report['device']
    measured={}
    for split in ('calibration','development'):
        rows=report[split+'_predictions'];estimated=np.array([r['prediction'] for r in rows]);truth=np.array([r['truth'] for r in rows]);audible=np.array([r['audible'] for r in rows])
        measured[split]=metrics(estimated,truth,audible,kind)
    if not all(r['semantic_passed'] for r in measured.values()):raise ValueError('semantic gates failed')
    paths=effect_files(Path('data/corpus/asrnn-physical-effects'),kind,'eval')
    if [p.name for p in paths]!=[r['file'] for r in report['development_predictions']]:raise ValueError('development row alignment mismatch')
    model,payload=load_stable_effect(args.checkpoint)
    validate_inverse_teacher(report,args.checkpoint,payload)
    teacher_evidence=forward_teacher_evidence(args.checkpoint)
    target=torch.device(args.compute);model=model.to(target)
    closure=reconstruct(paths,estimated,model,kind,target)
    accepted=closure['recovered_vs_bypass_esr_improvement']>=.75 and closure['recovered_vs_bypass_mae_improvement']>=.70
    result={'schema':1,'device':kind,'accepted':bool(accepted),'status':'accepted-internal-noncommercial-development' if accepted else 'rejected','semantic_metrics':measured,'closed_loop':closure,
            'inverse_evidence_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'renderer_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'compute':args.compute+'-float32','source_audio_modified':False,'physical_audio_devices_used':False,'locked_final_evidence':False,'ui_integration_allowed':False,
            'comparison_note':'5e-8 tolerance only for float32 normalized-control representation; no audio fidelity threshold changed',
            'scope':'paired semantic inverse only; forward fidelity requires separate admission',
            'forward_teacher_peak_gate_passed':teacher_evidence['passed'],'forward_teacher_evidence':teacher_evidence}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
