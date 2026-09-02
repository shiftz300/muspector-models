"""Replay-check accepted effect inverse runtimes with audio arrays only."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, torch
from .asrnn_effects import read_effect_pair
from .train_effect_inverse_features import partitions
from .semantic_effect_runtime import SemanticEffectRuntime
from .effect_admission import forward_teacher_evidence


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--compute',choices=('cpu','mps'),default='cpu');args=p.parse_args();torch.set_num_threads(2)
    acceptance_path=args.run/'acceptance.json';acceptance=json.loads(acceptance_path.read_text())
    evidence_path=args.run/'metrics.json';evidence=json.loads(evidence_path.read_text())
    if not acceptance['accepted'] or hashlib.sha256(evidence_path.read_bytes()).hexdigest()!=acceptance['inverse_evidence_sha256']:raise ValueError('inverse evidence is not accepted or changed')
    runtime=SemanticEffectRuntime(acceptance_path,args.checkpoint,args.compute,_allow_unverified_replay=True)
    paths=partitions(Path('data/corpus/asrnn-physical-effects'),acceptance['device'])['development']
    predictions=evidence['development_predictions'];results=[]
    for index in np.linspace(0,len(paths)-1,4,dtype=int):
        path=paths[index];row=predictions[index]
        if row['file']!=path.name:raise ValueError('replay row order differs')
        dry,wet,_=read_effect_pair(path,acceptance['device']);before=hashlib.sha256(dry.tobytes()+wet.tobytes()).hexdigest()
        prediction=np.asarray(runtime.infer(dry,wet)['controls'])
        result={'development_index':int(index),'max_control_error':float(np.abs(prediction-row['prediction']).max()),'source_arrays_unchanged':before==hashlib.sha256(dry.tobytes()+wet.tobytes()).hexdigest()}
        results.append(result);print(json.dumps(result),flush=True)
    accepted=all(r['max_control_error']<=2e-6 and r['source_arrays_unchanged'] for r in results)
    manifest={'schema':1,'device':acceptance['device'],'accepted':accepted,'status':'accepted-internal-noncommercial-development' if accepted else 'replay-failed','sample_rate':48000,'frames':144000,'renderer':{'path':str(args.checkpoint),'sha256':acceptance['renderer_sha256']},'acceptance_sha256':hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),'runtime_replay':results,'runtime_replay_compute':args.compute,'runtime_replay_verified':accepted,'semantic_metrics':acceptance['semantic_metrics']['development'],'closed_loop':acceptance['closed_loop'],'scope':'paired semantic inverse only; forward fidelity requires separate admission','forward_teacher_peak_gate_passed':acceptance['device']=='cs3','license':'CC-BY-NC-4.0','ui_integration_allowed':False,'physical_audio_devices_used':False,'source_audio_modified':False,'locked_final_evidence':False,'limitations':['requires aligned Dry/Wet three-second mono 48kHz pairs','DFZ forward peak fidelity is a separate gate','development evidence only; not universal pedal or independent final validation']}
    teacher_evidence=forward_teacher_evidence(args.checkpoint)
    manifest['forward_teacher_evidence']=teacher_evidence
    manifest['forward_teacher_peak_gate_passed']=teacher_evidence['passed']
    manifest['development_control_grid']=[sorted(set(float(r['truth'][i]) for r in predictions)) for i in range(len(predictions[0]['truth']))]
    manifest['calibration_control_grid']=[sorted(set(float(r['truth'][i]) for r in evidence['calibration_predictions'])) for i in range(len(predictions[0]['truth']))]
    manifest['unseen_control_interpolation_evaluated']=False
    manifest['limitations'].append('fine-grid search output does not establish accuracy at unseen intermediate hardware control settings')
    (args.run/'model-card.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'accepted':accepted,'device':acceptance['device']}),flush=True)


if __name__=='__main__':main()
