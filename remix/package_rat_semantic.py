"""Package and replay-check the accepted semantic RAT inverse on audio only."""
from __future__ import annotations
import argparse,hashlib,json,tempfile
from pathlib import Path
import joblib,numpy as np,torch
from .asrnn_effects import effect_files,read_effect_pair
from .semantic_rat_runtime import RATSemanticRuntime


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    root=Path('remix/runs/rat-identifiable-phase10')
    metrics=json.loads((root/'metrics.json').read_text())
    if not metrics['accepted']:raise ValueError('RAT semantic candidate failed')
    target=root/'semantic-heads.joblib'
    if target.exists() and not args.replay_only:raise ValueError('package already exists')
    if not args.replay_only:
        tone=joblib.load(root/'tone.joblib')
        original=joblib.load('remix/runs/rat-semantic-phase10/inverse.joblib')
        joblib.dump({'schema':1,'tone':tone['model'],'volume':original['models']['hist-boost'].estimators_[2]},target,compress=3)
    artifacts={'semantic_heads':target,'initializer':Path('remix/runs/asrnn-rat-inverse-pilot/rat-inverse.pt'),'renderer':Path('remix/runs/asrnn-rat-stable-pilot/rat-stable.pt')}
    manifest={'schema':1,'device':'rat','accepted':True,'status':'accepted-internal-noncommercial-development','sample_rate':48000,'frames':48000,'artifacts':{name:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in artifacts.items()},'semantic_metrics':metrics['development'],'closed_loop':metrics['closed_loop'],'license':'CC-BY-NC-4.0','ui_integration_allowed':False,'physical_audio_devices_used':False,'locked_final_evidence':False,'limitations':['22 of 128 development pairs have upstream controls unidentifiable at Wet peak below .001','paired Dry/Wet required','one-second mono 48kHz research domain; no arbitrary audio or universal pedal claim'],'runtime_replay_verified':False}
    manifest_path=root/'model-card.json'
    manifest['runtime_replay_compute']='cpu'
    manifest['control_convention']={'internal':['distortion','tone','volume'],'physical':['distortion','filter','volume'],'physical_filter':'1 - internal tone'}
    manifest['development_control_grid']=[sorted(set(float(r['truth'][i]) for r in metrics['development_predictions'])) for i in range(3)]
    manifest['unseen_control_interpolation_evaluated']=False
    manifest['calibration_control_grid']=[sorted(set(float(r['truth'][i]) for r in metrics['calibration_predictions'])) for i in range(3)]
    manifest['limitations'].append('accuracy at unseen intermediate hardware control settings has not been independently evaluated')
    manifest['accepted']=False
    # A cancelled replay must not replace the last completed admission report.
    with tempfile.TemporaryDirectory(prefix='muspector-rat-replay-') as temporary:
        provisional=Path(temporary)/'model-card.json'
        provisional.write_text(json.dumps(manifest,indent=2)+'\n')
        runtime=RATSemanticRuntime(provisional,_allow_unverified_replay=True)
    paths=effect_files(Path('data/corpus/asrnn-physical-effects'),'rat','eval')[:8]
    pairs=[read_effect_pair(p,'rat') for p in paths]
    dry=np.stack([v[0] for v in pairs]);wet=np.stack([v[1] for v in pairs])
    before=hashlib.sha256(dry.tobytes()+wet.tobytes()).hexdigest()
    prediction=runtime.infer(dry,wet)['controls']
    expected=np.array([r['prediction'] for r in metrics['development_predictions'][:8]])
    error=float(np.abs(prediction-expected).max())
    unchanged=before==hashlib.sha256(dry.tobytes()+wet.tobytes()).hexdigest()
    manifest['runtime_replay']={'examples':8,'max_control_error':error,'source_arrays_unchanged':unchanged}
    manifest['runtime_replay_verified']=bool(error<=2e-6 and unchanged)
    manifest['accepted']=manifest['runtime_replay_verified']
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'accepted':manifest['accepted'],'runtime_replay':manifest['runtime_replay']}),flush=True)


if __name__=='__main__':main()
