"""Read-only full-prefix peak diagnostics on the frozen DFZ calibration set."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from .asrnn_effects import effect_files,read_effect_pair
from .fit_asrnn_effect_output import _partition
from .stable_effect import load_stable_effect


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2)
    if args.output.exists():raise ValueError('diagnostic output exists')
    model,_=load_stable_effect(args.checkpoint)
    _,paths=_partition(effect_files(Path('data/corpus/asrnn-physical-effects'),'dfz','train'));rows=[]
    for offset in range(0,len(paths),4):
        chosen=paths[offset:offset+4];pairs=[read_effect_pair(path,'dfz') for path in chosen]
        dry=torch.from_numpy(np.stack([x for x,y,c in pairs]));controls=torch.from_numpy(np.stack([c for x,y,c in pairs]));state=None;chunks=[]
        for start in range(0,144000,4096):
            output,state=model(dry[:,start:start+4096],controls,state);chunks.append(output)
        predicted=torch.cat(chunks,1).numpy()
        for path,(x,y,c),v in zip(chosen,pairs,predicted):
            expected_index=int(np.abs(y[1024:]).argmax())+1024;predicted_index=int(np.abs(v[1024:]).argmax())+1024
            peak=float(abs(y[expected_index]));model_peak=float(abs(v[predicted_index]));lo=max(0,expected_index-1024);hi=min(len(y),expected_index+1024)
            rows.append({'file':path.name,'controls':c.tolist(),'expected_peak':peak,'predicted_peak':model_peak,'signed_peak_difference':model_peak-peak,'absolute_peak_error':abs(model_peak-peak),'expected_peak_time':expected_index/48000,'predicted_peak_time':predicted_index/48000,'error_at_expected_peak':float(v[expected_index]-y[expected_index]),'target_peak_neighborhood_mae':float(np.abs(v[lo:hi]-y[lo:hi]).mean()),'dry_peak_neighborhood_rms':float(np.sqrt(np.mean(x[lo:hi]**2))),'dry_sha256':hashlib.sha256(x.tobytes()).hexdigest()})
        print(json.dumps({'completed':len(rows),'total':len(paths)}),flush=True)
    rows.sort(key=lambda r:r['absolute_peak_error'],reverse=True)
    report={'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'rows':rows,'source_audio_modified':False,'physical_audio_devices_used':False,'official_eval_opened':False}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'worst':rows[:10]}),flush=True)


if __name__=='__main__':main()
