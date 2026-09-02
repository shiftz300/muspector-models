"""Calibrate transfer order heads with frozen audio-quality fallback rules."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from .evaluate_order_search import make_domains,predictions,score,admissible
from .train_order_transfer import transfer_batch
from .model import PairedEstimator
from .drive_model import DriveControlEstimator
from .delay_model import DelayControlEstimator
from .reverb_model import ReverbControlEstimator
from .order_search import rank_topologies,search_margin
from .spec import KINDS,ranked_topologies,order_targets
from .train import CORPUS,RUN


def decision(row,candidate,weight,margin):
    base=tuple(row['baseline'])
    if candidate is None: return base
    probabilities=np.clip(np.asarray(row['probabilities'][candidate]),1e-6,1-1e-6)
    learned=np.log(probabilities/(1-probabilities))
    logits=(1-weight)*np.asarray(row['logits'])+weight*learned
    top=tuple(row['active'])
    ranked=ranked_topologies(top,logits)
    choice=ranked[0][0]
    # A changed topology must have no larger residual than the old decision.
    # This uses measured audio only, never order truth or the evaluation mask.
    if len(ranked)>1 and ranked[0][1]-ranked[1][1]<margin: return base
    errors={tuple(v['topology']):v['error'] for v in row['errors']}
    baseline_error = row.get('baseline_error', errors.get(base,0))
    return choice if errors.get(choice,0)<=baseline_error+1e-12 else base


def measure(rows,candidate=None,weight=0,margin=0):
    correct=relations=exact=count=changed=0;error=0.
    for row in rows:
        mask=np.asarray(row['mask'])>.5
        if not mask.any(): continue
        choice=decision(row,candidate,weight,margin)
        pred=np.asarray(order_targets(choice)[0])>.5
        match=pred==(np.asarray(row['truth'])>.5)
        correct+=int((match&mask).sum());relations+=int(mask.sum())
        exact+=int((match|~mask).all());count+=1;changed+=choice!=tuple(row['baseline'])
        chosen_error=next(v['error'] for v in row['errors'] if tuple(v['topology'])==choice)
        error+=row.get('baseline_error',chosen_error) if candidate is None else chosen_error
    return {'exact':exact/max(count,1),'pairwise':correct/max(relations,1),'examples':count,'relations':relations,'changed':changed,'mean_gain_aligned_reconstruction_error':error/max(count,1)}


def domain_metrics(domains,candidate=None,weight=0,margin=0):
    return {name:measure(rows,candidate,weight,margin) for name,rows in domains.items()}


def exact_improvement_five_points(value,baseline):
    """Compare +1/20 using audited integer counts, not binary-float sums.

    254/320 is exactly five points above 238/320, but float arithmetic makes
    .74375 + .05 = .7937500000000001 and incorrectly rejects 254 successes.
    Cross multiplication preserves the gate exactly; it adds no tolerance.
    """
    count,base_count=int(value['examples']),int(baseline['examples'])
    if count<=0 or base_count<=0:raise ValueError('empty exact-order audit')
    successes=int(round(value['exact']*count));base_successes=int(round(baseline['exact']*base_count))
    if abs(value['exact']*count-successes)>1e-8 or abs(baseline['exact']*base_count-base_successes)>1e-8:
        raise ValueError('exact accuracy does not match integer audit counts')
    return 20*(successes*base_count-base_successes*count)>=count*base_count


def improved(values,baseline):
    return (admissible(values,baseline)
            and all(values[k]['mean_gain_aligned_reconstruction_error']<=baseline[k]['mean_gain_aligned_reconstruction_error']+1e-9 for k in baseline)
            and all(exact_improvement_five_points(values[k],baseline[k]) for k in ('real','pedalboard')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    source=args.run/'candidates.joblib'
    payload=joblib.load(source)
    if payload['encoder_sha256']!=hashlib.sha256((RUN/'paired-estimator.pt').read_bytes()).hexdigest(): raise ValueError('encoder changed')
    models=(PairedEstimator(),DriveControlEstimator(),DelayControlEstimator(),ReverbControlEstimator())
    for model,name in zip(models,('paired','drive','delay','reverb')):
        model.load_state_dict(torch.load(RUN/f'{name}-estimator.pt',map_location='cpu',weights_only=True));model.eval()
    baseline_report=json.loads((RUN/'order-search-metrics.json').read_text())
    samples=baseline_report['samples_per_synthetic_domain']
    threshold=baseline_report['selected_margin_threshold']
    selected=None;reports={};calibration_sweep={};replay_differences=[]
    for split in ('calibrate','valid'):
        evaluated={}
        for domain,dataset in make_domains(CORPUS,split,samples,-30.).items():
            cache=args.run/f'{split}-{domain}-audit.json'
            if cache.exists():
                evaluated[domain]=json.loads(cache.read_text());continue
            logits,controls,records=predictions(dataset,*models,torch.device('cpu'))
            rows=[]
            for index,item in enumerate(records):
                active=tuple(KINDS[i] for i in item['topology'] if i>=0)
                # Source topology is used only as the same active-family set
                # already supplied to the baseline, never its order.
                families=tuple(sorted(active))
                features=transfer_batch(models[0],{'dry':torch.from_numpy(item['dry'])[None],'wet':torch.from_numpy(item['wet'])[None]})
                probabilities={name:[float(head.predict_proba(features)[0,1]) for head in heads] for name,heads in payload['candidates'].items()}
                classifier=ranked_topologies(families,logits[index])[0][0]
                ranked=rank_topologies(item['dry'],item['wet'],families,controls[index]) if len(families)>=2 else [(families,0.)]
                baseline=ranked[0][0] if search_margin(ranked)>=threshold else classifier
                rows.append({'truth':item['order'].tolist(),'mask':item['order_mask'].tolist(),'active':families,'logits':logits[index].tolist(),'probabilities':probabilities,'baseline':baseline,'errors':[{'topology':top,'error':error} for top,error in ranked]})
                if (index+1)%64==0: print(json.dumps({'split':split,'domain':domain,'completed':index+1,'total':len(records)}),flush=True)
            cache.write_text(json.dumps(rows)+'\n');evaluated[domain]=rows
        baseline=domain_metrics(evaluated)
        if split=='calibrate':
            choices=[]
            for name in payload['candidates']:
                for weight in (.25,.5,.75,1.):
                    for margin in (0.,.1,.25):
                        current=domain_metrics(evaluated,name,weight,margin)
                        key=f'{name}:{weight}:{margin}'
                        calibration_sweep[key]=current
                        if admissible(current,baseline): choices.append((improved(current,baseline),score(current),name,weight,margin))
            if choices:
                _,_,name,weight,margin=max(choices);selected=(name,weight,margin)
            else: selected=(None,0,0)
        else:
            for domain,old in baseline_report['validation_selected'].items():
                for key in ('exact','pairwise','examples','relations','mean_gain_aligned_reconstruction_error'):
                    if abs(baseline[domain][key]-old[key])>1e-6:
                        replay_differences.append({'domain':domain,'metric':key,'historical':old[key],'current':baseline[domain][key]})
        reports[split]={'baseline':baseline,'selected':domain_metrics(evaluated,*selected)}
        print(json.dumps({'split':split,'selected':selected,'metrics':reports[split]}),flush=True)
    accepted=all(improved(reports[split]['selected'],baseline) for split,key in (('calibrate','calibration_selected'),('valid','validation_selected')) for baseline in (reports[split]['baseline'],baseline_report[key]))
    report={'schema':1,'accepted':bool(accepted),'selected':selected,'calibration_sweep':calibration_sweep,**reports,'physical_audio_devices_used':False,'source_audio_modified':False,'new_locked_final_audio_opened':False,'candidate_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'samples_per_synthetic_domain':samples,'equivalence_db':-30.,'limitation':'development order evidence; unchanged recovered controls; real hardware chains not claimed'}
    (args.run/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    report['baseline_replay_differences']=replay_differences
    report['baseline_policy']='Pass BOTH unchanged historical gates and freshly replayed CPU gates; do not replace historical numbers with easier ones.'
    (args.run/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'accepted':bool(accepted),'selected':selected}),flush=True)


if __name__=='__main__': main()
