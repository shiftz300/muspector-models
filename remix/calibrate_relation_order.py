"""Select relation-specific frozen heads using calibration only.

The three relations need not share a model. Head candidates and the search
budget are fixed here; development labels are loaded only after selection is
serialized and hashed. The unchanged reconstruction guard applies throughout.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json
from pathlib import Path
import numpy as np
from .evaluate_order_transfer import improved
from .evaluate_order_search import score,admissible
from .spec import ranked_topologies,order_targets
from .train import RUN


def matrices(rows):
    active=[row for row in rows if any(v>.5 for v in row['mask'])]
    result=[]
    for row in active:
        tops=list(itertools.permutations(row['active']))
        targets=np.asarray([order_targets(t)[0] for t in tops])
        errors={tuple(v['topology']):v['error'] for v in row['errors']}
        result.append((tops,targets,np.asarray([errors[t] for t in tops]),row))
    return result


def evaluate(domain_data,names):
    result={}
    for domain,items in domain_data.items():
        exact=correct=relations=changed=0;error=0.
        for tops,targets,errors,row in items:
            probs=np.asarray([row['probabilities'][n][k] for k,n in enumerate(names)])
            probs=np.clip(probs,1e-6,1-1e-6);logits=np.log(probs/(1-probs))
            # ranked_topologies scores each present pair; absent relations have
            # identical constant contributions and cannot change the choice.
            presence=np.asarray(order_targets(tuple(row['active']))[1])
            values=((2*targets-1)*logits*presence).sum(1)
            index=int(np.argmax(values))
            if errors[index]>row['baseline_error']+1e-12:index=tops.index(tuple(row['baseline']))
            mask=np.asarray(row['mask'])>.5
            match=(targets[index]>.5)==(np.asarray(row['truth'])>.5)
            exact+=bool((match|~mask).all());correct+=int((match&mask).sum());relations+=int(mask.sum())
            error+=errors[index];changed+=tops[index]!=tuple(row['baseline'])
        result[domain]={'exact':exact/len(items),'pairwise':correct/relations,'examples':len(items),'relations':relations,'changed':changed,'mean_gain_aligned_reconstruction_error':error/len(items)}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frontier',choices=('balanced','real-and-balanced'),default='balanced')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    domains=('real','reference','alternate','stress','pedalboard')
    rows={d:json.loads((args.source/f'blend-calibrate-{d}.json').read_text()) for d in domains}
    historical=json.loads((RUN/'order-search-metrics.json').read_text())
    fresh=json.loads((args.source/'blend-calibration.json').read_text())['baseline']
    names=tuple(rows['real'][0]['probabilities']);choices=[]
    for relation in range(3):
        ranked=[]
        for name in names:
            acc={}
            signature=[]
            for domain,items in rows.items():
                use=[r for r in items if r['mask'][relation]>.5]
                pred=np.asarray([r['probabilities'][name][relation]>=.5 for r in use]);truth=np.asarray([r['truth'][relation]>.5 for r in use])
                acc[domain]=float(np.mean(pred==truth)) if len(use) else 1.
                signature.extend(pred.tolist())
            # Real and independent-renderer transfer carry the two required
            # improvements, while the other three domains remain guardrails.
            quality=(2*acc['real']+2*acc['pedalboard']+acc['reference']+acc['alternate']+acc['stress'])/7
            ranked.append((quality,name,tuple(signature),acc['real']))
        distinct=[];seen=set()
        ordered=sorted(ranked,reverse=True)
        if args.frontier=='real-and-balanced':
            # Fixed budget: expose the real-domain frontier as well as the
            # macro frontier; never select this frontier using development.
            real_order=sorted(ranked,key=lambda item:(item[3],item[0],item[1]),reverse=True)
            ordered=[item for pair in zip(real_order,ordered) for item in pair]
        for _,name,signature,_ in ordered:
            if signature not in seen:distinct.append(name);seen.add(signature)
            if len(distinct)==6:break
        choices.append(distinct)
    data={d:matrices(v) for d,v in rows.items()};selected=None;best=None;sweep=[]
    for index,combination in enumerate(itertools.product(*choices)):
        metrics=evaluate(data,combination)
        valid=admissible(metrics,fresh) and admissible(metrics,historical['calibration_selected'])
        passed=improved(metrics,fresh) and improved(metrics,historical['calibration_selected'])
        minimum_gain=min(metrics[d]['exact']-max(fresh[d]['exact'],historical['calibration_selected'][d]['exact']) for d in ('real','pedalboard'))
        key=(passed,valid,minimum_gain,score(metrics))
        if best is None or key>best:best=key;selected=combination;candidate=metrics
        sweep.append({'names':combination,'passed':passed,'minimum_required_domain_gain':minimum_gain,'score':score(metrics)})
        if (index+1)%36==0:print(json.dumps({'stage':'relation-calibration','completed':index+1,'total':int(np.prod([len(x) for x in choices]))}),flush=True)
    target=args.output/'calibration.json'
    frozen={'names':selected,'passed':best[0],'candidate':candidate,'baseline':fresh,'historical_baseline':historical['calibration_selected'],'choices':choices,'sweep':sweep,'source':str(args.source),'frontier':args.frontier,'selection':'max minimum required-domain gain after unchanged all-domain gates; calibration only'}
    target.write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps({k:v for k,v in frozen.items() if k not in ('sweep','choices')}),flush=True)
    # Development does not influence either individual-head selection or the
    # selected joint tuple. Only the just-serialized tuple is replayed.
    frozen=json.loads(target.read_text());selected=frozen['names']
    rows={d:json.loads((args.source/f'blend-valid-{d}.json').read_text()) for d in domains}
    candidate=evaluate({d:matrices(v) for d,v in rows.items()},selected)
    fresh=json.loads((args.source/'blend-development.json').read_text())['baseline']
    report={'names':selected,'calibration_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'passed':frozen['passed'] and improved(candidate,fresh) and improved(candidate,historical['validation_selected']),'candidate':candidate,'baseline':fresh,'historical_baseline':historical['validation_selected'],'source_audio_modified':False,'physical_audio_devices_used':False,'new_locked_final_audio_opened':False}
    (args.output/'development.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
