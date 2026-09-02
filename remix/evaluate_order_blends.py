"""Add fixed causal DSP mixtures to order recovery without changing masks."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import torch
from .evaluate_order_search import make_domains,predictions,admissible,score
from .evaluate_order_transfer import domain_metrics,improved
from .model import PairedEstimator
from .drive_model import DriveControlEstimator
from .delay_model import DelayControlEstimator
from .reverb_model import ReverbControlEstimator
from .order_search import rank_topology_blends
from .train import CORPUS,RUN


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--split',choices=('calibrate','valid'),required=True)
    p.add_argument('--candidate-prefix',default=None,
                   help='Optional predeclared new model family for calibration; development always uses the frozen selection')
    p.add_argument('--selection-objective',choices=('macro','required-margin'),default='macro',
                   help='Calibration-only ranking after unchanged gates; required-margin maximizes the smaller real/Pedalboard gain')
    args=p.parse_args();torch.set_num_threads(2)
    models=(PairedEstimator(),DriveControlEstimator(),DelayControlEstimator(),ReverbControlEstimator())
    for model,name in zip(models,('paired','drive','delay','reverb')):
        model.load_state_dict(torch.load(RUN/f'{name}-estimator.pt',map_location='cpu',weights_only=True));model.eval()
    domains={}
    for domain,dataset in make_domains(CORPUS,args.split,320,-30.).items():
        target=args.run/f'blend-{args.split}-{domain}.json'
        if target.exists(): domains[domain]=json.loads(target.read_text());continue
        rows=json.loads((args.run/f'{args.split}-{domain}-audit.json').read_text())
        _,controls,records=predictions(dataset,*models,torch.device('cpu'))
        for index,(row,item,c) in enumerate(zip(rows,records,controls)):
            row['baseline_error']=next(v['error'] for v in row['errors'] if v['topology']==row['baseline'])
            if len(row['active'])>=2:
                row['errors']=rank_topology_blends(item['dry'],item['wet'],tuple(row['active']),c)
            row['controls']=c.tolist()
            if (index+1)%64==0: print(json.dumps({'stage':'blend-bank','split':args.split,'domain':domain,'completed':index+1,'total':len(rows)}),flush=True)
        target.write_text(json.dumps(rows)+'\n');domains[domain]=rows
    baseline=domain_metrics(domains)
    historical=json.loads((RUN/'order-search-metrics.json').read_text())
    historical_baseline=historical['calibration_selected' if args.split=='calibrate' else 'validation_selected']
    selection_path=args.run/'blend-calibration.json'
    if args.split=='calibrate':
        sweep={};choices=[]
        names=tuple(next(iter(domains.values()))[0]['probabilities'])
        if args.candidate_prefix is not None:
            names=tuple(name for name in names if name.startswith(args.candidate_prefix))
            if not names:raise ValueError('candidate prefix matches no calibrated model')
        for name in names:
            for weight in (.25,.5,.75,1.):
                for margin in (0.,.1,.25):
                    current=domain_metrics(domains,name,weight,margin)
                    sweep[f'{name}:{weight}:{margin}']=current
                    if admissible(current,baseline) and admissible(current,historical_baseline):
                        primary=(score(current) if args.selection_objective=='macro' else
                                 min(current[domain]['exact']-max(baseline[domain]['exact'],historical_baseline[domain]['exact'])
                                     for domain in ('real','pedalboard')))
                        choices.append((improved(current,baseline) and improved(current,historical_baseline),(primary,score(current)),name,weight,margin))
        selected=max(choices)[2:] if choices else (None,0,0)
        report={'selected':selected,'baseline':baseline,'candidate':domain_metrics(domains,*selected),'sweep':sweep}
        report['candidate_prefix']=args.candidate_prefix
        report['selection_objective']=args.selection_objective
        report['historical_baseline']=historical_baseline
        report['baseline_policy']='Require both historical and current CPU gates'
        report['passed']=improved(report['candidate'],baseline) and improved(report['candidate'],historical_baseline)
        selection_path.write_text(json.dumps(report,indent=2)+'\n')
    else:
        frozen=json.loads(selection_path.read_text());selected=frozen['selected']
        report={'selected':selected,'baseline':baseline,'candidate':domain_metrics(domains,*selected),'calibration_sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest()}
        report['historical_baseline']=historical_baseline
        report['passed']=improved(report['candidate'],baseline) and improved(report['candidate'],historical_baseline) and frozen['passed']
        (args.run/'blend-development.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sweep'}),flush=True)


if __name__=='__main__': main()
