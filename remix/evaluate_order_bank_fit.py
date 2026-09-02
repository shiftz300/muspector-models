"""Calibrate a bounded bank menu with the order classifier already frozen."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from .evaluate_order_search import make_domains,admissible,score
from .evaluate_order_transfer import domain_metrics,improved
from .order_bank_fit import MENUS,CLEAN_FRACTIONS,CONTROL_RADIUS,needs_fit,raw_proposal,fit_topology_options,apply_bank_fit
from .train import CORPUS,RUN


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_hash(item):
    digest=hashlib.sha256()
    for key in ('dry','wet'):
        value=item[key].numpy()
        digest.update(str((value.dtype.str,value.shape)).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def cache_signature(source,split,domain,selected,head_hash,menus):
    # Bind actual model and audit bytes, not just filenames or a candidate name.
    paths=[source/f'blend-{split}-{domain}.json',
           Path('remix/runs/order-interactions-phase11/interaction-heads.joblib'),
           Path('remix/runs/order-cascade-physics-phase11/physics-heads.joblib'),
           Path('remix/runs/order-temporal-phase11/temporal-heads.joblib'),
           RUN/'remixer-bundle.pt',RUN/'order-search-metrics.json']
    paths += [Path(__file__).parent/name for name in (
        'evaluate_order_bank_fit.py','order_bank_fit.py','order_search.py','render.py','spec.py',
        'data.py','pedalboard_data.py','quality.py','evaluate_order_search.py','evaluate_order_transfer.py')]
    return {'schema':2,'source_sha256':{str(p):file_hash(p) for p in paths},
            'split':split,'domain':domain,'selection':selected,'head_calibration_sha256':head_hash,
            'menus':list(menus),'control_radius':CONTROL_RADIUS,'clean_fractions':list(CLEAN_FRACTIONS),
            'samples_per_synthetic_domain':320,'equivalence_db':-30.}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('remix/runs/order-cascade-physics-phase11'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=('calibrate','valid'),required=True)
    p.add_argument('--frozen-bank-menu',choices=MENUS)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    head_calibration=args.source/'blend-calibration.json';frozen=json.loads(head_calibration.read_text());selected=frozen['selected']
    head_hash=hashlib.sha256(head_calibration.read_bytes()).hexdigest()
    selection_file=args.output/'blend-calibration.json'
    if args.split=='valid':
        chosen=json.loads(selection_file.read_text())
        if chosen['head_calibration_sha256']!=head_hash:raise ValueError('head calibration changed')
        if chosen['control_radius']!=CONTROL_RADIUS or chosen['parallel_clean_fractions']!=list(CLEAN_FRACTIONS):raise ValueError('frozen bank bounds changed')
        menu=chosen['bank_menu']
        if args.frozen_bank_menu and args.frozen_bank_menu!=menu:raise ValueError('frozen bank menu differs')
    menus=(menu,) if args.split=='valid' else ((args.frozen_bank_menu,) if args.frozen_bank_menu else MENUS)
    domains={};fit_counts={}
    for domain,dataset in make_domains(CORPUS,args.split,320,-30.).items():
        target=args.output/f'bank-{args.split}-{domain}.json'
        signature=cache_signature(args.source,args.split,domain,selected,head_hash,menus)
        manifest=target.with_suffix('.manifest.json')
        if target.exists():
            metadata=json.loads(manifest.read_text())
            if metadata['signature']!=signature or metadata['cache_sha256']!=file_hash(target):raise ValueError('stale or changed bank cache')
            rows=json.loads(target.read_text())
            if len(rows)!=len(dataset):raise ValueError('cached dataset length differs')
            for index,row in enumerate(rows):
                if row['audio_sha256']!=pair_hash(dataset[index]):raise ValueError('bank cache audio identity changed')
        else:
            rows=json.loads((args.source/f'blend-{args.split}-{domain}.json').read_text())
            # Keep only the frozen classifier in new caches, not the prior
            # experimental probability zoo. Truth/masks are audit outputs only.
            for row in rows:row['probabilities']={selected[0]:row['probabilities'][selected[0]]}
            if len(rows)!=len(dataset):raise ValueError('source dataset length differs')
            for index,row in enumerate(rows):
                item=dataset[index];row['audio_sha256']=pair_hash(item)
                if not np.array_equal(item['order'].numpy(),row['truth']) or not np.array_equal(item['order_mask'].numpy(),row['mask']):raise ValueError('source audit labels or eligibility differ')
                if needs_fit(row,selected):
                    row['bank_fit']=fit_topology_options(item['dry'].numpy(),item['wet'].numpy(),raw_proposal(row,selected),row['controls'],menus=menus)
                    print(json.dumps({'stage':'bounded-bank-fit','split':args.split,'domain':domain,'row':index,'total':len(rows)}),flush=True)
            target.write_text(json.dumps(rows)+'\n')
            manifest.write_text(json.dumps({'signature':signature,'cache_sha256':file_hash(target)},indent=2)+'\n')
        domains[domain]=rows;fit_counts[domain]=sum(bool(r.get('bank_fit')) for r in rows)
    baseline=domain_metrics(domains);historical=json.loads((RUN/'order-search-metrics.json').read_text())
    historical=historical['calibration_selected' if args.split=='calibrate' else 'validation_selected']
    if args.split=='calibrate':
        sweep={};choices=[]
        for index,menu in enumerate(menus):
            applied={d:[apply_bank_fit(row,selected,menu) for row in rows] for d,rows in domains.items()}
            metrics=domain_metrics(applied,*selected);passed=improved(metrics,baseline) and improved(metrics,historical)
            minimum=min(metrics[d]['exact']-max(baseline[d]['exact'],historical[d]['exact']) for d in ('real','pedalboard'))
            sweep[menu]=metrics;choices.append((passed,admissible(metrics,baseline) and admissible(metrics,historical),minimum,score(metrics),-index,menu))
        menu=max(choices)[-1]
        report={'selected':selected,'bank_menu':menu,'candidate':sweep[menu],'baseline':baseline,'historical_baseline':historical,'sweep':sweep,'head_source':str(args.source),'head_calibration_sha256':head_hash}
        report['passed']=improved(report['candidate'],baseline) and improved(report['candidate'],historical)
    else:
        applied={d:[apply_bank_fit(row,selected,menu) for row in rows] for d,rows in domains.items()}
        metrics=domain_metrics(applied,*selected)
        report={'selected':selected,'bank_menu':menu,'candidate':metrics,'baseline':baseline,'historical_baseline':historical,'head_source':str(args.source),'head_calibration_sha256':head_hash,'calibration_sha256':hashlib.sha256(selection_file.read_bytes()).hexdigest(),'passed':chosen['passed'] and improved(metrics,baseline) and improved(metrics,historical)}
    report.update({'fit_counts':fit_counts,'control_radius':CONTROL_RADIUS,'parallel_clean_fractions':list(CLEAN_FRACTIONS),'guard_unchanged':True,'actual_chain_controls_unchanged':True,'diagnostic_gain_applied_to_output':False,'new_locked_final_audio_opened':False,'source_audio_modified':False,'physical_audio_devices_used':False})
    report['cache_manifests_sha256']={f'bank-{args.split}-{d}.manifest.json':file_hash(args.output/f'bank-{args.split}-{d}.manifest.json') for d in domains}
    report['frozen_bank_menu_argument']=args.frozen_bank_menu
    destination=selection_file if args.split=='calibrate' else args.output/'blend-development.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    # Standard compact audit caches contain only the frozen menu after it was
    # selected on calibration; development always uses that same menu.
    for domain,rows in domains.items():
        (args.output/f'blend-{args.split}-{domain}.json').write_text(json.dumps([apply_bank_fit(r,selected,menu) for r in rows])+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sweep'}),flush=True)


if __name__=='__main__':main()
