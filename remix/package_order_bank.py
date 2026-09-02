"""Package frozen order heads and replay their actual audio-only runtime.

This is a development replay, never an independent final evaluation. Diagnostic
knob fits are not applied to the actual chain or either input audio array.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,torch
from .evaluate_order_bank_fit import pair_hash,cache_signature
from .evaluate_order_search import make_domains
from .evaluate_order_transfer import decision,domain_metrics,improved
from .inference import BUNDLE
from .order_bank_fit import CONTROL_RADIUS,CLEAN_FRACTIONS,apply_bank_fit
from .order_transfer_runtime import TransferRemixerRuntime,PACKAGE_SELECTION,PACKAGE_SOURCES,file_hash,assert_file_hashes,read_json_hashed,numerical_environment
from .train import CORPUS,RUN


def checked_evidence(run):
    calibration,calibration_hash=read_json_hashed(run/'blend-calibration.json')
    development,development_hash=read_json_hashed(run/'blend-development.json')
    if development['calibration_sha256']!=calibration_hash:raise ValueError('development is not bound to calibration')
    for split,report in (('calibrate',calibration),('valid',development)):
        if not report['passed'] or report['selected']!=PACKAGE_SELECTION or report['bank_menu']!='bounded-knobs':raise ValueError('frozen order gate not passed')
        expected_domains=('real','reference','alternate','stress','pedalboard')
        if set(report['cache_manifests_sha256'])!={f'bank-{split}-{domain}.manifest.json' for domain in expected_domains}:raise ValueError('five-domain cache manifest coverage differs')
        head_source=Path(report['head_source']);head_calibration=head_source/'blend-calibration.json'
        if file_hash(head_calibration)!=report['head_calibration_sha256'] or json.loads(head_calibration.read_text())['selected']!=report['selected']:raise ValueError('frozen classifier calibration binding differs')
        if report['control_radius']!=CONTROL_RADIUS or report['parallel_clean_fractions']!=list(CLEAN_FRACTIONS):raise ValueError('frozen bank bounds differ')
        domains={}
        for filename,digest in report['cache_manifests_sha256'].items():
            manifest_path=run/filename
            manifest,manifest_hash=read_json_hashed(manifest_path)
            if manifest_hash!=digest:raise ValueError('cache manifest changed')
            signature=manifest['signature']
            domain=filename.removeprefix(f'bank-{split}-').removesuffix('.manifest.json')
            expected=cache_signature(head_source,split,domain,report['selected'],report['head_calibration_sha256'],('bounded-knobs',))
            if signature!=expected:raise ValueError('cache source identity, split, menu or classifier binding differs')
            assert_file_hashes(signature['source_sha256'])
            domain=signature['domain'];bank=run/f'bank-{split}-{domain}.json'
            rows,bank_hash=read_json_hashed(bank)
            if bank_hash!=manifest['cache_sha256']:raise ValueError('bank cache changed')
            applied=[apply_bank_fit(row,report['selected'],report['bank_menu']) for row in rows]
            if json.loads((run/f'blend-{split}-{domain}.json').read_text())!=applied:raise ValueError('blend cache differs from signed source bank')
            domains[domain]=applied
        metrics=domain_metrics(domains,*report['selected'])
        baseline=domain_metrics(domains)
        historical=json.loads((RUN/'order-search-metrics.json').read_text())['calibration_selected' if split=='calibrate' else 'validation_selected']
        if baseline!=report['baseline'] or historical!=report['historical_baseline']:raise ValueError('reported baselines differ from actual frozen audits')
        if metrics!=report['candidate']:raise ValueError('aggregate metrics do not reproduce from cache')
        if not (improved(metrics,report['baseline']) and improved(metrics,report['historical_baseline'])):raise ValueError('frozen dual-baseline gate failed')
    assert_file_hashes({str(run/'blend-calibration.json'):calibration_hash,str(run/'blend-development.json'):development_hash})
    return calibration,development


def package(run):
    calibration,development=checked_evidence(run)
    if (run/'model-card.json').exists() or (run/'order-heads.joblib').exists():raise ValueError('refusing to replace an existing order package')
    bundle=torch.load(BUNDLE,map_location='cpu',weights_only=True)
    # Training used standalone checkpoint files. Their actual tensor states,
    # not filenames, must equal the canonical runtime bundle for all controls.
    for kind,filename in (('main','paired'),('drive','drive'),('delay','delay'),('reverb','reverb')):
        state=torch.load(RUN/f'{filename}-estimator.pt',map_location='cpu',weights_only=True)
        if set(state)!=set(bundle['states'][kind]) or any(not torch.equal(value,bundle['states'][kind][key]) for key,value in state.items()):raise ValueError(f'{kind} bundle state differs from audited standalone model')
    sources={
        'interaction':Path('remix/runs/order-interactions-phase11/interaction-heads.joblib'),
        'physics':Path('remix/runs/order-cascade-physics-phase11/physics-heads.joblib'),
        'temporal':Path('remix/runs/order-temporal-phase11/temporal-heads.joblib')}
    heads={'schema':1,'selection':PACKAGE_SELECTION,
           'interaction':joblib.load(sources['interaction'])['candidates']['interaction-hist'][:2],
           'physics':joblib.load(sources['physics'])['hist'],
           'temporal':joblib.load(sources['temporal'])['temporal-rbf'][2]}
    joblib.dump(heads,run/'order-heads.joblib',compress=3)
    artifacts=['order-heads.joblib','blend-calibration.json','blend-development.json']
    artifacts+=list(calibration['cache_manifests_sha256'])+list(development['cache_manifests_sha256'])
    card={'schema':1,'selection':PACKAGE_SELECTION,'bank_menu':'bounded-knobs',
          'numerical_environment':numerical_environment(),
          'analysis_frontend':'preserve-input-levels-float32-copy',
          'file_entrypoint':'TransferRemixerRuntime.infer_files / load_pair; do not pre-normalize through the legacy load_analysis_pair default',
          'analysis_sample_rate':44100,'analysis_frames':220500,'analysis_batch_size':8,
          'analysis_input_contract':'Aligned finite mono float32 analysis arrays at 44100 Hz, exactly 5 seconds. Preserve their provided common input level; no inferred gain or domain-dependent normalization. DAFx audit pairs were jointly peak-scaled upstream; synthetic/Pedalboard retained dry peak 0.22. Arbitrary input-level invariance is not claimed. Source files and rendered output levels are never changed by this analysis frontend.',
          'control_radius':CONTROL_RADIUS,'parallel_clean_fractions':list(CLEAN_FRACTIONS),
          'canonical_bundle_sha256':file_hash(BUNDLE),
          'artifacts_sha256':{name:file_hash(run/name) for name in artifacts},
          'source_sha256':{name:file_hash(Path(__file__).parent/name) for name in PACKAGE_SOURCES},
          'training_artifacts_sha256':{str(path):file_hash(path) for path in sources.values()},
          'training_artifacts_are_provenance_only':True,'standalone_and_bundle_states_equal':True,
          'architecture':'relations 0/1: 0.5 interaction-Hist + 0.5 physics-only-Hist; relation 2: invariant+temporal RBF',
          'order_admitted':False,'runtime_replay_verified':False,'scope':'five-domain order development only; actual full-chain quality replay remains separate',
          'source_audio_modified':False,'physical_audio_devices_used':False,'new_locked_final_audio_opened':False,
          'diagnostic_gain_applied_to_output':False,'actual_chain_controls_unchanged':True}
    (run/'model-card.json').write_text(json.dumps(card,indent=2)+'\n')


def replay(run):
    checked_evidence(run);runtime=TransferRemixerRuntime(run,allow_candidate=True)
    card_path=run/'model-card.json';card=json.loads(card_path.read_text())
    if card['order_admitted']:raise ValueError('refusing to rerun and replace an admitted package audit')
    rows_out=[];differences=[];maximum_probability=maximum_control=0.;start=time.monotonic()
    for domain,dataset in make_domains(CORPUS,'valid',320,-30.).items():
        rows=json.loads((run/f'blend-valid-{domain}.json').read_text())
        chosen={index for index,row in enumerate(rows) if row.get('bank_fit')}
        # All fitted proposals plus first/middle/last of EVERY active family set,
        # with no correctness, order truth or relation mask in case selection.
        families={tuple(row['active']) for row in rows if len(row['active'])>=2}
        for active in families:
            matching=[index for index,row in enumerate(rows) if tuple(row['active'])==active]
            chosen.update(matching[position] for position in (0,len(matching)//2,len(matching)-1))
        for index in sorted(chosen):
            row=rows[index];item=dataset[index]
            if pair_hash(item)!=row['audio_sha256']:raise ValueError('runtime replay audio differs from frozen cache')
            dry,wet=item['dry'].numpy(),item['wet'].numpy();before=(dry.copy(),wet.copy())
            dry.flags.writeable=False;wet.flags.writeable=False
            report=runtime.infer(dry,wet,tuple(row['active']));transfer=report['transfer']
            expected=decision(row,*runtime.selection)
            dp=float(np.max(abs(np.asarray(transfer['probabilities'])-row['probabilities'][runtime.selection[0]])))
            dc=float(np.max(abs(np.asarray(report['normalized_controls'])-row['controls'])))
            maximum_probability=max(maximum_probability,dp);maximum_control=max(maximum_control,dc)
            unchanged=np.array_equal(dry,before[0]) and np.array_equal(wet,before[1])
            same=list(expected)==report['order']['selected']
            if not (same and unchanged and dp<=2e-6 and dc<=2e-6):differences.append({'domain':domain,'row':index,'expected':expected,'actual':report['order']['selected'],'probability_difference':dp,'control_difference':dc,'inputs_unchanged':unchanged})
            rows_out.append({'domain':domain,'row':index,'active':row['active'],'bank_trigger_in_frozen_audit':bool(row.get('bank_fit')),'bank_trigger_in_runtime':bool(transfer.get('bank_fit')),
                             'selected':report['order']['selected'],'expected':expected,'max_probability_difference':dp,'max_control_difference':dc,'inputs_unchanged':unchanged})
        print(json.dumps({'stage':'packaged-runtime-replay','domain':domain,'cases':len(chosen),'all_fitted_proposals':sum(bool(r.get('bank_fit')) for r in rows),'failures_so_far':len(differences)}),flush=True)
    runtime.assert_artifacts_unchanged()
    audit={'passed':not differences,'scope':'development replay, not independent final','cases':rows_out,'differences':differences,
           'calibration_sha256':file_hash(run/'blend-calibration.json'),'development_sha256':file_hash(run/'blend-development.json'),
           'source_sha256':card['source_sha256'],
           'max_probability_difference':maximum_probability,'max_control_difference':maximum_control,'elapsed_seconds':time.monotonic()-start,
           'package_head_sha256':runtime.transfer_artifact_sha256,'source_audio_modified':False,'physical_audio_devices_used':False,'new_locked_final_audio_opened':False}
    destination=run/'runtime-replay.json';destination.write_text(json.dumps(audit,indent=2)+'\n')
    if differences:raise ValueError(f'packaged runtime differs on {len(differences)} cases; see {destination}')
    card['order_admitted']=True;card['runtime_replay_verified']=True
    card['artifacts_sha256']['runtime-replay.json']=file_hash(destination)
    card_path.write_text(json.dumps(card,indent=2)+'\n')
    TransferRemixerRuntime(run).assert_artifacts_unchanged()
    print(json.dumps({'passed':True,'run':str(run),'head_sha256':runtime.transfer_artifact_sha256,'cases':len(rows_out),'max_probability_difference':maximum_probability,'max_control_difference':maximum_control}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--replay-only',action='store_true')
    p.add_argument('--refresh-unadmitted-code',action='store_true',help='Explicitly rebind candidate-only source code after a rejected runtime replay; frozen heads/cal/dev cannot change')
    p.add_argument('--revalidate-package',action='store_true',help='Explicitly withdraw admission and rerun all raw cases after integrity-only source hardening; immutable heads/cal/dev must match the existing manifest')
    args=p.parse_args();torch.set_num_threads(2)
    if args.refresh_unadmitted_code or args.revalidate_package:
        if not args.replay_only:raise ValueError('source refresh requires replay-only')
        checked_evidence(args.run);card_path=args.run/'model-card.json';card=json.loads(card_path.read_text())
        if (card['order_admitted'] or card['runtime_replay_verified']) and not args.revalidate_package:raise ValueError('cannot refresh admitted package source')
        assert_file_hashes({str(args.run/name):value for name,value in card['artifacts_sha256'].items()})
        if args.revalidate_package:
            card['prior_revalidation_manifest_sha256']=file_hash(card_path)
            card['prior_revalidation_replay_sha256']=card['artifacts_sha256'].pop('runtime-replay.json',None)
            card['order_admitted']=False;card['runtime_replay_verified']=False
        card['source_sha256']={name:file_hash(Path(__file__).parent/name) for name in PACKAGE_SOURCES}
        card.update({'analysis_sample_rate':44100,'analysis_frames':220500,'analysis_batch_size':8,
                     'numerical_environment':numerical_environment(),
                     'file_entrypoint':'TransferRemixerRuntime.infer_files / load_pair; preserve package input levels, do not call the legacy normalization default first',
                     'analysis_input_contract':'Aligned finite mono float32 analysis arrays at 44100 Hz, exactly 5 seconds. Preserve provided input levels; no domain-dependent gain. DAFx audit pairs were jointly peak-scaled upstream; synthetic/Pedalboard retained dry peak 0.22. Arbitrary input-level invariance is not claimed. Source and rendered output levels are unchanged. Fixed zero-padded batch8 reproduces the audit numerical geometry; only the real first row is used.'})
        card_path.write_text(json.dumps(card,indent=2)+'\n')
    if not args.replay_only:package(args.run)
    replay(args.run)


if __name__=='__main__':main()
