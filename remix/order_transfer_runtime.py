"""Offline order candidate wrapper; existing audio renderer remains unchanged."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import joblib,numpy as np,scipy,sklearn,torch
from .inference import RemixerRuntime,BUNDLE,load_analysis_pair
from .quality import analysis_pair,checked_audio
from .train_order_transfer import transfer_batch
from .order_interaction_features import order_interaction_features
from .order_invariant_features import invariant_order_features
from .order_temporal_features import temporal_order_features
from .order_cascade_features import cascade_order_features
from .order_bank_fit import CONTROL_RADIUS,CLEAN_FRACTIONS,needs_fit,raw_proposal,fit_topology_options,apply_bank_fit
from .order_search import rank_topology_blends,decode_controls
from .evaluate_order_transfer import decision
from .spec import ChainSpec

PACKAGE_SELECTION=['cascade-drive-physics-hist-0.5-0.5',1.0,0.0]
PACKAGE_SOURCES=(
    'order_transfer_runtime.py','order_bank_fit.py','order_search.py',
    'order_temporal_features.py','order_cascade_features.py','order_invariant_features.py',
    'order_interaction_features.py','paired_transfer_features.py','train_order_transfer.py',
    'evaluate_order_transfer.py','inference.py','model.py','drive_model.py','delay_model.py',
    'reverb_model.py','physics.py','quality.py','render.py','spec.py',
    'package_order_bank.py','evaluate_order_bank_fit.py','evaluate_order_search.py',
    'data.py','pedalboard_data.py','pedalboard_renderer.py','order_identifiability.py','train.py')


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json_hashed(path):
    payload=Path(path).read_bytes()
    return json.loads(payload),hashlib.sha256(payload).hexdigest()


def numerical_environment():
    return {'torch':str(torch.__version__),'numpy':np.__version__,'scipy':scipy.__version__,
            'sklearn':sklearn.__version__,'joblib':joblib.__version__,'torch_cpu_threads':torch.get_num_threads()}


def assert_file_hashes(expected):
    """Read every dependency again; a cached hash is not an integrity check."""
    for filename,digest in expected.items():
        if file_hash(filename)!=digest:raise ValueError(f'order artifact changed: {filename}')


class TransferRemixerRuntime(RemixerRuntime):
    def __init__(self,run:Path,*,allow_candidate=False):
        run=Path(run).resolve();self.run=run;self.packaged=(run/'model-card.json').exists()
        if not self.packaged and not allow_candidate:raise ValueError('unpackaged order heads require explicit candidate mode')
        calibration,calibration_hash=read_json_hashed(run/'blend-calibration.json')
        development,development_hash=read_json_hashed(run/'blend-development.json')
        if not allow_candidate and not (calibration['passed'] and development['passed']):
            raise ValueError('order transfer is not admitted')
        if calibration['selected']!=development['selected']:
            raise ValueError('calibration and development selections differ')
        self.selection=calibration['selected'];name=self.selection[0]
        self._artifact_hashes={str(BUNDLE):file_hash(BUNDLE),str(run/'blend-calibration.json'):calibration_hash,str(run/'blend-development.json'):development_hash}
        root=Path(__file__).resolve().parents[1]/'remix/runs/order-transfer-phase11'
        if self.packaged:
            card_path=run/'model-card.json';card,card_hash=read_json_hashed(card_path);self.card=card
            if card.get('schema')!=1 or self.selection!=PACKAGE_SELECTION:raise ValueError('unsupported packaged order architecture')
            if not allow_candidate and not (card.get('order_admitted') and card.get('runtime_replay_verified')):
                raise ValueError('packaged order has not passed raw-audio replay')
            if card['selection']!=self.selection or card['bank_menu']!='bounded-knobs':raise ValueError('packaged selection differs')
            if card['analysis_frontend']!='preserve-input-levels-float32-copy':raise ValueError('packaged analysis frontend differs')
            if card['analysis_batch_size']!=8:raise ValueError('packaged analysis batch geometry differs')
            if card['numerical_environment']!=numerical_environment() or torch.get_num_threads()!=2:raise ValueError('packaged numerical environment or CPU thread count differs')
            if card['control_radius']!=CONTROL_RADIUS or card['parallel_clean_fractions']!=list(CLEAN_FRACTIONS):raise ValueError('packaged bank bounds differ')
            if calibration['bank_menu']!=card['bank_menu'] or development['bank_menu']!=card['bank_menu']:raise ValueError('bank menu is not frozen')
            if development['calibration_sha256']!=calibration_hash:raise ValueError('development calibration binding differs')
            if card['artifacts_sha256']['blend-calibration.json']!=calibration_hash or card['artifacts_sha256']['blend-development.json']!=development_hash:raise ValueError('parsed calibration/development bytes differ from manifest')
            if set(card['source_sha256'])!=set(PACKAGE_SOURCES):raise ValueError('incomplete source dependency manifest')
            self._artifact_hashes.update({str(Path(__file__).parent/key):value for key,value in card['source_sha256'].items()})
            self._artifact_hashes.update({str(run/key):value for key,value in card['artifacts_sha256'].items()})
            if not {'order-heads.joblib','blend-calibration.json','blend-development.json'}<=set(card['artifacts_sha256']):raise ValueError('incomplete artifact dependency manifest')
            if card.get('runtime_replay_verified') and 'runtime-replay.json' not in card['artifacts_sha256']:raise ValueError('admitted package lacks bound runtime replay')
            self._artifact_hashes[str(BUNDLE)]=card['canonical_bundle_sha256']
            self._artifact_hashes[str(card_path)]=card_hash
            assert_file_hashes(self._artifact_hashes)
            if card.get('runtime_replay_verified'):
                audit,audit_hash=read_json_hashed(run/'runtime-replay.json')
                if audit_hash!=card['artifacts_sha256']['runtime-replay.json']:raise ValueError('parsed runtime replay bytes differ from manifest')
                if not audit.get('passed') or audit['package_head_sha256']!=card['artifacts_sha256']['order-heads.joblib'] or audit['calibration_sha256']!=card['artifacts_sha256']['blend-calibration.json'] or audit['development_sha256']!=card['artifacts_sha256']['blend-development.json'] or audit['source_sha256']!=card['source_sha256']:raise ValueError('runtime replay does not bind the admitted model and evidence')
            artifact=run/'order-heads.joblib';self.heads=joblib.load(artifact)
            if self.heads['selection']!=self.selection or self.heads['schema']!=1:raise ValueError('packaged classifier metadata differs')
            if [m.n_features_in_ for m in self.heads['interaction']]!=[2705,2705] or [m.n_features_in_ for m in self.heads['physics']]!=[416,416] or self.heads['temporal'].n_features_in_!=3497:raise ValueError('packaged feature geometry differs')
            self.bank_menu=card['bank_menu']
        elif name and name.startswith('interaction-'):
            artifact=run/'interaction-heads.joblib';self.heads=joblib.load(artifact)['candidates'][name]
        elif name=='hist31':
            artifact=root/'hist31.joblib';self.heads=joblib.load(artifact)
        elif name:
            artifact=root/'candidates.joblib';self.heads=joblib.load(artifact)['candidates'][name]
        else:raise ValueError('no transfer head selected')
        self.transfer_artifact_sha256=file_hash(artifact)
        self._artifact_hashes.setdefault(str(artifact),self.transfer_artifact_sha256)
        self.transfer_gate_passed=calibration['passed'] and development['passed']
        super().__init__(BUNDLE,target=torch.device('cpu'))
        if self.packaged and (self.card['analysis_sample_rate']!=self.sample_rate or self.card['analysis_frames']!=self.sample_rate*self.analysis_seconds):raise ValueError('package analysis geometry differs from canonical bundle')
        self.assert_artifacts_unchanged()

    def assert_artifacts_unchanged(self):
        if self.packaged and self.card['numerical_environment']!=numerical_environment():raise ValueError('order numerical environment changed')
        assert_file_hashes(self._artifact_hashes)

    def load_pair(self,clean_path,wet_path,offset_seconds=0.):
        """File entrypoint with this package's fixed analysis-level policy."""
        return load_analysis_pair(Path(clean_path),Path(wet_path),self.sample_rate,self.analysis_seconds,offset_seconds,preserve_levels=self.packaged)

    def infer_files(self,clean_path,wet_path,active,offset_seconds=0.):
        clean,wet,source=self.load_pair(clean_path,wet_path,offset_seconds)
        report=self.infer(clean,wet,active);report['source']=source
        self.assert_artifacts_unchanged()
        return report

    def _prepare_analysis_pair(self,dry,wet):
        if not self.packaged:return super()._prepare_analysis_pair(dry,wet)
        # The frozen five-domain training/audit uses the input analysis level:
        # DAFx pairs arrive jointly scaled, while generated DSP pairs preserve
        # their calibrated dry drive level. No domain identifier or guessed
        # scale is consulted here; every input follows this identical rule.
        clean=checked_audio(dry,name='dry analysis copy');affected=checked_audio(wet,name='wet analysis copy')
        if clean.shape!=affected.shape:raise ValueError('paired audio geometry differs')
        return clean,affected

    def _model_analysis_batch(self,dry,wet):
        clean,affected=super()._model_analysis_batch(dry,wet)
        if not self.packaged:return clean,affected
        # Match the frozen audit GEMM/convolution geometry, including tiny
        # numerical differences that can cross an order-search guard. All
        # modules are in eval mode; there are no cross-example statistics.
        x=torch.zeros((8,clean.shape[1]),dtype=clean.dtype,device=clean.device)
        y=torch.zeros_like(x);x[0]=clean[0];y[0]=affected[0]
        return x,y

    def _packaged_probabilities(self,features,clean,affected):
        interaction=np.concatenate((features,order_interaction_features(clean,affected)[None]),1)
        physics=cascade_order_features(clean,affected)[None]
        temporal=np.concatenate((invariant_order_features(features,True),temporal_order_features(clean,affected)[None]),1)
        values=[.5*self.heads['interaction'][i].predict_proba(interaction)[0,1]+.5*self.heads['physics'][i].predict_proba(physics)[0,1] for i in (0,1)]
        return [float(v) for v in values]+[float(self.heads['temporal'].predict_proba(temporal)[0,1])]

    @torch.no_grad()
    def infer(self,dry,wet,active):
        if self.packaged and self.card['numerical_environment']!=numerical_environment():raise ValueError('order numerical environment changed')
        report=super().infer(dry,wet,tuple(sorted(active)))
        if len(active)<2:return report
        clean,affected=self._prepare_analysis_pair(dry,wet)
        x,y=self._model_analysis_batch(clean,affected)
        batch={'dry':x,'wet':y}
        features=transfer_batch(self.main,batch)[:1];logits=features[0,-3:]
        name,weight,margin=self.selection
        if self.packaged:
            probabilities=self._packaged_probabilities(features,clean,affected)
        elif name.startswith('interaction-'):
            features=np.concatenate((features,order_interaction_features(clean,affected)[None]),1)
            probabilities=[float(head.predict_proba(features)[0,1]) for head in self.heads]
        else:probabilities=[float(head.predict_proba(features)[0,1]) for head in self.heads]
        errors=rank_topology_blends(clean,affected,tuple(sorted(active)),np.array(report['normalized_controls']))
        row={'baseline':report['order']['selected'],'active':tuple(sorted(active)),'logits':logits.tolist(),'probabilities':{name:probabilities},'errors':errors,'baseline_error':report['order']['selected_reconstruction_error']}
        if self.packaged and needs_fit(row,self.selection):
            row['bank_fit']=fit_topology_options(clean,affected,raw_proposal(row,self.selection),report['normalized_controls'],menus=(self.bank_menu,))
            row=apply_bank_fit(row,self.selection,self.bank_menu)
        selected=decision(row,name,weight,margin)
        old=report['order']['selected'];report['chain']=ChainSpec(decode_controls(selected,np.array(report['normalized_controls']))).document()
        report['decision']='transfer-order' if list(selected)!=old else report['decision']
        report['order']['selected']=list(selected)
        report['order']['selected_reconstruction_error']=next(item['reconstruction_error'] for item in report['order']['ranked'] if tuple(item['topology'])==selected)
        report['transfer']={'selection':self.selection,'head_sha256':self.transfer_artifact_sha256,'development_gate_passed':self.transfer_gate_passed,'original_selected':old,
                            'probabilities':probabilities,'search_bank':row['errors'],'diagnostic_only':True,'final_audio_renderer_unchanged':True,
                            'bank_fit':row.get('bank_fit'),'actual_chain_controls_unchanged':True,'diagnostic_gain_applied_to_output':False,
                            'analysis_frontend':'preserve-input-levels-float32-copy' if self.packaged else 'joint-peak-analysis-normalization',
                            'selected_diagnostic_error':next(v['error'] for v in row['errors'] if tuple(v['topology'])==selected)}
        return report
