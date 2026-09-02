"""Audio-array-only CS-3/DFZ semantic inverse backed by acceptance evidence."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from .stable_effect import load_stable_effect
from .search_effect_controls import search_pair
from .search_dfz_controls import search as search_dfz


class SemanticEffectRuntime:
    def __init__(self,acceptance:Path,checkpoint:Path,compute='cpu',*,_allow_unverified_replay=False):
        report=json.loads(acceptance.read_text())
        if not report.get('accepted') or report.get('device') not in ('cs3','dfz'):
            raise ValueError('semantic inverse is not accepted')
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=report['renderer_sha256']:
            raise ValueError('semantic inverse renderer provenance mismatch')
        evidence=acceptance.parent/'metrics.json'
        if hashlib.sha256(evidence.read_bytes()).hexdigest()!=report['inverse_evidence_sha256']:
            raise ValueError('semantic inverse evaluation evidence changed')
        if not _allow_unverified_replay:
            card=json.loads((acceptance.parent/'model-card.json').read_text())
            if not card.get('accepted') or not card.get('runtime_replay_verified'):
                raise ValueError('semantic inverse runtime replay is not accepted')
            if card.get('acceptance_sha256')!=hashlib.sha256(acceptance.read_bytes()).hexdigest() or card.get('renderer',{}).get('sha256')!=report['renderer_sha256']:
                raise ValueError('semantic inverse replay provenance changed')
            if card.get('runtime_replay_compute')!=compute:
                raise ValueError('semantic inverse compute backend was not replay-verified')
        self.device_key=report['device'];self.model,payload=load_stable_effect(checkpoint)
        if payload['device']!=self.device_key:raise ValueError('effect type mismatch')
        self.model=self.model.to(torch.device(compute));self.compute=compute

    def infer(self,dry,wet):
        dry=np.asarray(dry,dtype=np.float32);wet=np.asarray(wet,dtype=np.float32)
        if dry.ndim!=1 or dry.shape!=wet.shape or len(dry)!=144000:
            raise ValueError('pilot requires aligned three-second mono 48kHz pairs')
        if not np.isfinite(dry).all() or not np.isfinite(wet).all():raise ValueError('non-finite pair')
        if float(np.abs(dry).max())<1e-5:raise ValueError('no usable Dry excitation')
        if float(np.abs(wet).max())<.001:raise ValueError('Wet too quiet for the validated control-identification domain')
        dry=dry.copy();wet=wet.copy()
        if self.device_key=='cs3':
            if self.compute=='cpu':estimate,_,_=search_pair(self.model,dry,wet,1,.05,16384)
            else:
                from .search_effect_controls_accelerated import search_batch
                estimate=search_batch(self.model,[(dry,wet)])[0][0]
            names=('attack',)
        else:
            estimate=search_dfz(self.model,[(dry,wet)])[0];names=('blend','filter')
        return {'control_names':names,'controls':estimate,'source_audio_modified':False,'sample_rate':48000,'scope':'paired semantic inverse; forward audio fidelity admitted separately'}
