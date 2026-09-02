"""Fail-closed provenance helpers shared by offline semantic packaging."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from .evaluate_asrnn_effect import _quality_gate


def validate_inverse_teacher(report,checkpoint,payload):
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if report.get('device') not in ('cs3','dfz') or payload.get('device')!=report['device']:
        raise ValueError('semantic inverse teacher device mismatch')
    if digest!=report.get('checkpoint_sha256'):
        raise ValueError('semantic inverse teacher changed after control search')
    return digest


def forward_teacher_evidence(checkpoint:Path):
    path=checkpoint.parent/'model-card.json'
    result={'passed':False,'model_card':None,'model_card_sha256':None}
    if not path.is_file():return result
    card=json.loads(path.read_text());digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result.update(model_card=str(path),model_card_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if not card.get('accepted') or card.get('checkpoint_sha256')!=digest:return result
    measured=card.get('development_challenge')
    if not isinstance(measured,dict):return result
    try:result['passed']=bool(_quality_gate(measured) and card.get('checks') and all(card['checks'].values()))
    except (KeyError,TypeError,ValueError):result['passed']=False
    return result
