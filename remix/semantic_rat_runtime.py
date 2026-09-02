"""Offline-only accepted RAT semantic inverse; no audio-device dependencies."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import joblib
import numpy as np
import torch
from .drive_model import DriveControlEstimator,drive_features
from .paired_transfer_features import paired_transfer_features
from .refine_asrnn_rat_inverse import refine


class RATSemanticRuntime:
    def __init__(self,manifest:Path,*,_allow_unverified_replay:bool=False):
        self.manifest=json.loads(manifest.read_text())
        admitted = self.manifest.get('accepted') and self.manifest.get('runtime_replay_verified')
        if (not admitted and not _allow_unverified_replay) or self.manifest.get('schema')!=1:
            raise ValueError('unaccepted semantic inverse manifest')
        if not _allow_unverified_replay and self.manifest.get('runtime_replay_compute') != 'cpu':
            raise ValueError('RAT semantic runtime requires a verified CPU replay')
        root=Path(__file__).resolve().parents[1]
        self.files={name:root/value['path'] for name,value in self.manifest['artifacts'].items()}
        for name,path in self.files.items():
            if hashlib.sha256(path.read_bytes()).hexdigest()!=self.manifest['artifacts'][name]['sha256']:
                raise ValueError(f'semantic inverse provenance mismatch: {name}')
        payload=torch.load(self.files['initializer'],map_location='cpu',weights_only=True)
        self.initializer=DriveControlEstimator();self.initializer.load_state_dict(payload['state_dict']);self.initializer.eval()
        # Only this locally trained, hash-pinned package is deserialized.
        self.heads=joblib.load(self.files['semantic_heads'])

    def infer(self,dry:np.ndarray,wet:np.ndarray):
        dry=np.asarray(dry,dtype=np.float32);wet=np.asarray(wet,dtype=np.float32)
        if dry.ndim==1: dry=dry[None]
        if wet.ndim==1: wet=wet[None]
        if dry.shape!=wet.shape or dry.ndim!=2 or dry.shape[1]!=48000 or not len(dry):
            raise ValueError('RAT pilot requires matching 1-second mono 48kHz pairs')
        if not np.isfinite(dry).all() or not np.isfinite(wet).all():raise ValueError('non-finite pair')
        if np.any(np.max(np.abs(dry),axis=1)<1e-5):
            raise ValueError('Dry has no usable excitation; controls cannot be identified')
        # Copies also protect caller-owned read-only views used by torch.
        dry=dry.copy();wet=wet.copy()
        with torch.no_grad():
            image,stats=drive_features(torch.from_numpy(dry),torch.from_numpy(wet))
            # Fix numerical batch geometry to the development audit. Tiny GEMM
            # rounding changes can otherwise be amplified by inverse optimization.
            initial_rows=[]
            for start in range(0,len(dry),128):
                im,st=image[start:start+128],stats[start:start+128]
                take=len(im);indices=torch.arange(128)%take
                initial_rows.append(torch.sigmoid(self.initializer(im[indices],st[indices]))[:take])
            initial=torch.cat(initial_rows)
        count=len(dry);indices=np.arange(((count+15)//16)*16)%count
        refined=refine(list(zip(dry[indices],wet[indices])),initial[indices],self.files['renderer'],steps=60,learning_rate=.05,frames=4096,batch_size=16,pair_reader=lambda pair:(pair[0],pair[1],None),compute='cpu').numpy()[:count]
        features=np.stack([paired_transfer_features(a,b) for a,b in zip(dry,wet)])
        tone=self.heads['tone'].predict(features).clip(0,1)
        volume=self.heads['volume'].predict(features).clip(0,1)
        refined[:,1]=.5*refined[:,1]+.5*tone
        refined[:,2]=.5*refined[:,2]+.5*volume
        physical=refined.copy();physical[:,1]=1-physical[:,1]
        return {'controls':refined,'control_names':('distortion','tone','volume'),
                'physical_controls':physical,'physical_control_names':('distortion','filter','volume'),
                'control_convention':'internal tone = 1 - physical Filter',
                'upstream_identifiable':np.max(np.abs(wet),axis=1)>=.001,
                'sample_rate':48000,'audio_returned_or_modified':False}
