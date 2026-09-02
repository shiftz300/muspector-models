"""Remove absolute guitar spectrum and level nuisance from paired features."""
from __future__ import annotations
import numpy as np


def invariant_order_features(features, include_embedding=False):
    X=np.asarray(features,dtype=np.float32)
    if X.ndim!=2 or X.shape[1]!=2077 or not np.isfinite(X).all():raise ValueError('unexpected transfer feature geometry')
    pieces=[]
    for offset in (0,582,1164):
        for start in (offset,offset+288):
            # Only transfer ratio, coherence, transfer magnitude, and phase;
            # exclude both absolute Dry and absolute Wet spectral channels.
            pieces.append(X[:,start:start+288].reshape(-1,6,48)[:,2:].reshape(len(X),-1))
        pieces.append(X[:,offset+576:offset+582][:,2:])
    dry,wet,delta=(X[:,1746+i*27:1746+(i+1)*27] for i in range(3))
    pieces.extend((wet-dry,delta-dry))
    onset=X[:,1827:1882].reshape(-1,11,5)
    pieces.append(np.stack((onset[:,:,0]-dry[:,0,None],onset[:,:,1]-wet[:,0,None],onset[:,:,2],np.log(onset[:,:,3]+1e-8)-np.log(onset[:,:,4]+1e-8)),2).reshape(len(X),-1))
    pieces.append(X[:,1882:] if include_embedding else X[:,-3:])
    result=np.concatenate(pieces,1)
    if not np.isfinite(result).all():raise ValueError('nonfinite invariant features')
    return result
