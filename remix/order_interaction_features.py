"""Nonlinear transfer kernels and level-dependent spectra for chain order."""
from __future__ import annotations
import numpy as np
from scipy.signal import stft
from scipy.fft import rfft,irfft,next_fast_len


def order_interaction_features(dry,wet,sample_rate=44100):
    x=np.asarray(dry,dtype=np.float64);y=np.asarray(wet,dtype=np.float64)
    if x.ndim!=1 or x.shape!=y.shape or len(x)<1024 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('invalid order feature pair')
    pieces=[]
    _,_,X=stft(x,fs=sample_rate,nperseg=1024,noverlap=768,boundary=None,padded=False)
    _,_,Y=stft(y,fs=sample_rate,nperseg=1024,noverlap=768,boundary=None,padded=False)
    frequencies=np.fft.rfftfreq(1024,1/sample_rate);edges=np.geomspace(40,min(16000,sample_rate*.45),13)
    bands=[]
    for spectrum in (X,Y):
        power=np.abs(spectrum)**2
        bands.append(np.stack([np.log(power[(frequencies>=lo)&(frequencies<hi)].mean(0)+1e-12) for lo,hi in zip(edges[:-1],edges[1:])]))
    a,b=bands;level=np.log((np.abs(X)**2).sum(0)+1e-12)
    quantiles=np.quantile(level,(0,.25,.5,.75,1))
    for lo,hi in zip(quantiles[:-1],quantiles[1:]):
        mask=(level>=lo)&(level<=hi)
        if not mask.any():mask[:]=True
        for value in (a,b,b-a):pieces.extend((value[:,mask].mean(1),value[:,mask].std(1)))
    for lag in (0,1,4,16,64,128):
        if lag>=a.shape[1]:pieces.append(np.zeros(12));continue
        left=a[:,:a.shape[1]-lag] if lag else a
        right=b[:,lag:] if lag else b
        left=left-left.mean(1,keepdims=True);right=right-right.mean(1,keepdims=True)
        pieces.append((left*right).mean(1)/(left.std(1)*right.std(1)+1e-8))
    # The relative shapes of kernels driven by linear and saturated Dry expose
    # whether a nonlinear stage precedes or follows the echo/reverb response.
    scale=max(float(np.sqrt(np.mean(x*x))),1e-6);normalized=x/scale
    nfft=next_fast_len(2*len(x));target=rfft(y,nfft)
    stops=np.unique(np.rint(np.geomspace(1,min(sample_rate*2,len(x)),33)).astype(int))
    for transformed in (normalized,np.tanh(normalized*.3),np.tanh(normalized),np.tanh(normalized*3)):
        source=rfft(transformed,nfft);power=np.abs(source)**2
        kernel=irfft(target*source.conj()/(power+max(float(power.mean())*.01,1e-12)),nfft)
        positive=kernel[:len(x)];negative=kernel[-len(x):]
        energy=max(float(np.sum(kernel*kernel)),1e-18)
        features=[]
        for start,stop in zip(np.r_[0,stops[:-1]],stops):
            features.append(np.log(np.sum(positive[start:stop]**2)/energy+1e-12))
            features.append(np.log(np.sum(negative[-stop:len(negative)-start if start else None]**2)/energy+1e-12))
        pieces.append(np.array(features))
        pieces.append(np.array([np.log(energy),float(np.sum(positive*positive)/energy),float(np.argmax(np.abs(positive)))/sample_rate]))
    result=np.concatenate(pieces).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('non-finite interaction features')
    return result
