"""Hammerstein/Wiener diagnostic fits for nonlinear-stage placement.

All normalization, downsampling and ridge gain fitting are analysis-only. This
module neither renders a user output nor modifies either input audio array.
"""
from __future__ import annotations
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
from scipy.signal import resample_poly


def cascade_order_features(dry,wet,sample_rate=44100):
    x,y=(np.asarray(v,dtype=np.float64) for v in (dry,wet))
    if x.ndim!=1 or x.shape!=y.shape or len(x)<4096 or sample_rate<=0:
        raise ValueError('equal mono pairs with at least 4096 frames required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite pair')
    # A small polyphase analysis copy retains the distortion/echo evidence
    # below Nyquist without ever touching the rendering sample-rate contract.
    factor=max(1,int(sample_rate//11025));x=resample_poly(x,1,factor);y=resample_poly(y,1,factor)
    rate=sample_rate/factor;n=len(x);nfft=next_fast_len(2*n)
    x=x/max(float(np.sqrt(np.mean(x*x))),1e-8)
    y=y/max(float(np.sqrt(np.mean(y*y))),1e-8)
    target=rfft(y,nfft);source=rfft(x,nfft)
    peak=max(float(np.max(abs(y))),1e-8)
    bases=[('pre',rfft(np.tanh(x*g)/g,nfft),target) for g in (.3,1.,3.,10.)]
    bases += [('post',source,rfft(np.arctanh(np.clip(y/peak*g,-.999999,.999999))/g,nfft)) for g in (.3,.8,.99)]
    bases.insert(0,('linear',source,target));pieces=[]
    for _,S,T in bases:
        power=abs(S)**2
        for ridge in (.001,.03):
            H=T*S.conj()/(power+max(float(power.mean())*ridge,1e-10))
            impulse=irfft(H,nfft)
            fullenergy=max(float(np.sum(impulse*impulse)),1e-12)
            pieces.append(np.asarray([np.sum(impulse[:n]**2)/fullenergy,np.argmax(abs(impulse[:n]))/rate]))
            wanted=irfft(T,nfft)[:n]
            energy=max(float(np.mean(wanted*wanted)),1e-10)
            for duration in (.08,.35,1.2):
                end=min(n,max(1,int(duration*rate)))
                causal=np.zeros(nfft);causal[:end]=impulse[:end]
                estimate=irfft(rfft(causal,nfft)*S,nfft)[:n]
                error=estimate-wanted
                values=[np.log(np.mean(error*error)/energy+1e-10),
                        np.mean(error*wanted)/energy,
                        np.max(abs(error))/np.sqrt(energy),
                        np.mean(error**4)/(np.mean(error*error)**2+1e-10)]
                for ids in np.array_split(np.arange(n),4):
                    values.append(np.log(np.mean(error[ids]**2)/(np.mean(wanted[ids]**2)+1e-10)+1e-10))
                pieces.append(np.asarray(values))
    result=np.concatenate(pieces).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('nonfinite cascade diagnostics')
    return result
