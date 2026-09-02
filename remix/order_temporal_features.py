"""Read-only modulation and time-varying transfer cues for ambiguous orders.

Linear time-invariant Delay and Reverb commute. These features look for the
nonstationarity / nonlinear departures that can carry order evidence, rather
than treating a metadata label as a guarantee that the order is identifiable.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import stft


def temporal_order_features(dry, wet, sample_rate=44100):
    x, y = (np.asarray(value, dtype=np.float64) for value in (dry, wet))
    if x.ndim != 1 or x.shape != y.shape or len(x) < 4096:
        raise ValueError('equal mono pairs of at least 4096 samples required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite audio')
    pieces = []
    for fft in (1024, 4096):
        _, _, X = stft(x, fs=sample_rate, nperseg=fft, noverlap=fft*3//4,
                       boundary=None, padded=False)
        _, _, Y = stft(y, fs=sample_rate, nperseg=fft, noverlap=fft*3//4,
                       boundary=None, padded=False)
        p, q = np.abs(X)**2, np.abs(Y)**2
        floor = max(float(p.mean()) * 1e-3, 1e-14)
        H = (Y * X.conj()).mean(1) / (p.mean(1) + floor)
        residual = Y - H[:, None] * X
        frequencies = np.fft.rfftfreq(fft, 1/sample_rate)
        edges = np.geomspace(40, min(16000, sample_rate*.45), 13)
        for lo, hi in zip(edges[:-1], edges[1:]):
            bins = (frequencies >= lo) & (frequencies < hi)
            if not bins.any(): bins[np.argmin(abs(frequencies-np.sqrt(lo*hi)))] = True
            a, b = p[bins].mean(0), q[bins].mean(0)
            z = (Y[bins] * X[bins].conj()).mean(0)
            r = np.abs(residual[bins])**2
            coherence = abs(z)**2 / (a*b + floor*floor)
            transfer = z / (a + floor)
            values = np.stack((np.log(b+floor)-np.log(a+floor), coherence,
                               np.log(r.mean(0)+floor)-np.log(b+floor),
                               transfer.real, transfer.imag))
            # Complex transfer is made scale-free; no analysis gain is ever
            # written back into the source or the chain renderer.
            values[3:] /= max(float(np.sqrt(np.mean(abs(transfer)**2))), 1e-8)
            pieces.append(np.quantile(values, (.1,.25,.5,.75,.9), axis=1).ravel())
            centered = values - values.mean(1, keepdims=True)
            variance = np.mean(centered*centered, axis=1)
            pieces.append(np.log(variance+1e-8))
            for lag in (1,2,4,8,16):
                correlation = (np.mean(centered[:, :-lag]*centered[:, lag:],1)/(variance+1e-8)
                               if lag < centered.shape[1] else np.zeros(5))
                pieces.append(correlation)
            # Repeated-echo modulation and diffuse reverb can differ even when
            # their average frequency responses are similar.
            modulation = abs(np.fft.rfft(centered, axis=1))**2
            modfreq = np.fft.rfftfreq(centered.shape[1], fft/4/sample_rate)
            total = modulation.sum(1)+1e-8
            for lower, upper in ((0,1),(1,2),(2,4),(4,8),(8,16),(16,64)):
                pieces.append(np.log(modulation[:,(modfreq>=lower)&(modfreq<upper)].sum(1)/total+1e-8))
    result = np.concatenate(pieces).astype(np.float32)
    if not np.isfinite(result).all(): raise ValueError('nonfinite temporal features')
    return result
