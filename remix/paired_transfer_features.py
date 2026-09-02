"""Read-only paired spectral transfer, coherence and onset-envelope features.

These analysis features never alter the source arrays and accept no filenames,
effect labels or knob targets. Their crop follows Dry activity only.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import stft


def paired_transfer_features(dry: np.ndarray, wet: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    dry, wet = np.asarray(dry, dtype=np.float32), np.asarray(wet, dtype=np.float32)
    if dry.ndim != 1 or dry.shape != wet.shape or not len(dry) or sample_rate <= 0:
        raise ValueError("paired features require equal nonempty mono arrays and a positive rate")
    if not np.isfinite(dry).all() or not np.isfinite(wet).all():
        raise ValueError("paired features require finite audio")
    active = np.flatnonzero(np.abs(dry) > max(1e-4, float(np.max(np.abs(dry))) * 0.005))
    onset = int(active[0]) if len(active) else 0
    stop = min(len(dry), int(active[-1])+1+sample_rate//10) if len(active) else len(dry)
    start = max(0, onset-sample_rate//100)
    x, y = dry[start:stop], wet[start:stop]
    minimum_frames = max(4096, int(sample_rate * .064))
    if len(x) < minimum_frames:
        x, y = np.pad(x, (0, minimum_frames-len(x))), np.pad(y, (0, minimum_frames-len(y)))
    pieces = []
    for fft in (256, 1024, 4096):
        _, _, X = stft(x, fs=sample_rate, nperseg=fft, noverlap=fft*3//4, boundary=None, padded=False)
        _, _, Y = stft(y, fs=sample_rate, nperseg=fft, noverlap=fft*3//4, boundary=None, padded=False)
        p, q = np.abs(X)**2, np.abs(Y)**2
        floor = max(float(p.max()) * 1e-8, 1e-14)
        pmean, qmean = p.mean(1), q.mean(1)
        cross = (Y * X.conj()).mean(1)
        coherent = np.abs(cross)**2 / (pmean*qmean+floor**2)
        transfer = cross / (pmean+floor)
        logratio = np.log(qmean+floor)-np.log(pmean+floor)
        values = np.stack((np.log(pmean+floor), np.log(qmean+floor), logratio,
                           coherent, np.log(np.abs(transfer)+1e-7), np.angle(transfer)))
        edges = np.geomspace(30, min(20000, sample_rate*0.48), 49)
        frequencies = np.fft.rfftfreq(fft, 1/sample_rate)
        pooled = []
        for low, high in zip(edges[:-1], edges[1:]):
            bins = np.flatnonzero((frequencies >= low) & (frequencies < high))
            if not len(bins):
                bins = np.array([int(np.argmin(np.abs(frequencies-np.sqrt(low*high))))])
            pooled.append(values[:, bins].mean(1))
        features = np.stack(pooled, 1)
        # Centered spectra remove level nuisance; their means remain separate features.
        means = features.mean(1, keepdims=True)
        pieces.extend((features.ravel(), (features-means).ravel(), means.ravel()))
    for audio in (x, y, y-x):
        absolute = np.abs(audio)
        rms = np.sqrt(np.mean(audio**2)+1e-14)
        pieces.append(np.array([np.log(rms), float(audio.mean()), np.mean(np.signbit(audio[1:]) != np.signbit(audio[:-1]))]))
        pieces.append(np.log(np.quantile(absolute, (0.1,0.25,0.5,0.75,0.9,0.95,0.99,1))+1e-8))
        for milliseconds in (1,4,16,64):
            size = max(1, int(sample_rate*milliseconds/1000))
            frames = len(audio)//size
            envelope = np.sqrt(np.mean(audio[:frames*size].reshape(frames,size)**2,axis=1)+1e-14)
            pieces.append(np.log(np.quantile(envelope,(0.1,0.5,0.9,1))+1e-8))
    # Identical absolute-time windows in Dry and Wet preserve compressor attack cues.
    boundaries = np.array((0,1,2,4,8,16,32,64,128,256,512,1024))*sample_rate//1000
    for left,right in zip(boundaries[:-1],boundaries[1:]):
        first,last = min(onset+int(left),len(dry)-1), min(onset+int(right),len(dry))
        a,b = dry[first:last],wet[first:last]
        dr,wr = np.sqrt(np.mean(a*a)+1e-14),np.sqrt(np.mean(b*b)+1e-14)
        pieces.append(np.array((np.log(dr),np.log(wr),np.log(wr/dr),np.max(np.abs(b)),np.max(np.abs(a)))))
    result = np.concatenate(pieces).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("paired analysis produced non-finite features")
    return result
