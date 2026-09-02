"""Small deterministic wet-only feature vector for effect-control proposals."""

from __future__ import annotations

import numpy as np


def extract(audio: np.ndarray, bands: int = 48) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 1 or not np.isfinite(value).all() or len(value) < 512:
        raise ValueError("wet-only features require at least 512 finite mono frames")
    frames = np.lib.stride_tricks.sliding_window_view(value, 512)[::128]
    window = np.hanning(512).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(frames * window, axis=1))
    log = np.log1p(magnitude)
    frequency = []
    for indices in np.array_split(np.arange(log.shape[1]), bands):
        band = log[:, indices].mean(1)
        frequency.extend((float(band.mean()), float(band.std())))
    envelope = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1.0e-12)
    temporal = [
        float(envelope.mean()), float(envelope.std()), float(np.mean(np.abs(np.diff(envelope)))),
        *np.quantile(envelope, (0.05, 0.25, 0.5, 0.75, 0.95)).tolist(),
    ]
    centered = envelope - envelope.mean(); denominator = max(float(np.dot(centered, centered)), 1.0e-12)
    temporal.extend(float(np.dot(centered[:-lag], centered[lag:]) / denominator) for lag in (1, 2, 4, 8, 16, 32) if len(centered) > lag)
    absolute = np.abs(value)
    waveform = [
        float(np.sqrt(np.mean(np.square(value, dtype=np.float64)))),
        float(absolute.max()),
        *np.quantile(absolute, (0.5, 0.75, 0.9, 0.99)).tolist(),
        float(np.mean(np.signbit(value[1:]) != np.signbit(value[:-1]))),
    ]
    result = np.asarray([*frequency, *temporal, *waveform], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("wet-only feature extraction produced non-finite values")
    return result


def context(audio: np.ndarray) -> np.ndarray:
    """Add long-memory statistics for effects whose tails outlive one STFT frame.

    The base vector remains unchanged for existing packages.  This extension is
    intentionally deterministic and cheap enough to run on a normal CPU.
    """

    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 1 or not np.isfinite(value).all() or len(value) < 32_768:
        raise ValueError("context features require at least 32768 finite mono frames")
    base = extract(value)
    frames = np.lib.stride_tricks.sliding_window_view(value, 1024)[::256]
    envelope = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1.0e-12)
    centered = envelope - envelope.mean()
    denominator = max(float(np.dot(centered, centered)), 1.0e-12)
    envelope_lags = [
        float(np.dot(centered[:-lag], centered[lag:]) / denominator)
        for lag in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        if len(centered) > lag
    ]
    waveform_denominator = max(float(np.dot(value, value)), 1.0e-12)
    waveform_lags = [
        float(np.dot(value[:-lag], value[lag:]) / waveform_denominator)
        for lag in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
        if len(value) > lag
    ]
    log_envelope = np.log(envelope + 1.0e-7)
    modulation = np.abs(np.fft.rfft(log_envelope - log_envelope.mean()))
    modulation_bands = [
        float(part.mean()) for part in np.array_split(modulation[1:], 24) if len(part)
    ]
    result = np.asarray(
        [*base, *envelope_lags, *waveform_lags, *modulation_bands], dtype=np.float32
    )
    if not np.isfinite(result).all():
        raise ValueError("context feature extraction produced non-finite values")
    return result
