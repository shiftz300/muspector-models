"""Fit-only spectral initialization for named Guitar-TECHS Amp+cab+mic profiles."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .amp_cab_model2 import FIR_TAPS
from .guitar_techs_amp_data import AmpCabPairs, PROFILES


FFT_SIZE = 8192
FIT_SEED = 20260923


def estimate_profile_firs(workspace: Path, samples: int = 400) -> tuple[torch.Tensor, dict]:
    if samples < 40 or samples % len(PROFILES):
        raise ValueError("spectral initialization needs balanced fit samples")
    dataset = AmpCabPairs(workspace, "fit", samples, FFT_SIZE, FIT_SEED)
    wet_power = np.zeros((len(PROFILES), FFT_SIZE // 2 + 1), dtype=np.float64)
    clean_power = np.zeros_like(wet_power)
    counts = np.zeros(len(PROFILES), dtype=np.int64)
    window = np.hanning(FFT_SIZE).astype(np.float64)
    for index in range(len(dataset)):
        row = dataset[index]
        profile_index = int(torch.argmax(row["profile"]).item())
        start = int(row["target_start"])
        wet = row["wet"][start:start + FFT_SIZE].numpy().astype(np.float64) * window
        clean = row["clean"][start:start + FFT_SIZE].numpy().astype(np.float64) * window
        wet_spectrum = np.fft.rfft(wet)
        clean_spectrum = np.fft.rfft(clean)
        wet_power[profile_index] += np.abs(wet_spectrum) ** 2
        clean_power[profile_index] += np.abs(clean_spectrum) ** 2
        counts[profile_index] += 1
    filters = []
    rows = []
    smooth = np.ones(31, dtype=np.float64) / 31.0
    for profile_index, profile in enumerate(PROFILES):
        floor = max(float(np.max(wet_power[profile_index])) * 1.0e-6, 1.0e-12)
        gain = np.sqrt(clean_power[profile_index] / (wet_power[profile_index] + floor))
        gain = np.exp(np.convolve(np.log(np.clip(gain, 0.25, 4.0)), smooth, mode="same"))
        gain = np.clip(gain, 0.25, 4.0)
        gain[:8] = gain[8]
        gain[-8:] = gain[-9]
        impulse = np.fft.fftshift(np.fft.irfft(gain, n=FFT_SIZE))
        center = FFT_SIZE // 2
        half = FIR_TAPS // 2
        selected = impulse[center - half:center + half + 1]
        selected *= np.kaiser(FIR_TAPS, 8.6)
        filters.append(selected.astype(np.float32))
        rows.append({
            "profile": profile.id,
            "fit_windows": int(counts[profile_index]),
            "gain_min": float(np.min(gain)),
            "gain_median": float(np.median(gain)),
            "gain_max": float(np.max(gain)),
        })
    tensor = torch.from_numpy(np.stack(filters))
    digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    return tensor, {
        "schema": 1,
        "method": "fit-only-smoothed-power-ratio-zero-phase-fir",
        "fft_size": FFT_SIZE,
        "fir_taps": FIR_TAPS,
        "fit_samples": samples,
        "profiles": rows,
        "sha256_float32": digest,
        "p3_used": False,
    }
