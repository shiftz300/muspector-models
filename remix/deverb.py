"""Regularized analytic inverse for the capture-fitted linear Reverb stage."""

from __future__ import annotations

import math

import numpy as np
from scipy.fft import next_fast_len

from .forward_reverb import ReverbDeviceProfile
from .spec import Reverb


def decode(control: np.ndarray) -> Reverb:
    value = np.asarray(control, dtype=np.float32)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Reverb controls must be finite [3]")
    if bool((value < 0.0).any()) or bool((value > 1.0).any()):
        raise ValueError("Reverb controls must be normalized to [0,1]")
    return Reverb(0.2 * 40.0 ** float(value[0]), float(value[1]), float(value[2]) * 0.7)


def restore(
    profile: ReverbDeviceProfile,
    wet: np.ndarray,
    control: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Invert one finite wet buffer without changing its length or level policy."""

    source = np.asarray(wet, dtype=np.float32)
    if source.ndim != 1 or not np.isfinite(source).all() or not len(source):
        raise ValueError("Reverb inverse expects finite non-empty mono audio")
    if not math.isfinite(regularization) or not 1.0e-8 <= regularization <= 1.0:
        raise ValueError("regularization must be within [1e-8,1]")
    effect = decode(control)
    impulse = profile.impulse(effect.decay_s, effect.damping, len(source)).astype(np.float64)
    transfer = impulse * effect.mix
    transfer[0] += 1.0 - effect.mix
    size = next_fast_len(len(source) * 2)
    response = np.fft.rfft(transfer, size)
    observed = np.fft.rfft(source, size)
    restored = np.fft.irfft(
        observed * np.conj(response) / (np.square(np.abs(response)) + regularization), size
    )[: len(source)]
    result = np.asarray(restored, dtype=np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Reverb inverse produced non-finite audio")
    return result
