"""Bounded known-profile room-response shortening for Product4 Reverb."""

from __future__ import annotations

import math

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import fftconvolve

from .ambience2 import AmbienceAbstention
from .ambience3 import _transfer
from .foundation_data import RATE


DEFAULT_LOOKAHEAD_FRAMES = 4_096
DEFAULT_CAUSAL_FRAMES = 12_288


def _bounded_response(
    transfer: np.ndarray,
    *,
    early_ms: float,
    target_rt60_ms: float,
    maximum_gain: float,
    low_band_strength: float,
    high_band_strength: float,
    low_band_end_hz: float,
    high_band_start_hz: float,
    lookahead_frames: int,
    causal_frames: int,
) -> tuple[np.ndarray, dict]:
    """Design a finite-lookahead response-shortening kernel.

    The desired response preserves the measured direct/early response and
    smoothly accelerates only its late decay.  A Tikhonov floor limits inverse
    gain, and frequency-dependent partial application protects the high band.
    """
    if not 5.0 <= early_ms <= 80.0 or not 20.0 <= target_rt60_ms <= 500.0:
        raise ValueError("invalid room-response shaping time constants")
    if not 1.0 < maximum_gain <= 8.0:
        raise ValueError("invalid room-response shaping gain ceiling")
    if not 0.0 <= high_band_strength <= low_band_strength <= 0.60:
        raise ValueError("invalid room-response shaping strengths")
    if not 500.0 <= low_band_end_hz < high_band_start_hz <= 8_000.0:
        raise ValueError("invalid room-response shaping transition band")
    if lookahead_frames < 0 or causal_frames < 1 or lookahead_frames + causal_frames < 4_096:
        raise ValueError("room-response shaping kernel is too short")

    early_frames = round(early_ms * RATE / 1_000.0)
    target_rt60_frames = target_rt60_ms * RATE / 1_000.0
    desired = np.asarray(transfer, dtype=np.float64).copy()
    positions = np.arange(len(desired), dtype=np.float64) - early_frames
    desired[early_frames:] *= 10.0 ** (
        -3.0 * np.maximum(positions[early_frames:], 0.0) / target_rt60_frames
    )

    kernel_frames = lookahead_frames + causal_frames
    fft_size = next_fast_len(len(transfer) + kernel_frames - 1)
    response = rfft(transfer, fft_size)
    target = rfft(desired, fft_size)
    power = np.square(np.abs(response))

    # Find the smallest regularization that respects the frequency-gain cap.
    lower = 0.0
    upper = max(float(np.max(power)), 1.0e-12)
    for _ in range(16):
        inverse = np.conj(response) * target / (power + upper)
        if float(np.max(np.abs(inverse))) <= maximum_gain:
            break
        upper *= 10.0
    else:
        raise AmbienceAbstention("profile shortening could not bound inverse gain")
    for _ in range(48):
        regularization = (lower + upper) * 0.5
        inverse = np.conj(response) * target / (power + regularization)
        if float(np.max(np.abs(inverse))) > maximum_gain:
            lower = regularization
        else:
            upper = regularization
    inverse = np.conj(response) * target / (power + upper)

    frequencies = np.fft.rfftfreq(fft_size, 1.0 / RATE)
    strength = np.where(
        frequencies <= low_band_end_hz,
        low_band_strength,
        np.where(
            frequencies >= high_band_start_hz,
            high_band_strength,
            low_band_strength
            + (high_band_strength - low_band_strength)
            * (frequencies - low_band_end_hz)
            / (high_band_start_hz - low_band_end_hz),
        ),
    )
    shortened = 1.0 + strength * (inverse - 1.0)
    circular_kernel = irfft(shortened, fft_size)
    kernel = np.concatenate((
        circular_kernel[-lookahead_frames:] if lookahead_frames else np.empty(0),
        circular_kernel[:causal_frames],
    ))
    if not np.isfinite(kernel).all():
        raise AmbienceAbstention("profile shortening produced a non-finite kernel")
    return kernel, {
        "implementation": "bounded-partial-room-response-shortening",
        "early_ms": early_ms,
        "target_rt60_ms": target_rt60_ms,
        "maximum_gain_contract": maximum_gain,
        "designed_maximum_frequency_gain": float(np.max(np.abs(shortened))),
        "low_band_strength": low_band_strength,
        "high_band_strength": high_band_strength,
        "transition_band_hz": [low_band_end_hz, high_band_start_hz],
        "lookahead_frames": lookahead_frames,
        "causal_frames": causal_frames,
        "lookahead_ms": lookahead_frames * 1_000.0 / RATE,
        "causal_state_ms": causal_frames * 1_000.0 / RATE,
        "profile_required": True,
        "clean_input": False,
        "chain_order_input": False,
        "graph_order_input": False,
        "neighbor_effect_input": False,
    }


def profile_shortening_inverse(
    wet: np.ndarray,
    impulse: np.ndarray,
    controls: dict,
    *,
    early_ms: float,
    target_rt60_ms: float,
    maximum_gain: float,
    low_band_strength: float,
    high_band_strength: float,
    low_band_end_hz: float = 2_000.0,
    high_band_start_hz: float = 4_000.0,
    lookahead_frames: int = DEFAULT_LOOKAHEAD_FRAMES,
    causal_frames: int = DEFAULT_CAUSAL_FRAMES,
) -> tuple[np.ndarray, dict]:
    """Shorten one known current-effect profile without graph/order context."""
    wet = np.asarray(wet, dtype=np.float64)
    impulse = np.asarray(impulse, dtype=np.float64)
    if wet.ndim != 1 or impulse.ndim != 1 or not len(wet) or not len(impulse):
        raise ValueError("profile shortening expects nonempty mono arrays")
    if not np.isfinite(wet).all() or not np.isfinite(impulse).all():
        raise ValueError("profile shortening expects finite input")
    transfer = _transfer(impulse, controls)
    kernel, report = _bounded_response(
        transfer,
        early_ms=early_ms,
        target_rt60_ms=target_rt60_ms,
        maximum_gain=maximum_gain,
        low_band_strength=low_band_strength,
        high_band_strength=high_band_strength,
        low_band_end_hz=low_band_end_hz,
        high_band_start_hz=high_band_start_hz,
        lookahead_frames=lookahead_frames,
        causal_frames=causal_frames,
    )
    restored = fftconvolve(wet, kernel, mode="full")[
        lookahead_frames : lookahead_frames + len(wet)
    ]
    peak = float(np.max(np.abs(restored)))
    if not math.isfinite(peak) or peak > 1.05:
        raise AmbienceAbstention("profile shortening output is non-finite or clips")
    return restored.astype(np.float32), {**report, "restored_peak": peak}
