"""Deterministic reference DSP for supervised chain reconstruction.

This is intentionally small and inspectable. It defines the first training
contract; production preview DSP must later prove numerical/ perceptual parity.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.signal import fftconvolve, lfilter

from .quality import checked_audio, checked_sample_rate, validate_render
from .spec import ChainSpec, Delay, Drive, Reverb


Renderer = Literal["reference", "alternate", "stress", "challenge", "final_challenge"]
# The first two domains define the original contract. ``stress`` is an
# augmentation domain added after the first OOD audit; ``challenge`` remains
# excluded from training and calibration.
RENDERERS: tuple[Renderer, ...] = ("reference", "alternate")


def _mono(audio: np.ndarray) -> np.ndarray:
    value = checked_audio(audio)
    if value.ndim != 1:
        raise ValueError(f"expected one channel, got {value.shape}")
    return value


def render_drive(
    audio: np.ndarray, effect: Drive, sample_rate: int, renderer: Renderer = "reference"
) -> np.ndarray:
    effect.validate()
    value = _mono(audio)
    gain = 10.0 ** (effect.gain_db / 20.0)
    if renderer == "reference":
        driven = np.tanh(value * gain) / max(math.tanh(gain), 1.0e-6)
        cutoff = 900.0 * (12_000.0 / 900.0) ** effect.tone
    elif renderer == "alternate":
        driven = np.arctan(value * gain * 1.7) / max(math.atan(gain * 1.7), 1.0e-6)
        cutoff = 700.0 * (14_000.0 / 700.0) ** effect.tone
    elif renderer == "stress":
        scaled = value * gain * 1.35
        driven = scaled / (1.0 + np.abs(scaled))
        normalization = gain * 1.35 / (1.0 + gain * 1.35)
        driven /= max(normalization, 1.0e-6)
        cutoff = 1_100.0 * (10_500.0 / 1_100.0) ** effect.tone
    elif renderer == "challenge":
        scaled = value * gain * 1.1
        driven = np.tanh(scaled + 0.08 * scaled * np.abs(scaled))
        normalization = math.tanh(gain * 1.1 + 0.08 * (gain * 1.1) ** 2)
        driven /= max(normalization, 1.0e-6)
        cutoff = 1_400.0 * (9_000.0 / 1_400.0) ** effect.tone
    elif renderer == "final_challenge":
        scaled = value * gain * 1.22
        driven = np.arcsinh(scaled * 1.35) / max(math.asinh(gain * 1.22 * 1.35), 1.0e-6)
        cutoff = 820.0 * (11_200.0 / 820.0) ** effect.tone
    else:
        raise ValueError(f"unknown renderer: {renderer}")
    coefficient = math.exp(-2.0 * math.pi * cutoff / sample_rate)
    shaped = lfilter([1.0 - coefficient], [1.0, -coefficient], driven)
    return (shaped * 10.0 ** (effect.level_db / 20.0)).astype(np.float32)


def render_delay(
    audio: np.ndarray, effect: Delay, sample_rate: int, renderer: Renderer = "reference"
) -> np.ndarray:
    effect.validate()
    value = _mono(audio)
    offset = max(1, round(effect.time_ms * sample_rate / 1_000.0))
    if renderer == "reference":
        repeat = value
    elif renderer == "alternate":
        # A darker repeat path approximates a different delay implementation
        # while retaining the same physical time/feedback/mix contract.
        cutoff = 5_500.0
        coefficient = math.exp(-2.0 * math.pi * cutoff / sample_rate)
        repeat = lfilter([1.0 - coefficient], [1.0, -coefficient], value).astype(np.float32)
    elif renderer == "stress":
        cutoff = 3_800.0
        coefficient = math.exp(-2.0 * math.pi * cutoff / sample_rate)
        repeat = lfilter(
            [(1.0 - coefficient) ** 2],
            [1.0, -2.0 * coefficient, coefficient * coefficient],
            value,
        ).astype(np.float32)
    elif renderer == "challenge":
        cutoff = 2_900.0
        coefficient = math.exp(-2.0 * math.pi * cutoff / sample_rate)
        repeat = lfilter(
            [(1.0 - coefficient) ** 2],
            [1.0, -2.0 * coefficient, coefficient * coefficient],
            value,
        ).astype(np.float32)
    elif renderer == "final_challenge":
        cutoff = 4_600.0
        coefficient = math.exp(-2.0 * math.pi * cutoff / sample_rate)
        repeat = lfilter([1.0 - coefficient], [1.0, -coefficient], value).astype(np.float32)
    else:
        raise ValueError(f"unknown renderer: {renderer}")
    wet = np.zeros_like(value)
    amplitude = 1.0
    position = offset
    while position < len(value) and amplitude >= 1.0e-3:
        wet[position:] += repeat[:-position] * amplitude
        amplitude *= effect.feedback
        position += offset
    return (value * (1.0 - effect.mix) + wet * effect.mix).astype(np.float32)


def render_reverb(
    audio: np.ndarray, effect: Reverb, sample_rate: int, renderer: Renderer = "reference"
) -> np.ndarray:
    effect.validate()
    value = _mono(audio)
    length = max(64, min(len(value), round(effect.decay_s * sample_rate)))
    if renderer == "reference":
        seed = 0x4D555350
        cutoff_high, cutoff_low = 12_000.0, 700.0
    elif renderer == "alternate":
        seed = 0x45435452
        cutoff_high, cutoff_low = 10_000.0, 500.0
    elif renderer == "stress":
        seed = 0x53545253
        cutoff_high, cutoff_low = 8_500.0, 900.0
    elif renderer == "challenge":
        seed = 0x43484C47
        cutoff_high, cutoff_low = 7_200.0, 1_200.0
    elif renderer == "final_challenge":
        seed = 0x464E4C31
        cutoff_high, cutoff_low = 9_300.0, 620.0
    else:
        raise ValueError(f"unknown renderer: {renderer}")
    rng = np.random.default_rng(seed)
    time = np.arange(length, dtype=np.float64) / sample_rate
    envelope = 10.0 ** (-3.0 * time / max(effect.decay_s, 1.0e-3))
    impulse = rng.standard_normal(length) * envelope
    cutoff = cutoff_high * (cutoff_low / cutoff_high) ** effect.damping
    coefficient = math.exp(-2.0 * math.pi * cutoff / sample_rate)
    impulse = lfilter([1.0 - coefficient], [1.0, -coefficient], impulse)
    impulse[: min(32, length)] = 0.0
    impulse[min(32, length - 1)] += 1.0
    if renderer == "alternate":
        # Give the second domain a different early-reflection geometry.
        for delay_ms, amplitude in ((11.0, 0.65), (29.0, -0.42), (47.0, 0.31)):
            index = min(length - 1, round(delay_ms * sample_rate / 1_000.0))
            impulse[index] += amplitude
    elif renderer == "stress":
        for delay_ms, amplitude in ((7.0, -0.58), (19.0, 0.47), (61.0, -0.28)):
            index = min(length - 1, round(delay_ms * sample_rate / 1_000.0))
            impulse[index] += amplitude
    elif renderer == "challenge":
        for delay_ms, amplitude in ((5.0, 0.71), (23.0, -0.51), (73.0, 0.33)):
            index = min(length - 1, round(delay_ms * sample_rate / 1_000.0))
            impulse[index] += amplitude
    elif renderer == "final_challenge":
        for delay_ms, amplitude in ((13.0, -0.62), (37.0, 0.45), (89.0, -0.26)):
            index = min(length - 1, round(delay_ms * sample_rate / 1_000.0))
            impulse[index] += amplitude
    norm = float(np.sqrt(np.sum(impulse * impulse)))
    wet = fftconvolve(value, impulse / max(norm, 1.0e-8), mode="full")[: len(value)]
    return (value * (1.0 - effect.mix) + wet * effect.mix).astype(np.float32)


def render_chain(
    audio: np.ndarray,
    spec: ChainSpec,
    sample_rate: int,
    renderer: Renderer = "reference",
) -> np.ndarray:
    spec.validate()
    checked_sample_rate(sample_rate)
    source = checked_audio(audio)
    if source.ndim == 2:
        channel_axis = 1 if source.shape[1] <= 2 else 0
        channels = np.moveaxis(source, channel_axis, 0)
        rendered = np.stack(
            [render_chain(channel, spec, sample_rate, renderer) for channel in channels]
        )
        result = np.moveaxis(rendered, 0, channel_axis)
        validate_render(source, result)
        return result
    value = source
    for effect in spec.effects:
        if isinstance(effect, Drive):
            value = render_drive(value, effect, sample_rate, renderer)
        elif isinstance(effect, Delay):
            value = render_delay(value, effect, sample_rate, renderer)
        elif isinstance(effect, Reverb):
            value = render_reverb(value, effect, sample_rate, renderer)
        else:  # pragma: no cover - guarded by ChainSpec typing and validation.
            raise TypeError(f"unsupported effect: {effect!r}")
    validate_render(source, value)
    return value
