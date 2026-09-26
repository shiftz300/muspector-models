"""Frequency-dependent known-profile response shortening for Product4 Reverb."""

from __future__ import annotations

import math

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.ndimage import median_filter
from scipy.signal import fftconvolve, istft, stft

from .ambience2 import AmbienceAbstention
from .ambience3 import _transfer
from .foundation_data import RATE


DEFAULT_N_FFT = 1_024
DEFAULT_HOP = 256
DEFAULT_LOOKAHEAD_FRAMES = 4_096
DEFAULT_CAUSAL_FRAMES = 12_288
FREQUENCY_SHAPING_CANDIDATES = tuple(
    {
        "id": (
            f"early{early:g}-ratio{ratio:.2f}-strength{strength:.2f}"
            f"-high{high_ratio:.2f}"
        ),
        "early_ms": early,
        "decay_ratio": ratio,
        "correction_strength": strength,
        "high_frequency_ratio": high_ratio,
        "maximum_gain": 4.0,
    }
    for early, ratio, strength, high_ratio in (
        (10.0, 0.05, 0.25, 0.00),
        (10.0, 0.05, 0.35, 0.00),
        (10.0, 0.10, 0.30, 0.00),
        (10.0, 0.15, 0.35, 0.00),
        (25.0, 0.05, 0.30, 0.00),
        (25.0, 0.10, 0.35, 0.00),
        (10.0, 0.05, 0.30, 0.10),
        (25.0, 0.05, 0.30, 0.10),
    )
)


def _frequency_decay_seconds(
    impulse: np.ndarray,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a robust T60 independently for every STFT frequency bin."""
    frequencies, times, spectrum = stft(
        impulse,
        fs=RATE,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    power = np.square(np.abs(spectrum))
    energy = np.cumsum(power[:, ::-1], axis=1)[:, ::-1]
    reference = np.maximum(energy[:, :1], 1.0e-18)
    decay_db = 10.0 * np.log10(np.maximum(energy / reference, 1.0e-12))

    estimates = np.empty(len(frequencies), dtype=np.float64)
    fallback = max(len(impulse) / RATE * 0.60, 0.10)
    for index, curve in enumerate(decay_db):
        above_five = np.flatnonzero(curve <= -5.0)
        above_thirty_five = np.flatnonzero(curve <= -35.0)
        if len(above_five) and len(above_thirty_five):
            start = int(above_five[0])
            end = int(above_thirty_five[0])
            estimate = 2.0 * max(float(times[end] - times[start]), 0.0)
        else:
            estimate = fallback
        estimates[index] = estimate
    estimates = median_filter(estimates, size=9, mode="nearest")
    estimates = np.clip(estimates, 0.08, 1.50)
    return frequencies, times, spectrum, estimates


def frequency_faded_transfer(
    impulse: np.ndarray,
    controls: dict,
    *,
    decay_ratio: float,
    early_ms: float,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
) -> tuple[np.ndarray, dict]:
    """Build a desired transfer whose late T60 is shortened per frequency bin."""
    impulse = np.asarray(impulse, dtype=np.float64)
    if impulse.ndim != 1 or not len(impulse) or not np.isfinite(impulse).all():
        raise ValueError("frequency shaping expects a finite mono impulse")
    if not 0.03 <= decay_ratio <= 0.85 or not 5.0 <= early_ms <= 80.0:
        raise ValueError("invalid frequency shaping decay target")
    if n_fft < 512 or hop < 64 or hop > n_fft:
        raise ValueError("invalid frequency shaping spectral geometry")

    frequencies, times, spectrum, original_t60 = _frequency_decay_seconds(
        impulse, n_fft=n_fft, hop=hop
    )
    late_seconds = np.maximum(times - early_ms / 1_000.0, 0.0)
    # If the original amplitude envelope has T60=T, multiplying by a fade with
    # T60=T*r/(1-r) yields a combined T60 of approximately T*r.
    fade_t60 = np.maximum(original_t60 * decay_ratio / (1.0 - decay_ratio), 0.02)
    fade = 10.0 ** (-3.0 * late_seconds[None, :] / fade_t60[:, None])
    faded_spectrum = spectrum * fade
    _, faded = istft(
        faded_spectrum,
        fs=RATE,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        input_onesided=True,
        boundary=True,
    )
    desired_impulse = np.zeros_like(impulse)
    copied = min(len(desired_impulse), len(faded))
    desired_impulse[:copied] = faded[:copied]

    # Preserve direct sound and early reflections exactly, then crossfade into
    # the frequency-dependent late response to avoid a time-domain seam.
    early_frames = min(round(early_ms * RATE / 1_000.0), len(impulse))
    transition = min(round(5.0 * RATE / 1_000.0), len(impulse) - early_frames)
    desired_impulse[:early_frames] = impulse[:early_frames]
    if transition:
        blend = np.linspace(0.0, 1.0, transition, endpoint=False)
        stop = early_frames + transition
        desired_impulse[early_frames:stop] = (
            impulse[early_frames:stop] * (1.0 - blend)
            + desired_impulse[early_frames:stop] * blend
        )

    mix = float(controls["mix"])
    room_gain = 10.0 ** (float(controls["room_gain_db"]) / 20.0)
    desired_transfer = desired_impulse * (mix * room_gain)
    desired_transfer[0] += 1.0 - mix
    return desired_transfer, {
        "frequency_bins": len(frequencies),
        "estimated_t60_seconds": {
            "minimum": float(np.min(original_t60)),
            "median": float(np.median(original_t60)),
            "maximum": float(np.max(original_t60)),
        },
        "target_decay_ratio": decay_ratio,
        "early_ms": early_ms,
    }


def _bounded_frequency_kernel(
    transfer: np.ndarray,
    desired: np.ndarray,
    *,
    correction_strength: float,
    high_frequency_ratio: float,
    maximum_gain: float,
    lookahead_frames: int,
    causal_frames: int,
) -> tuple[np.ndarray, dict]:
    if not 0.0 < correction_strength <= 0.60:
        raise ValueError("invalid frequency shaping correction strength")
    if not 0.0 <= high_frequency_ratio <= 1.0:
        raise ValueError("invalid frequency shaping high-frequency ratio")
    if not 1.0 < maximum_gain <= 8.0:
        raise ValueError("invalid frequency shaping gain ceiling")
    if lookahead_frames < 0 or causal_frames < 1 or lookahead_frames + causal_frames < 4_096:
        raise ValueError("frequency shaping kernel is too short")
    kernel_frames = lookahead_frames + causal_frames
    fft_size = next_fast_len(max(len(transfer), len(desired)) + kernel_frames - 1)
    response = rfft(transfer, fft_size)
    target = rfft(desired, fft_size)
    power = np.square(np.abs(response))

    lower = 0.0
    upper = max(float(np.max(power)), 1.0e-12)
    for _ in range(16):
        inverse = np.conj(response) * target / (power + upper)
        if float(np.max(np.abs(inverse))) <= maximum_gain:
            break
        upper *= 10.0
    else:
        raise AmbienceAbstention("frequency shaping could not bound inverse gain")
    for _ in range(48):
        regularization = (lower + upper) * 0.5
        inverse = np.conj(response) * target / (power + regularization)
        if float(np.max(np.abs(inverse))) > maximum_gain:
            lower = regularization
        else:
            upper = regularization
    inverse = np.conj(response) * target / (power + upper)
    frequencies = np.fft.rfftfreq(fft_size, 1.0 / RATE)
    high_strength = correction_strength * high_frequency_ratio
    strength = np.where(
        frequencies <= 2_000.0,
        correction_strength,
        np.where(
            frequencies >= 4_000.0,
            high_strength,
            correction_strength
            + (high_strength - correction_strength)
            * (frequencies - 2_000.0) / 2_000.0,
        ),
    )
    shortened = 1.0 + strength * (inverse - 1.0)
    circular = irfft(shortened, fft_size)
    kernel = np.concatenate((
        circular[-lookahead_frames:] if lookahead_frames else np.empty(0),
        circular[:causal_frames],
    ))
    if not np.isfinite(kernel).all():
        raise AmbienceAbstention("frequency shaping produced a non-finite kernel")
    return kernel, {
        "designed_maximum_frequency_gain": float(np.max(np.abs(shortened))),
        "regularization": float(upper),
        "high_frequency_ratio": high_frequency_ratio,
        "transition_band_hz": [2_000.0, 4_000.0],
    }


def frequency_shortening_kernel(
    impulse: np.ndarray,
    controls: dict,
    *,
    decay_ratio: float,
    early_ms: float,
    correction_strength: float,
    high_frequency_ratio: float = 0.0,
    maximum_gain: float = 4.0,
    lookahead_frames: int = DEFAULT_LOOKAHEAD_FRAMES,
    causal_frames: int = DEFAULT_CAUSAL_FRAMES,
) -> tuple[np.ndarray, dict]:
    """Design one cacheable current-profile shortening kernel."""
    impulse = np.asarray(impulse, dtype=np.float64)
    transfer = _transfer(impulse, controls)
    desired, decay_report = frequency_faded_transfer(
        impulse, controls, decay_ratio=decay_ratio, early_ms=early_ms
    )
    kernel, design_report = _bounded_frequency_kernel(
        transfer,
        desired,
        correction_strength=correction_strength,
        high_frequency_ratio=high_frequency_ratio,
        maximum_gain=maximum_gain,
        lookahead_frames=lookahead_frames,
        causal_frames=causal_frames,
    )
    return kernel, {
        "implementation": "bounded-frequency-bin-faded-profile-shortening",
        **decay_report,
        **design_report,
        "correction_strength": correction_strength,
        "maximum_gain_contract": maximum_gain,
        "lookahead_frames": lookahead_frames,
        "causal_frames": causal_frames,
        "profile_required": True,
        "clean_input": False,
        "chain_order_input": False,
        "graph_order_input": False,
        "neighbor_effect_input": False,
    }


def partitioned_convolve_full(
    signal: np.ndarray,
    kernel: np.ndarray,
    *,
    block_frames: int = 8_192,
) -> np.ndarray:
    """Exact finite overlap-add convolution with bounded input partitions."""
    signal = np.asarray(signal, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if signal.ndim != 1 or kernel.ndim != 1 or not len(signal) or not len(kernel):
        raise ValueError("partitioned convolution expects non-empty mono inputs")
    if not np.isfinite(signal).all() or not np.isfinite(kernel).all():
        raise ValueError("partitioned convolution inputs must be finite")
    if block_frames < 256:
        raise ValueError("partitioned convolution block is too short")
    fft_size = next_fast_len(block_frames + len(kernel) - 1)
    kernel_spectrum = rfft(kernel, fft_size)
    output = np.zeros(len(signal) + len(kernel) - 1, dtype=np.float64)
    for start in range(0, len(signal), block_frames):
        block = signal[start : start + block_frames]
        convolved = irfft(rfft(block, fft_size) * kernel_spectrum, fft_size)
        count = len(block) + len(kernel) - 1
        output[start : start + count] += convolved[:count]
    return output


def partitioned_frequency_profile_candidate_bank(
    wet: np.ndarray,
    kernels: np.ndarray,
    *,
    lookahead_frames: int = DEFAULT_LOOKAHEAD_FRAMES,
    block_frames: int = 8_192,
) -> np.ndarray:
    """Apply a cacheable profile bank through bounded overlap-add partitions."""
    wet = np.asarray(wet, dtype=np.float64)
    kernels = np.asarray(kernels, dtype=np.float64)
    if kernels.ndim != 2 or kernels.shape[0] != len(FREQUENCY_SHAPING_CANDIDATES):
        raise ValueError("frequency profile kernel bank geometry differs")
    restored = []
    for kernel in kernels:
        full = partitioned_convolve_full(wet, kernel, block_frames=block_frames)
        candidate = full[lookahead_frames : lookahead_frames + len(wet)]
        peak = float(np.max(np.abs(candidate)))
        if not math.isfinite(peak) or peak > 1.05:
            raise AmbienceAbstention("partitioned frequency candidate clips")
        restored.append(candidate.astype(np.float32))
    return np.stack(restored)


def frequency_profile_candidate_kernels(
    impulse: np.ndarray,
    controls: dict,
) -> tuple[np.ndarray, list[dict]]:
    """Prepare all fixed candidates once when the current profile is loaded."""
    kernels = []
    reports = []
    for candidate in FREQUENCY_SHAPING_CANDIDATES:
        parameters = {key: value for key, value in candidate.items() if key != "id"}
        kernel, report = frequency_shortening_kernel(impulse, controls, **parameters)
        kernels.append(kernel)
        reports.append({"candidate": candidate["id"], **report})
    return np.stack(kernels), reports


def profile_frequency_shortening_inverse(
    wet: np.ndarray,
    impulse: np.ndarray,
    controls: dict,
    *,
    decay_ratio: float,
    early_ms: float,
    correction_strength: float,
    high_frequency_ratio: float = 0.0,
    maximum_gain: float = 4.0,
    lookahead_frames: int = DEFAULT_LOOKAHEAD_FRAMES,
    causal_frames: int = DEFAULT_CAUSAL_FRAMES,
) -> tuple[np.ndarray, dict]:
    """Shorten one known current-effect profile without any chain context."""
    wet = np.asarray(wet, dtype=np.float64)
    impulse = np.asarray(impulse, dtype=np.float64)
    if wet.ndim != 1 or not len(wet) or not np.isfinite(wet).all():
        raise ValueError("profile frequency shortening expects finite mono Wet")
    kernel, report = frequency_shortening_kernel(
        impulse,
        controls,
        decay_ratio=decay_ratio,
        early_ms=early_ms,
        correction_strength=correction_strength,
        high_frequency_ratio=high_frequency_ratio,
        maximum_gain=maximum_gain,
        lookahead_frames=lookahead_frames,
        causal_frames=causal_frames,
    )
    restored = fftconvolve(wet, kernel, mode="full")[
        lookahead_frames : lookahead_frames + len(wet)
    ]
    peak = float(np.max(np.abs(restored)))
    if not math.isfinite(peak) or peak > 1.05:
        raise AmbienceAbstention("profile frequency shortening output is non-finite or clips")
    return restored.astype(np.float32), {
        **report,
        "restored_peak": peak,
    }
