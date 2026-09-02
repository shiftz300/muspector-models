"""Perceptual restoration gates for blur, residual effect, and new artifacts.

These paired metrics deliberately complement waveform ESR.  A candidate must
move several audible properties toward the original clean recording; improving
only a scale-aligned waveform average is not sufficient.
"""

from __future__ import annotations

import numpy as np


RATE = 48_000
EPSILON = 1.0e-10


def _audio(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or len(result) < 4096 or not np.isfinite(result).all():
        raise ValueError("restoration quality expects finite mono audio of at least 4096 frames")
    return result


def _frames(value: np.ndarray, size: int, hop: int) -> np.ndarray:
    count = 1 + (len(value) - size) // hop
    if count <= 0:
        raise ValueError("audio is shorter than the quality-analysis frame")
    shape = (count, size)
    strides = (value.strides[0] * hop, value.strides[0])
    return np.lib.stride_tricks.as_strided(value, shape=shape, strides=strides).copy()


def _spectra(value: np.ndarray, size: int) -> np.ndarray:
    framed = _frames(value, size, size // 4)
    return np.abs(np.fft.rfft(framed * np.hanning(size), axis=1))


def _spectral_error(value: np.ndarray, clean: np.ndarray) -> float:
    errors = []
    for size in (256, 1024, 4096):
        left, right = _spectra(value, size), _spectra(clean, size)
        scale = np.maximum(np.mean(right, axis=1, keepdims=True), 1.0e-6)
        errors.append(float(np.mean(np.abs(np.log1p(left / scale) - np.log1p(right / scale)))))
    return float(np.mean(errors))


def _band_envelope(value: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    size = 1024
    spectrum = _spectra(value, size)
    frequencies = np.fft.rfftfreq(size, 1.0 / RATE)
    selected = (frequencies >= low_hz) & (frequencies < high_hz)
    return np.sqrt(np.mean(np.square(spectrum[:, selected]), axis=1) + EPSILON)


def _relative_error(value: np.ndarray, clean: np.ndarray) -> float:
    denominator = float(np.mean(np.abs(clean))) + 1.0e-6
    return float(np.mean(np.abs(value - clean)) / denominator)


def _transient_envelope(value: np.ndarray) -> np.ndarray:
    framed = _frames(value, 240, 120)
    envelope = np.sqrt(np.mean(np.square(framed), axis=1) + EPSILON)
    # Positive attacks carry most of the pick-definition evidence.  Log energy
    # prevents one loud frame from hiding many smeared attacks.
    return np.maximum(np.diff(np.log(envelope + 1.0e-6)), 0.0)


def _crest_error(value: np.ndarray, clean: np.ndarray) -> float:
    def crest(audio: np.ndarray) -> np.ndarray:
        framed = _frames(audio, 1024, 256)
        rms = np.sqrt(np.mean(np.square(framed), axis=1) + EPSILON)
        return np.max(np.abs(framed), axis=1) / rms

    return _relative_error(crest(value), crest(clean))


def _improvement(baseline: float, restored: float) -> float:
    # A one-sided ratio explodes when the Wet baseline is already nearly zero
    # (common for crest/transient error on delay or compressor pairs).  Use a
    # symmetric relative improvement so every score stays in [-1, 1] while
    # preserving the ordering of clearly improved/degraded candidates.
    denominator = max(baseline + restored, EPSILON)
    return float((baseline - restored) / denominator)


def measure(wet: np.ndarray, restored: np.ndarray, clean: np.ndarray) -> dict:
    clean, wet, restored = _audio(clean), _audio(wet), _audio(restored)
    length = min(len(clean), len(wet), len(restored))
    clean, wet, restored = clean[:length], wet[:length], restored[:length]

    wet_spectral = _spectral_error(wet, clean)
    restored_spectral = _spectral_error(restored, clean)
    clean_high = _band_envelope(clean, 3_000.0, 12_000.0)
    wet_high = _band_envelope(wet, 3_000.0, 12_000.0)
    restored_high = _band_envelope(restored, 3_000.0, 12_000.0)
    wet_high_error = _relative_error(wet_high, clean_high)
    restored_high_error = _relative_error(restored_high, clean_high)
    clean_attack = _transient_envelope(clean)
    wet_attack = _transient_envelope(wet)
    restored_attack = _transient_envelope(restored)
    wet_attack_error = _relative_error(wet_attack, clean_attack)
    restored_attack_error = _relative_error(restored_attack, clean_attack)
    wet_crest_error = _crest_error(wet, clean)
    restored_crest_error = _crest_error(restored, clean)

    spectral = _improvement(wet_spectral, restored_spectral)
    high_band = _improvement(wet_high_error, restored_high_error)
    transient = _improvement(wet_attack_error, restored_attack_error)
    dynamics = _improvement(wet_crest_error, restored_crest_error)
    peak = float(np.max(np.abs(restored)))
    clipped = bool(np.mean(np.abs(restored) >= 0.999) > np.mean(np.abs(wet) >= 0.999) + 1.0e-4)
    correction = float(np.sqrt(np.mean(np.square(restored - wet))))
    wet_distance = float(np.sqrt(np.mean(np.square(wet - clean))))
    meaningful = correction >= 0.05 * max(wet_distance, 1.0e-6)

    gates = {
        "meaningful_correction": meaningful,
        "multires_spectrum": spectral >= 0.15,
        "high_band_detail": high_band >= 0.10,
        "transient_definition": transient >= 0.10,
        "dynamics": dynamics >= 0.0,
        "no_new_clipping": not clipped and peak <= 1.05,
    }
    return {
        "spectral_improvement": spectral,
        "high_band_improvement": high_band,
        "transient_improvement": transient,
        "dynamics_improvement": dynamics,
        "correction_to_wet_distance": correction / max(wet_distance, 1.0e-6),
        "restored_peak": peak,
        "added_clipping": clipped,
        "gates": gates,
        "passed": all(gates.values()),
    }


def summarize(wet: list[np.ndarray], restored: list[np.ndarray], clean: list[np.ndarray]) -> dict:
    rows = [measure(left, middle, right) for left, middle, right in zip(wet, restored, clean, strict=True)]
    names = ("spectral_improvement", "high_band_improvement", "transient_improvement", "dynamics_improvement")
    medians = {f"median_{name}": float(np.median([row[name] for row in rows])) for name in names}
    pass_fraction = float(np.mean([row["passed"] for row in rows]))
    added_clipping_fraction = float(np.mean([row["added_clipping"] for row in rows]))
    gates = {
        "spectrum": medians["median_spectral_improvement"] >= 0.15,
        "high_band": medians["median_high_band_improvement"] >= 0.10,
        "transients": medians["median_transient_improvement"] >= 0.10,
        "dynamics": medians["median_dynamics_improvement"] >= 0.0,
        "pass_fraction": pass_fraction >= 0.50,
        "no_new_clipping": added_clipping_fraction <= 0.01,
    }
    return {"examples": len(rows), **medians, "pass_fraction": pass_fraction, "added_clipping_fraction": added_clipping_fraction, "gates": gates, "accepted": all(gates.values())}
