"""Audibility metrics that reject near-identity wet-to-clean restorers."""

from __future__ import annotations

import numpy as np


MIN_CLOSURE = 0.15
MIN_CORRECTION = 0.25
MIN_DIRECTION = 0.50


def measure(wet: np.ndarray, restored: np.ndarray, clean: np.ndarray) -> dict[str, float | bool]:
    """Measure how much of the wet-to-clean displacement was actually closed."""

    wet = np.asarray(wet, dtype=np.float64)
    restored = np.asarray(restored, dtype=np.float64)
    clean = np.asarray(clean, dtype=np.float64)
    if wet.shape != restored.shape or wet.shape != clean.shape or wet.ndim != 1:
        raise ValueError("audibility inputs must be matching mono waveforms")
    if not all(np.isfinite(value).all() for value in (wet, restored, clean)):
        raise ValueError("audibility inputs must be finite")

    effect = clean - wet
    correction = restored - wet
    effect_norm = max(float(np.linalg.norm(effect)), 1.0e-12)
    correction_norm = float(np.linalg.norm(correction))
    remaining_norm = float(np.linalg.norm(clean - restored))
    correction_ratio = correction_norm / effect_norm
    closure = 1.0 - remaining_norm / effect_norm
    direction = float(np.dot(effect, correction) / max(effect_norm * correction_norm, 1.0e-12))
    passed = closure >= MIN_CLOSURE and correction_ratio >= MIN_CORRECTION and direction >= MIN_DIRECTION
    return {
        "closure": closure,
        "correction_ratio": correction_ratio,
        "correction_direction": direction,
        "audible": passed,
    }

