"""Per-example Product4 Reverb safety: effective restoration or exact bypass."""

from __future__ import annotations

import numpy as np
from scipy.signal import stft

from .foundation_data import RATE
from .quality2 import measure as measure_universal
from .reverb_quality import measure as measure_tail


HARD_BYPASS_MAX_ABSOLUTE_ERROR = 1.0e-6
ACTIVE_NONREGRESSION_MINIMUM = -0.02
ACTIVE_EFFECTIVE_REDUCTION_MINIMUM = 0.05
MINIMUM_EFFECTIVE_ELIGIBLE_COVERAGE = 0.50


def _active_spectral_measure(
    wet: np.ndarray,
    restored: np.ndarray,
    clean: np.ndarray,
) -> dict:
    values = [np.asarray(value, dtype=np.float64) for value in (wet, restored, clean)]
    length = min(map(len, values))
    wet, restored, clean = (value[:length] for value in values)
    spectra = []
    for value in (wet, restored, clean):
        _, _, spectrum = stft(
            value,
            fs=RATE,
            window="hann",
            nperseg=1_024,
            noverlap=768,
            boundary=None,
            padded=False,
        )
        spectra.append(np.abs(spectrum))
    wet_spectrum, restored_spectrum, clean_spectrum = spectra
    clean_level = np.sqrt(np.mean(np.square(clean_spectrum), axis=0))
    active_floor = max(1.0e-5, 0.05 * float(np.max(clean_level)))
    active = clean_level >= active_floor
    if int(active.sum()) < 4:
        return {"eligible": False, "active_frames": int(active.sum())}
    magnitude_floor = max(1.0e-8, 1.0e-4 * float(np.max(clean_spectrum[:, active])))

    def error(value: np.ndarray) -> float:
        delta = np.log(value[:, active] + magnitude_floor) - np.log(
            clean_spectrum[:, active] + magnitude_floor
        )
        return float(np.sqrt(np.mean(np.square(delta))))

    baseline = error(wet_spectrum)
    candidate = error(restored_spectrum)
    measurable = baseline >= 1.0e-3
    reduction = 1.0 - candidate / max(baseline, 1.0e-12)
    return {
        "eligible": bool(measurable),
        "active_frames": int(active.sum()),
        "baseline_log_spectral_error": baseline,
        "restored_log_spectral_error": candidate,
        "reduction": float(reduction),
        "nonregression": bool(reduction >= ACTIVE_NONREGRESSION_MINIMUM),
        "effective": bool(reduction >= ACTIVE_EFFECTIVE_REDUCTION_MINIMUM),
    }


def measure(wet: np.ndarray, restored: np.ndarray, clean: np.ndarray) -> dict:
    wet = np.asarray(wet, dtype=np.float32)
    restored = np.asarray(restored, dtype=np.float32)
    clean = np.asarray(clean, dtype=np.float32)
    length = min(len(wet), len(restored), len(clean))
    wet, restored, clean = wet[:length], restored[:length], clean[:length]
    bypass_error = float(np.max(np.abs(restored - wet)))
    hard_bypass = bypass_error <= HARD_BYPASS_MAX_ABSOLUTE_ERROR
    tail = measure_tail(wet, restored, clean)
    active = _active_spectral_measure(wet, restored, clean)
    universal = measure_universal("ambience", wet, restored, clean)
    evidence = bool(tail["eligible"] and active["eligible"])
    effective = bool(
        evidence
        and tail["passed"]
        and active["effective"]
        and universal["passed"]
    )
    if hard_bypass:
        decision = "hard-bypass"
        passed = True
    elif effective:
        decision = "effective-restoration"
        passed = True
    else:
        decision = "unsafe-or-ineffective-processing"
        passed = False
    return {
        "passed": passed,
        "decision": decision,
        "evidence_measurable": evidence,
        "hard_bypass": hard_bypass,
        "hard_bypass_max_absolute_error": bypass_error,
        "tail": tail,
        "active_foreground": active,
        "universal": universal,
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("reverb quality2 summary requires examples")
    measurable = [row for row in rows if row["evidence_measurable"]]
    effective = [row for row in measurable if row["decision"] == "effective-restoration"]
    unmeasurable = [row for row in rows if not row["evidence_measurable"]]
    safe_fraction = float(np.mean([row["passed"] for row in rows]))
    effective_coverage = len(effective) / max(len(measurable), 1)
    unmeasurable_bypass_fraction = (
        1.0 if not unmeasurable
        else float(np.mean([row["hard_bypass"] for row in unmeasurable]))
    )
    gates = {
        "every_example_safe": safe_fraction == 1.0,
        "effective_eligible_coverage": (
            bool(measurable)
            and effective_coverage >= MINIMUM_EFFECTIVE_ELIGIBLE_COVERAGE
        ),
        "unmeasurable_hard_bypass": unmeasurable_bypass_fraction == 1.0,
    }
    return {
        "schema": 1,
        "contract": "per-example-effective-restoration-or-hard-bypass",
        "examples": len(rows),
        "measurable_examples": len(measurable),
        "effective_examples": len(effective),
        "hard_bypass_examples": sum(row["hard_bypass"] for row in rows),
        "safe_fraction": safe_fraction,
        "effective_eligible_coverage": effective_coverage,
        "unmeasurable_hard_bypass_fraction": unmeasurable_bypass_fraction,
        "gates": gates,
        "accepted": all(gates.values()),
    }
