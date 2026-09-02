"""Versioned mechanism-aware restoration gates with stable old-style semantics."""

from __future__ import annotations

import numpy as np

from .restoration_quality import (
    _audio,
    _band_envelope,
    _crest_error,
    _relative_error,
    _spectral_error,
    _transient_envelope,
)


SCHEMA = 2
ERROR_FLOOR = 1.0e-4
REGRESSION_RELATIVE_TOLERANCE = 0.02
THRESHOLDS = {
    "spectrum": 0.15,
    "high_band": 0.10,
    "transient": 0.10,
    "dynamics": 0.10,
}
REQUIRED_IMPROVEMENTS = {
    "nonlinear": {"spectrum", "high_band", "transient"},
    "dynamics": {"dynamics"},
    "spectral": {"spectrum", "high_band"},
    "modulation": {"transient", "dynamics"},
    "echo": {"spectrum"},
    "ambience": {"spectrum", "high_band", "transient"},
    "amp": {"spectrum", "high_band", "transient", "dynamics"},
}


def reduction(baseline: float, restored: float) -> dict:
    baseline = float(baseline)
    restored = float(restored)
    eligible = baseline >= ERROR_FLOOR
    raw = 1.0 - restored / max(baseline, ERROR_FLOOR)
    bounded = float(np.clip(raw, -1.0, 1.0))
    nonregression_limit = baseline * (1.0 + REGRESSION_RELATIVE_TOLERANCE) + ERROR_FLOOR
    return {
        "baseline_error": baseline,
        "restored_error": restored,
        "eligible": eligible,
        "reduction": bounded,
        "raw_reduction": raw,
        "nonregression": restored <= nonregression_limit,
    }


def measure(mechanism: str, wet: np.ndarray, restored: np.ndarray, clean: np.ndarray) -> dict:
    if mechanism not in REQUIRED_IMPROVEMENTS:
        raise ValueError(f"unsupported quality2 mechanism: {mechanism}")
    clean, wet, restored = _audio(clean), _audio(wet), _audio(restored)
    length = min(len(clean), len(wet), len(restored))
    clean, wet, restored = clean[:length], wet[:length], restored[:length]
    clean_high = _band_envelope(clean, 3000.0, 12000.0)
    clean_attack = _transient_envelope(clean)
    metrics = {
        "spectrum": reduction(_spectral_error(wet, clean), _spectral_error(restored, clean)),
        "high_band": reduction(
            _relative_error(_band_envelope(wet, 3000.0, 12000.0), clean_high),
            _relative_error(_band_envelope(restored, 3000.0, 12000.0), clean_high),
        ),
        "transient": reduction(
            _relative_error(_transient_envelope(wet), clean_attack),
            _relative_error(_transient_envelope(restored), clean_attack),
        ),
        "dynamics": reduction(_crest_error(wet, clean), _crest_error(restored, clean)),
    }
    required = REQUIRED_IMPROVEMENTS[mechanism]
    gates = {}
    for name, row in metrics.items():
        if name in required and row["eligible"]:
            gates[name] = row["reduction"] >= THRESHOLDS[name]
        else:
            gates[name] = row["nonregression"]
    correction = float(np.sqrt(np.mean(np.square(restored - wet))))
    wet_distance = float(np.sqrt(np.mean(np.square(wet - clean))))
    gates["meaningful_correction"] = correction >= 0.05 * max(wet_distance, 1.0e-6)
    added_clipping = bool(
        np.mean(np.abs(restored) >= 0.999) > np.mean(np.abs(wet) >= 0.999) + 1.0e-4
    )
    peak = float(np.max(np.abs(restored)))
    gates["no_new_clipping"] = not added_clipping and peak <= 1.05
    return {
        "schema": SCHEMA,
        "mechanism": mechanism,
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "correction_to_wet_distance": correction / max(wet_distance, 1.0e-6),
        "restored_peak": peak,
        "added_clipping": added_clipping,
    }


def summarize(mechanism: str, wet: list[np.ndarray], restored: list[np.ndarray], clean: list[np.ndarray]) -> dict:
    rows = [measure(mechanism, left, middle, right) for left, middle, right in zip(wet, restored, clean, strict=True)]
    required = REQUIRED_IMPROVEMENTS[mechanism]
    metric_reports = {}
    gates = {}
    for name in ("spectrum", "high_band", "transient", "dynamics"):
        selected = [row["metrics"][name] for row in rows]
        eligible = [item for item in selected if item["eligible"]]
        median = None if not eligible else float(np.median([item["reduction"] for item in eligible]))
        nonregression_fraction = float(np.mean([item["nonregression"] for item in selected]))
        metric_reports[name] = {
            "eligible_examples": len(eligible),
            "eligible_fraction": len(eligible) / len(selected),
            "median_reduction": median,
            "nonregression_fraction": nonregression_fraction,
        }
        if name in required:
            gates[name] = bool(eligible and median is not None and median >= THRESHOLDS[name])
        else:
            gates[name] = nonregression_fraction >= 0.90
    pass_fraction = float(np.mean([row["passed"] for row in rows]))
    gates["pass_fraction"] = pass_fraction >= 0.50
    gates["no_new_clipping"] = float(np.mean([row["added_clipping"] for row in rows])) <= 0.01
    return {
        "schema": SCHEMA,
        "mechanism": mechanism,
        "examples": len(rows),
        "metrics": metric_reports,
        "pass_fraction": pass_fraction,
        "gates": gates,
        "accepted": all(gates.values()),
    }
