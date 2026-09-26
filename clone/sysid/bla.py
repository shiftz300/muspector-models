"""Best linear approximation diagnostics for black-box audio observations."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _frames(value: np.ndarray, size: int, hop: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or len(value) < size:
        raise ValueError("BLA input is shorter than the FFT frame")
    count = 1 + (len(value) - size) // hop
    shape = (count, size)
    strides = (value.strides[0] * hop, value.strides[0])
    return np.lib.stride_tricks.as_strided(value, shape=shape, strides=strides).copy()


def estimate(rows: list[dict], fft_size: int = 1024, hop: int | None = None) -> dict:
    """Estimate a transfer function from fit-only input/output rows.

    The result is a sufficient statistic plus the transfer function.  It is
    intentionally an analysis object; it does not expose any hidden DUT
    metadata to an estimator or create a model checkpoint.
    """

    if not rows:
        raise ValueError("BLA needs at least one observation")
    hop = fft_size // 4 if hop is None else hop
    if fft_size < 64 or hop < 1 or hop > fft_size:
        raise ValueError("invalid BLA FFT geometry")
    window = np.hanning(fft_size).astype(np.float64)
    cross = np.zeros(fft_size // 2 + 1, dtype=np.complex128)
    power = np.zeros(fft_size // 2 + 1, dtype=np.float64)
    frame_count = 0
    for row in rows:
        x = _frames(row["x"], fft_size, hop) * window
        y = _frames(row["y"], fft_size, hop) * window
        x_spectrum = np.fft.rfft(x, axis=1)
        y_spectrum = np.fft.rfft(y, axis=1)
        cross += np.sum(np.conj(x_spectrum) * y_spectrum, axis=0)
        power += np.sum(np.abs(x_spectrum) ** 2, axis=0)
        frame_count += x.shape[0]
    transfer = cross / np.maximum(power, 1.0e-12)
    return {
        "fft_size": fft_size,
        "hop": hop,
        "frames": frame_count,
        "transfer": transfer,
        "input_power": power,
    }


def _weighted_stats(reference: dict, candidate: dict) -> dict:
    if reference["fft_size"] != candidate["fft_size"]:
        raise ValueError("BLA comparisons require the same FFT size")
    reference_transfer = np.asarray(reference["transfer"], dtype=np.complex128)
    candidate_transfer = np.asarray(candidate["transfer"], dtype=np.complex128)
    reference_power = np.asarray(reference["input_power"], dtype=np.float64)
    candidate_power = np.asarray(candidate["input_power"], dtype=np.float64)
    power = np.minimum(reference_power, candidate_power)
    threshold = max(float(np.max(power)) * 1.0e-6, 1.0e-12)
    selected = power >= threshold
    if not np.any(selected):
        raise ValueError("BLA comparison has no excited frequency bins")
    weights = power[selected] / np.sum(power[selected])
    reference_values = reference_transfer[selected]
    candidate_values = candidate_transfer[selected]
    reference_norm = np.sqrt(np.sum(weights * np.abs(reference_values) ** 2))
    candidate_norm = np.sqrt(np.sum(weights * np.abs(candidate_values) ** 2))
    reference_unit = reference_values / max(float(reference_norm), 1.0e-12)
    candidate_unit = candidate_values / max(float(candidate_norm), 1.0e-12)
    complex_similarity = float(np.abs(np.sum(weights * np.conj(reference_unit) * candidate_unit)))
    reference_magnitude = np.abs(reference_unit)
    candidate_magnitude = np.abs(candidate_unit)
    magnitude_drift = float(np.sum(weights * np.abs(
        np.log1p(candidate_magnitude) - np.log1p(reference_magnitude)
    )))
    phase_delta = np.angle(candidate_values * np.conj(reference_values))
    phase_rms = float(np.sqrt(np.sum(weights * np.square(phase_delta))))
    return {
        "excited_bins": int(np.sum(selected)),
        "shape_similarity": complex_similarity,
        "shape_drift": float(np.clip(1.0 - complex_similarity, 0.0, 1.0)),
        "normalized_magnitude_drift": magnitude_drift,
        "phase_rms_radians": phase_rms,
        "rms_gain_ratio": float(candidate_norm / max(float(reference_norm), 1.0e-12)),
    }


def compare(reference: dict, candidate: dict) -> dict:
    """Compare normalized BLA shape and phase at two excitation conditions."""

    return _weighted_stats(reference, candidate)


def grouped_bla(rows, *, fft_size: int = 1024) -> dict:
    """Return level and control drift reports for each hidden DUT group."""

    groups = defaultdict(list)
    for row in rows:
        groups[(row.topology, row.level, row.control)].append(row.public())
    levels = sorted({row.level for row in rows})
    controls = sorted({row.control for row in rows})
    topologies = sorted({row.topology for row in rows})
    result = {}
    for topology in topologies:
        transfers = {
            f"level={level:.6g};control={control:.6g}": estimate(
                groups[(topology, level, control)], fft_size,
            )
            for level in levels for control in controls
        }
        level_rows = []
        for control in controls:
            reference = transfers[f"level={levels[0]:.6g};control={control:.6g}"]
            for level in levels[1:]:
                level_rows.append({
                    "reference_level": levels[0],
                    "candidate_level": level,
                    "control": control,
                    **compare(reference, transfers[f"level={level:.6g};control={control:.6g}"]),
                })
        control_rows = []
        reference_level = levels[len(levels) // 2]
        reference = transfers[f"level={reference_level:.6g};control={controls[0]:.6g}"]
        for control in controls[1:]:
            control_rows.append({
                "reference_control": controls[0],
                "candidate_control": control,
                "level": reference_level,
                **compare(reference, transfers[f"level={reference_level:.6g};control={control:.6g}" ]),
            })
        result[topology] = {
            "fft_size": fft_size,
            "levels": levels,
            "controls": controls,
            "level_drift": level_rows,
            "control_drift": control_rows,
            "summary": {
                "level_shape_drift_mean": float(np.mean([row["shape_drift"] for row in level_rows])),
                "level_shape_drift_max": float(np.max([row["shape_drift"] for row in level_rows])),
                "level_magnitude_drift_mean": float(np.mean([
                    row["normalized_magnitude_drift"] for row in level_rows
                ])),
                "control_shape_drift_mean": float(np.mean([row["shape_drift"] for row in control_rows])),
                "control_shape_drift_max": float(np.max([row["shape_drift"] for row in control_rows])),
                "control_magnitude_drift_mean": float(np.mean([
                    row["normalized_magnitude_drift"] for row in control_rows
                ])),
            },
        }
    return result
