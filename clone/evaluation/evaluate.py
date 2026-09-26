"""Unified absolute metrics for the Structural-OOD benchmark."""

from __future__ import annotations

import time

import numpy as np


def _pre_emphasis(value: np.ndarray, coefficient: float = 0.85) -> np.ndarray:
    return np.concatenate(([value[0]], value[1:] - coefficient * value[:-1]))


def _esr(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2) / max(float(np.mean(target ** 2)), 1.0e-12))


def _spectrum(prediction: np.ndarray, target: np.ndarray) -> float:
    values = []
    for size in (256, 512, 1024):
        hop = size // 4
        window = np.hanning(size)
        count = max(1, 1 + (len(target) - size) // hop)
        prediction_frames = []
        target_frames = []
        for index in range(count):
            start = index * hop
            prediction_frames.append(np.abs(np.fft.rfft(prediction[start:start + size] * window, n=size)))
            target_frames.append(np.abs(np.fft.rfft(target[start:start + size] * window, n=size)))
        p = np.stack(prediction_frames)
        t = np.stack(target_frames)
        values.append(float(np.mean(np.abs(np.log1p(p) - np.log1p(t)))))
    return float(np.mean(values))


def one_example(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    emphasized_prediction = _pre_emphasis(prediction)
    emphasized_target = _pre_emphasis(target)
    return {
        "absolute_esr": _esr(prediction, target),
        "pre_emphasis_esr": _esr(emphasized_prediction, emphasized_target),
        "mae": float(np.mean(np.abs(prediction - target))),
        "mr_spectrum": _spectrum(prediction, target),
        "peak_error": float(abs(np.max(np.abs(prediction)) - np.max(np.abs(target)))),
    }


def summarize(rows: list[dict[str, float]]) -> dict:
    if not rows:
        raise ValueError("cannot summarize empty metric rows")
    names = tuple(rows[0])
    result = {"examples": len(rows)}
    for name in names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        result[name] = {
            "mean": float(np.mean(values)),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "p99": float(np.quantile(values, 0.99)),
            "worst": float(np.max(values)),
        }
    return result


def benchmark_runtime(model, frames: int, rate: int) -> float:
    sample = np.zeros(frames, dtype=np.float32)
    started = time.perf_counter()
    for _ in range(3):
        model.predict(sample, 0.5)
    elapsed = (time.perf_counter() - started) / 3.0
    return float(elapsed / (frames / rate))
