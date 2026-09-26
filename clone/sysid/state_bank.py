"""Low-order dynamic gray-box estimator with a fixed envelope state bank."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter


RATE = 48_000
TIME_CONSTANTS = (0.0005, 0.002, 0.008, 0.032, 0.128, 0.5)


def _state_bank(value: np.ndarray) -> np.ndarray:
    """Return causal absolute and signed states for fixed time constants."""

    value = np.asarray(value, dtype=np.float64)
    states = np.empty((len(TIME_CONSTANTS), len(value)), dtype=np.float64)
    for row, seconds in enumerate(TIME_CONSTANTS):
        coefficient = float(np.exp(-1.0 / (seconds * RATE)))
        state = 0.0
        for index, sample in enumerate(value):
            state = coefficient * state + (1.0 - coefficient) * abs(float(sample))
            states[row, index] = state
    return states


def _signed_state(value: np.ndarray, seconds: float) -> np.ndarray:
    coefficient = float(np.exp(-1.0 / (seconds * RATE)))
    result = np.empty(len(value), dtype=np.float64)
    state = 0.0
    for index, sample in enumerate(value):
        state = coefficient * state + (1.0 - coefficient) * float(sample)
        result[index] = state
    return result


def _dynamic_basis(value: np.ndarray, control: float) -> list[np.ndarray]:
    scaled = np.asarray(value, dtype=np.float64) * (0.72 + 1.85 * float(control))
    absolute_states = _state_bank(value)
    signed_states = np.stack(
        [_signed_state(value, seconds) for seconds in TIME_CONSTANTS], axis=0
    )
    basis = [
        scaled,
        scaled ** 2,
        scaled ** 3,
        scaled ** 5,
        np.tanh(scaled),
        np.sign(scaled) * np.abs(scaled) ** 1.5,
    ]
    for absolute, signed in zip(absolute_states, signed_states):
        basis.extend((scaled * absolute, np.sign(scaled) * absolute, scaled * signed))
    return basis


def _lag_matrix(value: np.ndarray, taps: int) -> np.ndarray:
    padded = np.pad(np.asarray(value, dtype=np.float64), (taps - 1, 0))
    return np.stack(
        [padded[taps - 1 - lag:taps - 1 - lag + len(value)] for lag in range(taps)],
        axis=1,
    )


@dataclass
class DynamicGrayBoxEstimator:
    """State-bank nonlinear branches followed by a short fitted FIR bank."""

    taps: int = 24
    ridge: float = 1.0e-3

    def fit(self, rows: list[dict]) -> "DynamicGrayBoxEstimator":
        started = time.perf_counter()
        branch_count = 6 + 3 * len(TIME_CONSTANTS)
        width = branch_count * self.taps
        gram = np.zeros((width, width), dtype=np.float64)
        rhs = np.zeros(width, dtype=np.float64)
        for row in rows:
            design = np.concatenate(
                [_lag_matrix(branch, self.taps) for branch in _dynamic_basis(row["x"], row["control"])],
                axis=1,
            )
            target = np.asarray(row["y"], dtype=np.float64)
            gram += design.T @ design
            rhs += design.T @ target
        self.coefficients = np.linalg.solve(gram + self.ridge * np.eye(width), rhs)
        self.branch_count = branch_count
        self.fit_seconds = time.perf_counter() - started
        return self

    @property
    def parameters(self) -> int:
        return int(self.branch_count * self.taps)

    def predict(self, x: np.ndarray, control: float) -> np.ndarray:
        result = np.zeros(len(x), dtype=np.float64)
        for index, branch in enumerate(_dynamic_basis(x, control)):
            coefficients = self.coefficients[index * self.taps:(index + 1) * self.taps]
            result += lfilter(coefficients, [1.0], branch)
        return np.asarray(result, dtype=np.float32)
