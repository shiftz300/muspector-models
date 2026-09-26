"""Small black-box system-identification baselines.

The public estimator API only receives input/output arrays and a normalized
control value.  No estimator is allowed to inspect DUT or split metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import lfilter


def _lag_matrix(value: np.ndarray, taps: int) -> np.ndarray:
    padded = np.pad(np.asarray(value, dtype=np.float64), (taps - 1, 0))
    return np.stack(
        [padded[taps - 1 - lag:taps - 1 - lag + len(value)] for lag in range(taps)],
        axis=1,
    )


def _control_input(value: np.ndarray, control: float) -> np.ndarray:
    return np.asarray(value, dtype=np.float64) * (0.72 + 1.85 * float(control))


def _parallel_basis(value: np.ndarray, control: float) -> list[np.ndarray]:
    scaled = _control_input(value, control)
    return [
        scaled,
        scaled ** 2,
        scaled ** 3,
        scaled ** 4,
        scaled ** 5,
        np.tanh(scaled),
        np.sign(scaled) * np.abs(scaled) ** 1.5,
    ]


def _bounded_parallel_basis(value: np.ndarray, control: float) -> list[np.ndarray]:
    """Use bounded nonlinear branches to avoid polynomial amplitude blow-up."""

    scaled = _control_input(value, control)
    positive = np.maximum(scaled, 0.0)
    negative = np.minimum(scaled, 0.0)
    return [
        scaled,
        np.tanh(0.75 * scaled),
        np.tanh(1.50 * scaled),
        np.tanh(1.50 * positive),
        np.tanh(3.00 * positive),
        np.tanh(1.50 * negative),
        np.tanh(3.00 * negative),
    ]


@dataclass
class LinearEstimator:
    taps: int = 64
    ridge: float = 1.0e-5

    def fit(self, rows: list[dict]) -> "LinearEstimator":
        started = time.perf_counter()
        gram = np.zeros((self.taps, self.taps), dtype=np.float64)
        rhs = np.zeros(self.taps, dtype=np.float64)
        for row in rows:
            design = _lag_matrix(row["x"], self.taps)
            target = np.asarray(row["y"], dtype=np.float64)
            gram += design.T @ design
            rhs += design.T @ target
        self.coefficients = np.linalg.solve(
            gram + self.ridge * np.eye(self.taps), rhs
        ).astype(np.float32)
        self.fit_seconds = time.perf_counter() - started
        return self

    @property
    def parameters(self) -> int:
        return self.taps

    def predict(self, x: np.ndarray, control: float) -> np.ndarray:
        del control
        return np.asarray(lfilter(self.coefficients, [1.0], x), dtype=np.float32)


@dataclass
class ParallelHammersteinEstimator:
    taps: int = 64
    branches: int = 7
    ridge: float = 1.0e-5

    def fit(self, rows: list[dict]) -> "ParallelHammersteinEstimator":
        started = time.perf_counter()
        gram = np.zeros((self.taps * self.branches, self.taps * self.branches), dtype=np.float64)
        rhs = np.zeros(self.taps * self.branches, dtype=np.float64)
        for row in rows:
            matrices = [_lag_matrix(basis, self.taps) for basis in _parallel_basis(row["x"], row["control"])]
            design = np.concatenate(matrices, axis=1)
            target = np.asarray(row["y"], dtype=np.float64)
            gram += design.T @ design
            rhs += design.T @ target
        coefficients = np.linalg.solve(
            gram + self.ridge * np.eye(self.taps * self.branches), rhs
        )
        self.coefficients = coefficients.astype(np.float32).reshape(self.branches, self.taps)
        self.fit_seconds = time.perf_counter() - started
        return self

    @property
    def parameters(self) -> int:
        return self.taps * self.branches

    def predict(self, x: np.ndarray, control: float) -> np.ndarray:
        result = np.zeros_like(x, dtype=np.float64)
        for coefficient, basis in zip(self.coefficients, _parallel_basis(x, control)):
            result += lfilter(coefficient, [1.0], basis)
        return np.asarray(result, dtype=np.float32)


@dataclass
class BoundedParallelHammersteinEstimator:
    """Same parameter budget as PH, with bounded asymmetric nonlinearities."""

    taps: int = 64
    branches: int = 7
    ridge: float = 1.0e-5

    def fit(self, rows: list[dict]) -> "BoundedParallelHammersteinEstimator":
        if self.branches != 7:
            raise ValueError("bounded basis has exactly seven branches")
        started = time.perf_counter()
        width = self.taps * self.branches
        gram = np.zeros((width, width), dtype=np.float64)
        rhs = np.zeros(width, dtype=np.float64)
        for row in rows:
            matrices = [
                _lag_matrix(basis, self.taps)
                for basis in _bounded_parallel_basis(row["x"], row["control"])
            ]
            design = np.concatenate(matrices, axis=1)
            target = np.asarray(row["y"], dtype=np.float64)
            gram += design.T @ design
            rhs += design.T @ target
        coefficients = np.linalg.solve(
            gram + self.ridge * np.eye(width), rhs
        )
        self.coefficients = coefficients.astype(np.float32).reshape(self.branches, self.taps)
        self.fit_seconds = time.perf_counter() - started
        return self

    @property
    def parameters(self) -> int:
        return self.taps * self.branches

    def predict(self, x: np.ndarray, control: float) -> np.ndarray:
        result = np.zeros_like(x, dtype=np.float64)
        for coefficient, basis in zip(self.coefficients, _bounded_parallel_basis(x, control)):
            result += lfilter(coefficient, [1.0], basis)
        return np.asarray(result, dtype=np.float32)


class _WienerHammerstein(torch.nn.Module):
    def __init__(self, taps: int = 17) -> None:
        super().__init__()
        self.taps = taps
        self.first = torch.nn.Parameter(torch.zeros(1, 1, taps))
        self.second = torch.nn.Parameter(torch.zeros(1, 1, taps))
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        with torch.no_grad():
            self.first[0, 0, -1] = 1.0
            self.second[0, 0, -1] = 1.0

    def _causal(self, value: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        length = value.shape[-1]
        return torch.nn.functional.conv1d(
            torch.nn.functional.pad(value.unsqueeze(1), (self.taps - 1, 0)),
            kernel,
        ).squeeze(1)[..., :length]

    def forward(self, x: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        first = self._causal(x, self.first)
        drive = 0.72 + 1.85 * control.unsqueeze(1)
        middle = torch.tanh(first * drive)
        return self.scale * self._causal(middle, self.second)


class WienerHammersteinEstimator:
    """Low-order WH fit refined with a deterministic small torch optimizer."""

    def __init__(self, taps: int = 17, epochs: int = 90, device: str = "cpu") -> None:
        self.taps = taps
        self.epochs = epochs
        self.device = torch.device(device)

    def fit(self, rows: list[dict]) -> "WienerHammersteinEstimator":
        started = time.perf_counter()
        torch.manual_seed(20260902)
        self.model = _WienerHammerstein(self.taps).to(self.device)
        x = torch.from_numpy(np.stack([row["x"] for row in rows])).to(self.device)
        y = torch.from_numpy(np.stack([row["y"] for row in rows])).to(self.device)
        control = torch.tensor([row["control"] for row in rows], dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)
        best_loss = float("inf")
        best_state = None
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            prediction = self.model(x, control)
            loss = torch.mean((prediction - y) ** 2) + 1.0e-4 * (
                self.model.first.square().mean() + self.model.second.square().mean()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()
            value = float(loss.detach().cpu())
            if value < best_loss:
                best_loss = value
                best_state = {name: parameter.detach().cpu().clone() for name, parameter in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model = self.model.cpu().eval()
        self.fit_seconds = time.perf_counter() - started
        return self

    @property
    def parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def predict(self, x: np.ndarray, control: float) -> np.ndarray:
        with torch.inference_mode():
            prediction = self.model(
                torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0),
                torch.tensor([control], dtype=torch.float32),
            )
        return prediction[0].numpy().astype(np.float32)
