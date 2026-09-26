"""The fixed approximately 4.8k-parameter causal LSTM baseline."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.nn import functional as F


def _smooth(value: torch.Tensor, kernel: int) -> torch.Tensor:
    return F.avg_pool1d(value, kernel, stride=1, padding=kernel // 2)


def restoration_training_loss(
    prediction: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Use the repository's restoration-aware objective for inverse training."""

    waveform = F.l1_loss(prediction, target)
    prediction_difference = prediction[..., 1:] - prediction[..., :-1]
    target_difference = target[..., 1:] - target[..., :-1]
    transient = F.l1_loss(prediction_difference, target_difference)
    envelope = F.l1_loss(
        _smooth(torch.abs(prediction[:, None]), 65),
        _smooth(torch.abs(target[:, None]), 65),
    )
    long_shape = F.l1_loss(
        _smooth(prediction[:, None], 513),
        _smooth(target[:, None], 513),
    )
    loss = waveform + 0.5 * transient + 0.25 * envelope + 0.15 * long_shape
    return loss, {
        "waveform": float(waveform.detach()),
        "transient": float(transient.detach()),
        "envelope": float(envelope.detach()),
        "long_shape": float(long_shape.detach()),
    }


class CausalLSTM48:
    """Control-preconditioned one-layer causal LSTM, about 4.8k parameters."""

    def __init__(
        self,
        hidden_size: int = 33,
        epochs: int = 22,
        device: str = "mps",
        amplitude_scale: float = 1.0,
        residual: bool = False,
    ) -> None:
        if not np.isfinite(amplitude_scale) or amplitude_scale <= 0.0:
            raise ValueError("amplitude_scale must be finite and positive")
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.device = torch.device(device)
        self.amplitude_scale = float(amplitude_scale)
        self.residual = bool(residual)
        self.network = torch.nn.ModuleDict({
            "lstm": torch.nn.LSTM(1, hidden_size, batch_first=True),
            "output": torch.nn.Linear(hidden_size, 1),
        })
        if self.residual:
            # Start from the safe identity path; the network must earn its
            # correction rather than introduce a transient at initialization.
            torch.nn.init.zeros_(self.network["output"].weight)
            torch.nn.init.zeros_(self.network["output"].bias)

    @property
    def parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.network.parameters())

    def _input(self, x: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        scale = 0.72 + 1.85 * control.unsqueeze(1)
        return (x * scale).unsqueeze(-1)

    def fit(self, rows: list[dict], objective: str = "mse") -> "CausalLSTM48":
        if objective not in ("mse", "restoration"):
            raise ValueError(f"unsupported training objective: {objective}")
        started = time.perf_counter()
        torch.manual_seed(20260902)
        self.network = self.network.to(self.device)
        x = torch.from_numpy(np.stack([row["x"] for row in rows])).to(self.device)
        y = torch.from_numpy(np.stack([row["y"] for row in rows])).to(self.device)
        x = x / self.amplitude_scale
        y = y / self.amplitude_scale
        control = torch.tensor([row["control"] for row in rows], dtype=torch.float32, device=self.device)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=0.004, weight_decay=1.0e-5)
        best_loss = float("inf")
        best_state = None
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            hidden, _ = self.network["lstm"](self._input(x, control))
            prediction = self.network["output"](hidden).squeeze(-1)
            if self.residual:
                prediction = prediction + x
            if objective == "mse":
                loss = torch.mean((prediction - y) ** 2)
            else:
                loss, _ = restoration_training_loss(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            optimizer.step()
            value = float(loss.detach().cpu())
            if value < best_loss:
                best_loss = value
                best_state = {name: parameter.detach().cpu().clone() for name, parameter in self.network.state_dict().items()}
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network = self.network.cpu().eval()
        self.objective = objective
        self.fit_seconds = time.perf_counter() - started
        return self

    def predict(self, x: np.ndarray, control: float) -> np.ndarray:
        with torch.inference_mode():
            input_value = torch.from_numpy(
                np.asarray(x, dtype=np.float32) / self.amplitude_scale
            ).unsqueeze(0)
            condition = torch.tensor([control], dtype=torch.float32)
            hidden, _ = self.network["lstm"](self._input(input_value, condition))
            prediction = self.network["output"](hidden).squeeze(-1)
            if self.residual:
                prediction = prediction + input_value
        return (prediction[0].numpy() * self.amplitude_scale).astype(np.float32)
