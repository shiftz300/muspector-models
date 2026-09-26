"""A small residual learner on top of the dynamic gray-box estimate."""

from __future__ import annotations

import time

import numpy as np
import torch

from ..sysid.state_bank import DynamicGrayBoxEstimator


class _ResidualGRU(torch.nn.Module):
    def __init__(self, hidden_size: int = 24) -> None:
        super().__init__()
        self.recurrent = torch.nn.GRU(2, hidden_size, batch_first=True)
        self.output = torch.nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, structured: torch.Tensor) -> torch.Tensor:
        features = torch.stack((x, structured), dim=-1)
        hidden, _ = self.recurrent(features)
        return 0.35 * torch.tanh(self.output(hidden).squeeze(-1))


class StructuredPlusResidual2k:
    """576-parameter state bank plus a 2,041-parameter causal residual."""

    def __init__(self, device: str = "mps", epochs: int = 16) -> None:
        self.device = torch.device(device)
        self.epochs = epochs

    @property
    def parameters(self) -> int:
        return int(self.structured.parameters + sum(
            parameter.numel() for parameter in self.network.parameters()
        ))

    @property
    def residual_parameters(self) -> int:
        return int(sum(parameter.numel() for parameter in self.network.parameters()))

    def fit(self, rows: list[dict]) -> "StructuredPlusResidual2k":
        started = time.perf_counter()
        self.structured = DynamicGrayBoxEstimator().fit(rows)
        structured = np.stack([
            self.structured.predict(row["x"], row["control"]) for row in rows
        ])
        x = torch.from_numpy(np.stack([row["x"] for row in rows])).to(self.device)
        base = torch.from_numpy(structured.astype(np.float32)).to(self.device)
        target = torch.from_numpy(np.stack([row["y"] for row in rows])).to(self.device)
        self.network = _ResidualGRU().to(self.device)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=0.003, weight_decay=1.0e-5)
        best_loss = float("inf")
        best_state = None
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            prediction = base + self.network(x, base)
            loss = torch.mean((prediction - target) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            optimizer.step()
            value = float(loss.detach().cpu())
            if value < best_loss:
                best_loss = value
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in self.network.state_dict().items()
                }
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network = self.network.cpu().eval()
        self.fit_seconds = time.perf_counter() - started
        self.residual_fit_loss = best_loss
        return self

    def predict(self, x: np.ndarray, control: float) -> np.ndarray:
        base = self.structured.predict(x, control)
        with torch.inference_mode():
            correction = self.network(
                torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0),
                torch.from_numpy(base).unsqueeze(0),
            )[0].numpy()
        return np.asarray(base + correction, dtype=np.float32)
