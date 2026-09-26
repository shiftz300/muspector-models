"""A single small bounded residual on top of a structured PH forward clone."""

from __future__ import annotations

import time

import numpy as np
import torch


class _ResidualGRU(torch.nn.Module):
    def __init__(self, hidden_size: int = 24) -> None:
        super().__init__()
        self.recurrent = torch.nn.GRU(2, hidden_size, batch_first=True)
        self.output = torch.nn.Linear(hidden_size, 1)
        # The structured clone is the exact initial path.  The residual must
        # earn any correction during fit instead of adding an initialization
        # transient.
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, structured: torch.Tensor) -> torch.Tensor:
        features = torch.stack((x, structured), dim=-1)
        hidden, _ = self.recurrent(features)
        return 0.35 * torch.tanh(self.output(hidden).squeeze(-1))


class ParallelPlusResidual2k:
    """A fixed tiny causal residual with a bounded output correction."""

    def __init__(
        self,
        base,
        device: str = "mps",
        epochs: int = 12,
        hidden_size: int = 24,
    ) -> None:
        self.base = base
        self.device = torch.device(device)
        self.epochs = epochs
        self.hidden_size = hidden_size

    @property
    def residual_parameters(self) -> int:
        return int(sum(parameter.numel() for parameter in self.network.parameters()))

    @property
    def parameters(self) -> int:
        return int(self.base.parameters + self.residual_parameters)

    def fit(self, rows: list[dict]) -> "ParallelPlusResidual2k":
        started = time.perf_counter()
        torch.manual_seed(20260903)
        structured = np.stack([
            self.base.predict(row["x"], row["control"]) for row in rows
        ]).astype(np.float32)
        x = torch.from_numpy(np.stack([row["x"] for row in rows])).to(self.device)
        base = torch.from_numpy(structured).to(self.device)
        target = torch.from_numpy(np.stack([row["y"] for row in rows])).to(self.device)
        self.network = _ResidualGRU(self.hidden_size).to(self.device)
        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=0.003, weight_decay=1.0e-5,
        )
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
        self.fit_seconds = float(time.perf_counter() - started)
        self.residual_fit_loss = best_loss
        return self

    def predict(self, x: np.ndarray, control: float) -> np.ndarray:
        base = self.base.predict(x, control)
        with torch.inference_mode():
            correction = self.network(
                torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0),
                torch.from_numpy(base).unsqueeze(0),
            )[0].numpy()
        return np.asarray(base + correction, dtype=np.float32)
