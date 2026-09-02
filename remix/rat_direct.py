"""Causal direct renderer for the ASRNN ProCo RAT pilot."""

from __future__ import annotations

import torch


class RatDirectRenderer(torch.nn.Module):
    """Deep bias-free RNN with a physically factored output-volume control.

    Distortion and filter condition the nonlinear recurrent path. The RAT volume
    control is modeled as a post-circuit gain, so a zero volume structurally
    produces exact digital silence without a limiter or post-render repair.
    """

    def __init__(self, hidden_size: int = 32, layers: int = 4) -> None:
        super().__init__()
        if hidden_size <= 0 or layers <= 0:
            raise ValueError("RAT renderer dimensions must be positive")
        self.hidden_size = hidden_size
        self.layers = layers
        self.recurrent = torch.nn.LSTM(
            3,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            bias=False,
        )
        self.output = torch.nn.Linear(hidden_size, 1, bias=False)
        self.control_gain = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 1),
        )
        torch.nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-3)
        torch.nn.init.zeros_(self.control_gain[-1].weight)
        torch.nn.init.zeros_(self.control_gain[-1].bias)

    def forward(
        self,
        dry: torch.Tensor,
        controls: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if not torch.jit.is_tracing():
            if dry.ndim != 2 or controls.shape != (dry.shape[0], 3):
                raise ValueError("RAT renderer expects dry [batch,time] and controls [batch,3]")
            if not torch.isfinite(dry).all() or not torch.isfinite(controls).all():
                raise ValueError("RAT renderer inputs must be finite")
            if torch.any(controls < 0.0) or torch.any(controls > 1.0):
                raise ValueError("RAT renderer controls must be normalized")
        centered = controls[:, :2].mul(2.0).sub(1.0)
        features = torch.cat(
            (
                dry.unsqueeze(-1),
                dry.unsqueeze(-1) * centered.unsqueeze(1),
            ),
            dim=-1,
        )
        hidden, next_state = self.recurrent(features, state)
        circuit = dry + self.output(hidden).squeeze(-1)
        volume = controls[:, 2:3].square()
        learned_gain = torch.exp(self.control_gain(controls).clamp(-4.0, 3.0))
        rendered = circuit * volume * learned_gain
        if not torch.jit.is_tracing() and not torch.isfinite(rendered).all():
            raise ValueError("RAT renderer produced non-finite audio")
        return rendered, next_state

