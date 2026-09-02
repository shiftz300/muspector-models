"""Control-conditioned time-domain inverse for nonlinear Drive stages."""

from __future__ import annotations

import torch
from torch import nn


DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)


class Block(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.filter = nn.Conv1d(channels, channels * 2, 9, padding=dilation * 4, dilation=dilation)
        self.condition = nn.Linear(3, channels * 2, bias=False)
        self.mix = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, value: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        left, right = (self.filter(value) + self.condition(controls).unsqueeze(-1)).chunk(2, 1)
        return value + self.mix(torch.tanh(left) * torch.sigmoid(right))


class TimeNet(nn.Module):
    """Finite-history residual TCN with exact strength-zero bypass."""

    def __init__(self, channels: int = 20, dilations: tuple[int, ...] = DILATIONS) -> None:
        super().__init__()
        self.channels = int(channels)
        self.dilations = tuple(map(int, dilations))
        self.stem = nn.Conv1d(1, channels, 15, padding=7)
        self.input_condition = nn.Linear(3, channels, bias=False)
        self.blocks = nn.ModuleList(Block(channels, value) for value in self.dilations)
        self.head = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.input_condition.weight)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _controls(controls: torch.Tensor, batch: int) -> torch.Tensor:
        controls = torch.as_tensor(controls)
        if controls.shape != (batch, 3) or not torch.isfinite(controls).all():
            raise ValueError("controls must be finite [batch,3]")
        if bool((controls < 0.0).any()) or bool((controls > 1.0).any()):
            raise ValueError("controls must be normalized to [0,1]")
        return controls

    def forward(self, wet: torch.Tensor, controls: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        if wet.ndim != 2 or not torch.isfinite(wet).all():
            raise ValueError("wet audio must be finite [batch,time]")
        controls = self._controls(controls, wet.shape[0]).to(wet)
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError("restoration strength must be between zero and one")
        if strength == 0.0:
            return wet
        value = self.stem(wet.unsqueeze(1)) + self.input_condition(controls).unsqueeze(-1)
        for block in self.blocks:
            value = block(value, controls)
        return wet + self.head(torch.tanh(value)).squeeze(1) * float(strength)
