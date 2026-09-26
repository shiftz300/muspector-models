"""Small causal Wiener-Hammerstein forward replay model for Guitar-TECHS."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AmpForwardReplay(nn.Module):
    """Profile-specific direct-input to Amp+cab+mic forward clone.

    This is a diagnostic replay model, not an inverse or a physical-device
    replacement.  Its causal filters and bounded nonlinearity make the
    structure explicit before any inverse model is trained against replay.
    """

    def __init__(self, pre_taps: int = 65, post_taps: int = 257) -> None:
        super().__init__()
        if pre_taps < 3 or post_taps < 3 or pre_taps % 2 == 0 or post_taps % 2 == 0:
            raise ValueError("forward replay taps must be odd and at least three")
        self.pre_taps = pre_taps
        self.post_taps = post_taps
        self.pre = nn.Conv1d(1, 1, pre_taps, bias=False)
        self.post = nn.Conv1d(1, 1, post_taps, bias=False)
        self.log_drive = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        with torch.no_grad():
            self.pre.weight.zero_()
            self.post.weight.zero_()
            self.pre.weight[0, 0, -1] = 1.0
            self.post.weight[0, 0, -1] = 1.0

    @staticmethod
    def _causal(value: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        taps = kernel.shape[-1]
        return F.conv1d(F.pad(value.unsqueeze(1), (taps - 1, 0)), kernel).squeeze(1)

    def forward(self, direct: torch.Tensor) -> torch.Tensor:
        if direct.ndim != 2 or not torch.isfinite(direct).all():
            raise ValueError("forward replay expects finite direct [batch,time]")
        filtered = self._causal(direct, self.pre.weight)
        drive = torch.exp(self.log_drive).clamp(0.05, 20.0)
        shaped = torch.tanh(filtered * drive + self.bias)
        replay = self._causal(shaped, self.post.weight)
        scale = torch.exp(self.log_scale).clamp(0.05, 20.0)
        return scale * replay

    def manifest(self) -> dict:
        return {
            "schema": 1,
            "architecture": "causal-wiener-hammerstein-forward-replay",
            "scope": "Guitar-TECHS fixed direct-input to Amp+cab+mic profile",
            "pre_taps": self.pre_taps,
            "post_taps": self.post_taps,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "causal": True,
            "topology_input": False,
            "neighbor_effect_input": False,
            "physical_audio_devices_used": False,
        }
