"""Long-context hidden-trajectory estimator for independent Tremolo removal."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .modulation3 import CONTROL_WIDTH, DEPTH_MAX, DEPTH_MIN, RATE_MAX, RATE_MIN


FRAME = 240
HOP = 120


class _Block(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.temporal = nn.Conv1d(
            channels, channels * 2, 5, padding=2 * dilation, dilation=dilation
        )
        self.mix = nn.Conv1d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        content, gate = self.temporal(value).chunk(2, dim=1)
        return value + 0.25 * self.mix(torch.tanh(content) * torch.sigmoid(gate))


class TremoloInverseV3(nn.Module):
    mechanism = "modulation"
    family = "tremolo"

    def __init__(self, channels: int = 20, depth: int = 9) -> None:
        super().__init__()
        if channels < 6 or not 6 <= depth <= 10:
            raise ValueError("invalid Tremolo expert geometry")
        self.channels = channels
        self.depth = depth
        self.dilations = tuple(2**index for index in range(depth))
        self.stem = nn.Conv1d(4 + CONTROL_WIDTH, channels, 1)
        self.blocks = nn.ModuleList(_Block(channels, dilation) for dilation in self.dilations)
        self.trajectory = nn.Conv1d(channels, 1, 1)
        self.uncertainty = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.trajectory.weight)
        nn.init.zeros_(self.trajectory.bias)
        nn.init.zeros_(self.uncertainty.weight)
        nn.init.constant_(self.uncertainty.bias, -3.0)

    def forward(
        self, wet: torch.Tensor, controls: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if wet.ndim != 2 or controls.shape != (wet.shape[0], CONTROL_WIDTH):
            raise ValueError("Tremolo inverse expects Wet [batch,time] and controls [batch,5]")
        if not torch.isfinite(wet).all() or not torch.isfinite(controls).all():
            raise ValueError("Tremolo inverse inputs must be finite")
        absolute = F.avg_pool1d(wet.abs()[:, None], FRAME, HOP, padding=FRAME // 2).clamp_min(1.0e-6)
        power = F.avg_pool1d(wet.square()[:, None], FRAME, HOP, padding=FRAME // 2).clamp_min(1.0e-10).sqrt()
        log_absolute = torch.log(absolute)
        log_power = torch.log(power)
        delta = F.pad(log_power[:, :, 1:] - log_power[:, :, :-1], (1, 0))
        curvature = F.pad(delta[:, :, 1:] - delta[:, :, :-1], (1, 0))
        features = torch.cat((log_absolute, log_power, delta, curvature), dim=1)
        features = features - features.mean(dim=2, keepdim=True)
        condition = controls.mul(2.0).sub(1.0)[:, :, None].expand(-1, -1, features.shape[2])
        hidden = torch.tanh(self.stem(torch.cat((features, condition), dim=1)))
        for block in self.blocks:
            hidden = block(hidden)
        depth = DEPTH_MIN + controls[:, 1:2] * (DEPTH_MAX - DEPTH_MIN)
        maximum_log_gain = -torch.log1p(-depth).unsqueeze(-1)
        frame_inverse_log_gain = torch.sigmoid(self.trajectory(hidden)) * maximum_log_gain
        frame_uncertainty = F.softplus(self.uncertainty(hidden)) + 1.0e-5
        inverse_log_gain = F.interpolate(
            frame_inverse_log_gain, size=wet.shape[1], mode="linear", align_corners=False
        ).squeeze(1)
        uncertainty = F.interpolate(
            frame_uncertainty, size=wet.shape[1], mode="linear", align_corners=False
        ).squeeze(1)
        restored = wet * torch.exp(inverse_log_gain)
        return restored, uncertainty, inverse_log_gain

    def manifest(self) -> dict:
        frame_context = 1 + 4 * sum(self.dilations)
        return {
            "schema": 1,
            "architecture": "pooled-envelope-dilated-tcn-hidden-lfo-trajectory-inverse",
            "mechanism": self.mechanism,
            "family": self.family,
            "channels": self.channels,
            "depth": self.depth,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "control_width": CONTROL_WIDTH,
            "controls": ["rate_hz", "depth", "waveform", "reserved", "reserved"],
            "hidden_forward_state": "lfo_start_phase",
            "hidden_forward_state_is_inference_input": False,
            "frame": FRAME,
            "hop": HOP,
            "context_frames": frame_context * HOP + FRAME,
            "normalization_scope": "bounded input window",
            "causal": False,
            "uncertainty_output": True,
            "graph_order_input": False,
            "neighbouring_effect_input": False,
            "clean_or_oracle_input": False,
            "rate_domain_hz": [RATE_MIN, RATE_MAX],
            "depth_domain": [DEPTH_MIN, DEPTH_MAX],
            "waveforms": ["sine", "triangle"],
        }
