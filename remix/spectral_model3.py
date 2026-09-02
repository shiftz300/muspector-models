"""Physics-first independent inverse for the generic Spectral/EQ family."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .foundation_data import RATE
from .product2 import CONTROL_WIDTH
from .spectral3 import GAIN_DB, MAXIMUM_CURVE_DB


MAXIMUM_INVERSE_GAIN_DB = 12.0
OUTPUT_PEAK_CONTRACT = 0.98


def _response(controls: torch.Tensor, frames: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    frequency = torch.fft.rfftfreq(frames, 1.0 / RATE, device=device, dtype=dtype).clamp_min(20.0)
    logf = torch.log2(frequency)[None, :]
    low = torch.sigmoid((math.log2(320.0) - logf) * 5.0)
    high = torch.sigmoid((logf - math.log2(3600.0)) * 5.0)
    low_gain = (controls[:, 0:1] * 2.0 - 1.0) * GAIN_DB
    mid_gain = (controls[:, 1:2] * 2.0 - 1.0) * GAIN_DB
    center = 250.0 * torch.exp(controls[:, 2:3] * math.log(5200.0 / 250.0))
    q = 0.45 * torch.exp(controls[:, 3:4] * math.log(3.5 / 0.45))
    high_gain = (controls[:, 4:5] * 2.0 - 1.0) * GAIN_DB
    width = 1.20 / torch.sqrt(q)
    mid = torch.exp(-0.5 * ((logf - torch.log2(center)) / width).square())
    curve = low_gain * low + mid_gain * mid + high_gain * high
    scale = torch.clamp(MAXIMUM_CURVE_DB / curve.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-8), max=1.0)
    return torch.pow(10.0, curve * scale / 20.0)


def bounded_inverse_base(wet: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
    """Invert known EQ magnitude while enforcing an explicit gain ceiling."""
    if wet.ndim != 2 or controls.shape != (wet.shape[0], CONTROL_WIDTH):
        raise ValueError("spectral inverse expects wet [batch,time] and controls [batch,5]")
    if not torch.isfinite(wet).all() or not torch.isfinite(controls).all():
        raise ValueError("spectral inverse input must be finite")
    response = _response(controls, wet.shape[1], wet.dtype, wet.device)
    maximum = 10.0 ** (MAXIMUM_INVERSE_GAIN_DB / 20.0)
    inverse = torch.clamp(response.reciprocal(), max=maximum)
    return torch.fft.irfft(torch.fft.rfft(wet) * inverse, n=wet.shape[1])


class SpectralInverseV3(nn.Module):
    """Bounded analytic EQ inverse with a small optional local residual."""

    mechanism = "spectral"

    def __init__(self, channels: int = 12, depth: int = 5) -> None:
        super().__init__()
        if channels < 4 or depth < 1:
            raise ValueError("invalid spectral expert geometry")
        self.channels = channels
        self.depth = depth
        self.dilations = tuple(2**index for index in range(depth))
        self.input = nn.Conv1d(4 + CONTROL_WIDTH, channels, 1)
        self.temporal = nn.ModuleList(
            nn.Conv1d(channels, channels * 2, 5, padding=2 * dilation, dilation=dilation)
            for dilation in self.dilations
        )
        self.mix = nn.ModuleList(nn.Conv1d(channels, channels, 1) for _ in self.dilations)
        self.correction = nn.Conv1d(channels, 1, 1)
        self.uncertainty = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        nn.init.zeros_(self.uncertainty.weight)
        nn.init.constant_(self.uncertainty.bias, -5.0)

    def forward(self, wet: torch.Tensor, controls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = bounded_inverse_base(wet, controls)
        condition = controls.mul(2.0).sub(1.0).unsqueeze(-1).expand(-1, -1, wet.shape[1])
        difference = F.pad(base[:, 1:] - base[:, :-1], (1, 0))
        features = torch.cat(
            (wet[:, None], wet.abs()[:, None], base[:, None], difference[:, None], condition), dim=1
        )
        hidden = torch.tanh(self.input(features))
        for temporal, mix in zip(self.temporal, self.mix, strict=True):
            value, gate = temporal(hidden).chunk(2, dim=1)
            hidden = hidden + 0.25 * mix(torch.tanh(value) * torch.sigmoid(gate))
        limit = torch.minimum(torch.full_like(base, 0.05), (OUTPUT_PEAK_CONTRACT - base.abs()).clamp_min(0.0))
        correction = limit * torch.tanh(self.correction(hidden).squeeze(1))
        uncertainty = F.softplus(self.uncertainty(hidden).squeeze(1)) + 1.0e-6
        return base + correction, uncertainty

    def manifest(self) -> dict:
        return {
            "schema": 1,
            "architecture": "bounded-analytic-three-band-eq-inverse-plus-local-tcn",
            "mechanism": self.mechanism,
            "channels": self.channels,
            "depth": self.depth,
            "control_width": CONTROL_WIDTH,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "sample_rate": RATE,
            "normalization_across_time": False,
            "causal": False,
            "bounded_context": True,
            "maximum_inverse_gain_db": MAXIMUM_INVERSE_GAIN_DB,
            "output_peak_contract": OUTPUT_PEAK_CONTRACT,
            "uncertainty_output": True,
            "graph_order_input": False,
            "neighbouring_effect_input": False,
            "physical_device_claim": False,
        }
