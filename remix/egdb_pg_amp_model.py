"""Wet-only, self-conditioned Amp+cab inverse for unseen profiles."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


RATE = 44_100
OUTPUT_PEAK_CONTRACT = 1.05


class _ConditionedBlock(nn.Module):
    def __init__(self, channels: int, condition_size: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels * 2, 9, padding=4 * dilation,
            dilation=dilation, groups=channels,
        )
        self.mix = nn.Conv1d(channels, channels, 1)
        self.film = nn.Linear(condition_size, channels * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, bias = self.film(condition).chunk(2, dim=1)
        left, gate = self.depthwise(value).chunk(2, dim=1)
        shaped = torch.tanh(left) * torch.sigmoid(gate)
        shaped = shaped * (1.0 + 0.25 * torch.tanh(scale).unsqueeze(-1))
        shaped = shaped + 0.25 * torch.tanh(bias).unsqueeze(-1)
        return value + 0.25 * self.mix(shaped)


class WetConditionedAmpCabInverse(nn.Module):
    """Infers its latent transfer profile from Wet audio, with no profile ID."""

    mechanism = "amp"

    def __init__(self, channels: int = 48, depth: int = 9, condition_size: int = 48) -> None:
        super().__init__()
        if channels < 16 or depth < 4 or condition_size < 16:
            raise ValueError("invalid Wet-conditioned Amp+cab geometry")
        self.channels = channels
        self.depth = depth
        self.condition_size = condition_size
        self.conditioner = nn.Sequential(
            nn.Conv1d(2, 16, 15, stride=4, padding=7), nn.GELU(),
            nn.Conv1d(16, 24, 15, stride=4, padding=7), nn.GELU(),
            nn.Conv1d(24, 32, 15, stride=4, padding=7), nn.GELU(),
            nn.Conv1d(32, condition_size, 15, stride=4, padding=7), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.stem = nn.Conv1d(6, channels, 15, padding=7)
        self.dilations = tuple(2 ** (index % 9) for index in range(depth))
        self.blocks = nn.ModuleList(
            _ConditionedBlock(channels, condition_size, dilation)
            for dilation in self.dilations
        )
        self.base_scale = nn.Linear(condition_size, 1)
        self.correction = nn.Conv1d(channels, 1, 1)
        self.uncertainty = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.base_scale.weight)
        # EGDB-PG presets commonly add 10--30 dB of level.  A 0.1 Wet scale is
        # a safer product prior than identity and still leaves polarity/gain
        # fully learnable from the Wet-derived condition vector.
        nn.init.constant_(self.base_scale.bias, math.atanh(0.05))
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        nn.init.zeros_(self.uncertainty.weight)
        nn.init.constant_(self.uncertainty.bias, -3.0)

    @staticmethod
    def _features(wet: torch.Tensor) -> torch.Tensor:
        mono = wet.unsqueeze(1)
        difference = F.pad(wet[:, 1:] - wet[:, :-1], (1, 0)).unsqueeze(1)
        envelope = mono.abs()
        return torch.cat((
            mono,
            envelope,
            difference,
            F.avg_pool1d(envelope, 65, 1, 32),
            F.avg_pool1d(envelope, 513, 1, 256),
            F.avg_pool1d(mono, 513, 1, 256),
        ), dim=1)

    def forward(
        self, wet: torch.Tensor, state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if state is not None:
            raise ValueError("offline Amp+cab expert accepts no graph or external state")
        if wet.ndim != 2 or not torch.isfinite(wet).all():
            raise ValueError("Wet-conditioned Amp+cab expects finite Wet [batch,time]")
        condition = self.conditioner(torch.stack((wet, wet.abs()), dim=1)).squeeze(-1)
        hidden = torch.tanh(self.stem(self._features(wet)))
        for block in self.blocks:
            hidden = block(hidden, condition)
        scale = 2.0 * torch.tanh(self.base_scale(condition))
        base = wet * scale
        correction = 0.75 * torch.tanh(self.correction(hidden).squeeze(1))
        restored = (base + correction).clamp(-OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT)
        uncertainty = F.softplus(self.uncertainty(hidden).squeeze(1)) + 1.0e-5
        return restored, uncertainty, None

    def manifest(self) -> dict:
        receptive_field = 15 + 8 * sum(self.dilations)
        return {
            "schema": 1,
            "architecture": "wet-self-conditioned-multiscale-dilated-tcn",
            "mechanism": "amp",
            "scope": "software Amp+cab preset to DI",
            "sample_rate": RATE,
            "channels": self.channels,
            "depth": self.depth,
            "condition_size": self.condition_size,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "local_receptive_field_frames": receptive_field,
            "conditioning_source": "Wet audio only",
            "profile_id_input": False,
            "gain_category_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
            "recurrent_state_input": False,
            "clean_or_oracle_input": False,
            "causal": False,
            "offline_chunk_conditioning": True,
            "output_peak_contract": OUTPUT_PEAK_CONTRACT,
        }
