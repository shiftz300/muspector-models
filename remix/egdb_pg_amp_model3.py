"""Two-stage Wet-only spectral plus temporal Amp+cab inverse."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .egdb_pg_amp_model import OUTPUT_PEAK_CONTRACT, RATE, _ConditionedBlock
from .egdb_pg_amp_model2 import WetSpectralAmpCabInverse


class WetHybridAmpCabInverse(nn.Module):
    """Restore tone first, then refine transient shape and sample-wise gain."""

    mechanism = "amp"

    def __init__(self, channels: int = 48, depth: int = 9, condition_size: int = 48) -> None:
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.condition_size = condition_size
        self.spectral = WetSpectralAmpCabInverse(channels, depth, condition_size)
        self.conditioner = nn.Sequential(
            nn.Conv1d(2, 16, 15, stride=4, padding=7), nn.GELU(),
            nn.Conv1d(16, 24, 15, stride=4, padding=7), nn.GELU(),
            nn.Conv1d(24, condition_size, 15, stride=4, padding=7), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.stem = nn.Conv1d(6, channels, 15, padding=7)
        self.dilations = tuple(2 ** (index % 9) for index in range(depth))
        self.blocks = nn.ModuleList(
            _ConditionedBlock(channels, condition_size, dilation)
            for dilation in self.dilations
        )
        self.log_gain = nn.Conv1d(channels, 1, 1)
        self.correction = nn.Conv1d(channels, 1, 1)
        self.uncertainty = nn.Conv1d(channels, 1, 1)
        for layer in (self.log_gain, self.correction, self.uncertainty):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.uncertainty.bias, -3.0)

    @staticmethod
    def _features(wet: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        spectral_diff = F.pad(spectral[:, 1:] - spectral[:, :-1], (1, 0))
        envelope = F.avg_pool1d(spectral.abs().unsqueeze(1), 129, 1, 64).squeeze(1)
        slow = F.avg_pool1d(spectral.abs().unsqueeze(1), 1025, 1, 512).squeeze(1)
        return torch.stack((wet, wet.abs(), spectral, spectral_diff, envelope, slow), dim=1)

    def forward(
        self, wet: torch.Tensor, state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if state is not None:
            raise ValueError("offline hybrid Amp+cab expert accepts no graph or external state")
        if wet.ndim != 2 or not torch.isfinite(wet).all():
            raise ValueError("hybrid Amp+cab expects finite Wet [batch,time]")
        spectral, _, _ = self.spectral(wet)
        condition = self.conditioner(torch.stack((wet, wet.abs()), dim=1)).squeeze(-1)
        hidden = torch.tanh(self.stem(self._features(wet, spectral)))
        for block in self.blocks:
            hidden = block(hidden, condition)
        log_gain = 2.0 * torch.tanh(self.log_gain(hidden).squeeze(1))
        local_scale = F.avg_pool1d(
            spectral.abs().unsqueeze(1), 129, 1, 64
        ).squeeze(1).clamp_min(1.0e-4)
        correction = 0.50 * local_scale * torch.tanh(self.correction(hidden).squeeze(1))
        restored = (spectral * torch.exp(log_gain) + correction).clamp(
            -OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT
        )
        uncertainty = F.softplus(self.uncertainty(hidden).squeeze(1)) + 1.0e-5
        return restored, uncertainty, None

    def manifest(self) -> dict:
        return {
            "schema": 3,
            "architecture": "wet-self-conditioned-spectral-mask-plus-temporal-refiner",
            "mechanism": "amp",
            "scope": "software Amp+cab preset to DI",
            "sample_rate": RATE,
            "channels": self.channels,
            "depth_per_stage": self.depth,
            "condition_size": self.condition_size,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "stages": ["Wet spectral restoration", "Wet-conditioned temporal refinement"],
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
