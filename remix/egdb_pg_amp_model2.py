"""Wet-only spectral denoiser for profile-disjoint Amp+cab restoration."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .egdb_pg_amp_model import OUTPUT_PEAK_CONTRACT, RATE


class _SpectralBlock(nn.Module):
    def __init__(self, channels: int, condition_size: int, frequency_dilation: int, time_dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, (5, 3),
            padding=(2 * frequency_dilation, time_dilation),
            dilation=(frequency_dilation, time_dilation), groups=channels,
        )
        self.mix = nn.Conv2d(channels, channels * 2, 1)
        self.film = nn.Linear(condition_size, channels * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, bias = self.film(condition).chunk(2, dim=1)
        hidden = self.depthwise(value)
        hidden = hidden * (1.0 + 0.25 * torch.tanh(scale)[:, :, None, None])
        hidden = hidden + 0.25 * torch.tanh(bias)[:, :, None, None]
        left, gate = self.mix(F.gelu(hidden)).chunk(2, dim=1)
        return value + 0.25 * left * torch.sigmoid(gate)


class WetSpectralAmpCabInverse(nn.Module):
    """Predict a time-varying spectral inverse while preserving Wet phase."""

    mechanism = "amp"

    def __init__(self, channels: int = 32, depth: int = 8, condition_size: int = 32) -> None:
        super().__init__()
        if channels < 12 or depth < 4 or condition_size < 12:
            raise ValueError("invalid spectral Amp+cab geometry")
        self.channels = channels
        self.depth = depth
        self.condition_size = condition_size
        self.n_fft = 1024
        self.hop = 256
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.conditioner = nn.Sequential(
            nn.Conv1d(2, 16, 9, stride=2, padding=4), nn.GELU(),
            nn.Conv1d(16, 24, 9, stride=2, padding=4), nn.GELU(),
            nn.Conv1d(24, condition_size, 9, stride=2, padding=4), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.stem = nn.Conv2d(3, channels, (7, 5), padding=(3, 2))
        frequency_dilations = (1, 2, 4, 8)
        time_dilations = (1, 1, 2, 4)
        self.blocks = nn.ModuleList(
            _SpectralBlock(
                channels, condition_size,
                frequency_dilations[index % len(frequency_dilations)],
                time_dilations[index % len(time_dilations)],
            )
            for index in range(depth)
        )
        self.log_mask = nn.Conv2d(channels, 1, 1)
        self.uncertainty_logit = nn.Parameter(torch.tensor(-3.0))
        nn.init.zeros_(self.log_mask.weight)
        initial = math.log(0.1)
        nn.init.constant_(self.log_mask.bias, math.atanh(initial / 4.0))

    def forward(
        self, wet: torch.Tensor, state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if state is not None:
            raise ValueError("offline spectral Amp+cab expert accepts no graph or external state")
        if wet.ndim != 2 or not torch.isfinite(wet).all():
            raise ValueError("spectral Amp+cab expects finite Wet [batch,time]")
        spectrum = torch.stft(
            wet, self.n_fft, self.hop, window=self.window,
            center=True, pad_mode="constant", return_complex=True,
        )
        magnitude = spectrum.abs().clamp_min(1.0e-7)
        log_magnitude = torch.log1p(magnitude)
        mean = log_magnitude.mean(2)
        deviation = (log_magnitude - mean.unsqueeze(2)).square().mean(2).add(1.0e-8).sqrt()
        condition = self.conditioner(torch.stack((mean, deviation), dim=1)).squeeze(-1)
        unit = spectrum / magnitude
        features = torch.stack((log_magnitude, unit.real, unit.imag), dim=1)
        hidden = F.gelu(self.stem(features))
        for block in self.blocks:
            hidden = block(hidden, condition)
        log_mask = 4.0 * torch.tanh(self.log_mask(hidden).squeeze(1))
        restored_spectrum = spectrum * torch.exp(log_mask)
        restored = torch.istft(
            restored_spectrum, self.n_fft, self.hop, window=self.window,
            center=True, length=wet.shape[1],
        ).clamp(-OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT)
        uncertainty = F.softplus(self.uncertainty_logit).expand_as(restored) + 1.0e-5
        return restored, uncertainty, None

    def manifest(self) -> dict:
        return {
            "schema": 2,
            "architecture": "wet-self-conditioned-complex-stft-mask",
            "mechanism": "amp",
            "scope": "software Amp+cab preset to DI",
            "sample_rate": RATE,
            "channels": self.channels,
            "depth": self.depth,
            "condition_size": self.condition_size,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "n_fft": self.n_fft,
            "hop_frames": self.hop,
            "conditioning_source": "Wet log-spectrum statistics only",
            "phase_policy": "preserve Wet STFT phase",
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
