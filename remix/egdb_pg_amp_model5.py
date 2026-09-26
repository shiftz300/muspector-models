"""Phase-generating Wet-only Amp+cab inverse with a frozen tone encoder."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .egdb_pg_amp_model import OUTPUT_PEAK_CONTRACT, RATE
from .egdb_pg_tone_encoder import WetToneEncoder


class _MagnitudePhaseBlock(nn.Module):
    def __init__(self, channels: int, condition_size: int, frequency_dilation: int, time_dilation: int) -> None:
        super().__init__()
        geometry = {
            "kernel_size": (5, 3),
            "padding": (2 * frequency_dilation, time_dilation),
            "dilation": (frequency_dilation, time_dilation),
            "groups": channels,
        }
        self.magnitude_depthwise = nn.Conv2d(channels, channels, **geometry)
        self.phase_depthwise = nn.Conv2d(channels, channels, **geometry)
        self.magnitude_mix = nn.Conv2d(channels * 2, channels, 1)
        self.phase_mix = nn.Conv2d(channels * 2, channels, 1)
        self.magnitude_film = nn.Linear(condition_size, channels * 2)
        self.phase_film = nn.Linear(condition_size, channels * 2)
        for film in (self.magnitude_film, self.phase_film):
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

    @staticmethod
    def _film(value: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        scale, bias = parameters.chunk(2, dim=1)
        return value * (1.0 + 0.25 * torch.tanh(scale)[:, :, None, None]) + 0.25 * torch.tanh(bias)[:, :, None, None]

    def forward(
        self, magnitude: torch.Tensor, phase: torch.Tensor, condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        magnitude_local = self._film(self.magnitude_depthwise(magnitude), self.magnitude_film(condition))
        phase_local = self._film(self.phase_depthwise(phase), self.phase_film(condition))
        magnitude_update = self.magnitude_mix(F.gelu(torch.cat((magnitude_local, phase_local), dim=1)))
        phase_update = self.phase_mix(F.gelu(torch.cat((phase_local, magnitude_local), dim=1)))
        return magnitude + 0.20 * magnitude_update, phase + 0.20 * phase_update


class WetToneComplexAmpCabInverse(nn.Module):
    """Estimate clean magnitude and phase jointly from Wet audio only."""

    mechanism = "amp"

    def __init__(self, channels: int = 24, depth: int = 6, condition_size: int = 64) -> None:
        super().__init__()
        if channels < 12 or depth < 4 or condition_size < 16:
            raise ValueError("invalid complex Amp+cab geometry")
        self.channels = channels
        self.depth = depth
        self.condition_size = condition_size
        self.n_fft = 512
        self.hop = 128
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.tone_encoder = WetToneEncoder(64)
        for parameter in self.tone_encoder.parameters():
            parameter.requires_grad_(False)
        self.tone_encoder_sha256 = None
        self.condition_projection = nn.Sequential(
            nn.Linear(64, condition_size), nn.GELU(),
            nn.Linear(condition_size, condition_size),
        )
        self.magnitude_stem = nn.Conv2d(1, channels, (7, 5), padding=(3, 2))
        self.phase_stem = nn.Conv2d(2, channels, (7, 5), padding=(3, 2))
        frequency_dilations = (1, 2, 4, 8)
        time_dilations = (1, 1, 2, 4)
        self.blocks = nn.ModuleList(
            _MagnitudePhaseBlock(
                channels, condition_size,
                frequency_dilations[index % len(frequency_dilations)],
                time_dilations[index % len(time_dilations)],
            )
            for index in range(depth)
        )
        self.log_magnitude_mask = nn.Conv2d(channels, 1, 1)
        self.phase_delta = nn.Conv2d(channels, 1, 1)
        self.uncertainty_logit = nn.Parameter(torch.tensor(-3.0))
        nn.init.zeros_(self.log_magnitude_mask.weight)
        nn.init.constant_(self.log_magnitude_mask.bias, math.atanh(math.log(0.1) / 4.0))
        nn.init.zeros_(self.phase_delta.weight)
        nn.init.zeros_(self.phase_delta.bias)

    def load_tone_encoder(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        architecture = payload.get("architecture", {})
        if architecture.get("architecture") != "wet-log-spectrum-content-invariant-tone-encoder":
            raise ValueError("checkpoint is not an admitted Wet tone encoder")
        self.tone_encoder.load_state_dict(payload["state_dict"])
        self.tone_encoder_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def forward(
        self, wet: torch.Tensor, state: torch.Tensor | None = None,
        *, tone_reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if state is not None:
            raise ValueError("offline complex Amp+cab expert accepts no graph or external state")
        reference = wet if tone_reference is None else tone_reference
        if wet.ndim != 2 or reference.ndim != 2 or wet.shape[0] != reference.shape[0]:
            raise ValueError("complex Amp+cab expects batch-aligned Wet audio")
        if not torch.isfinite(wet).all() or not torch.isfinite(reference).all():
            raise ValueError("complex Amp+cab expects finite Wet audio")
        with torch.no_grad():
            tone = self.tone_encoder(reference)
        condition = self.condition_projection(tone)
        spectrum = torch.stft(
            wet, self.n_fft, self.hop, window=self.window,
            center=True, pad_mode="constant", return_complex=True,
        )
        magnitude = spectrum.abs().clamp_min(1.0e-7)
        unit = spectrum / magnitude
        magnitude_hidden = F.gelu(self.magnitude_stem(torch.log1p(magnitude).unsqueeze(1)))
        phase_hidden = F.gelu(self.phase_stem(torch.stack((unit.real, unit.imag), dim=1)))
        for block in self.blocks:
            magnitude_hidden, phase_hidden = block(magnitude_hidden, phase_hidden, condition)
        log_mask = 4.0 * torch.tanh(self.log_magnitude_mask(magnitude_hidden).squeeze(1))
        delta = torch.pi * torch.tanh(self.phase_delta(phase_hidden).squeeze(1))
        rotation = torch.complex(torch.cos(delta), torch.sin(delta))
        restored_spectrum = spectrum * torch.exp(log_mask) * rotation
        restored = torch.istft(
            restored_spectrum, self.n_fft, self.hop, window=self.window,
            center=True, length=wet.shape[1],
        ).clamp(-OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT)
        uncertainty = F.softplus(self.uncertainty_logit).expand_as(restored) + 1.0e-5
        return restored, uncertainty, None

    def manifest(self) -> dict:
        return {
            "schema": 5,
            "architecture": "frozen-wet-tone-encoder-plus-magnitude-phase-complex-mask",
            "mechanism": "amp",
            "scope": "software Amp+cab preset to DI",
            "sample_rate": RATE,
            "channels": self.channels,
            "depth": self.depth,
            "condition_size": self.condition_size,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "n_fft": self.n_fft,
            "hop_frames": self.hop,
            "phase_policy": "predict bounded phase delta; do not preserve Wet phase",
            "magnitude_phase_cross_stream": True,
            "tone_encoder_frozen": True,
            "tone_encoder_sha256": self.tone_encoder_sha256,
            "tone_reference_source": "another 3-second span of the same Wet recording",
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
