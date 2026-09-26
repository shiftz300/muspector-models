"""Wet-only Amp+cab inverse conditioned by a frozen content-invariant tone encoder."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .egdb_pg_amp_model import OUTPUT_PEAK_CONTRACT, RATE, _ConditionedBlock
from .egdb_pg_tone_encoder import WetToneEncoder


class WetToneConditionedAmpCabInverse(nn.Module):
    """Decode a Wet chunk using tone inferred from another span of the same Wet recording."""

    mechanism = "amp"

    def __init__(self, channels: int = 40, depth: int = 11, condition_size: int = 64) -> None:
        super().__init__()
        if channels < 16 or depth < 4 or condition_size < 16:
            raise ValueError("invalid tone-conditioned Amp+cab geometry")
        self.channels = channels
        self.depth = depth
        self.condition_size = condition_size
        self.tone_encoder = WetToneEncoder(64)
        for parameter in self.tone_encoder.parameters():
            parameter.requires_grad_(False)
        self.tone_encoder_sha256 = None
        self.condition_projection = nn.Sequential(
            nn.Linear(64, condition_size), nn.GELU(),
            nn.Linear(condition_size, condition_size),
        )
        self.stem = nn.Conv1d(6, channels, 15, padding=7)
        self.dilations = tuple(2 ** (index % 10) for index in range(depth))
        self.blocks = nn.ModuleList(
            _ConditionedBlock(channels, condition_size, dilation)
            for dilation in self.dilations
        )
        self.base_scale = nn.Linear(condition_size, 1)
        self.log_gain = nn.Conv1d(channels, 1, 1)
        self.correction = nn.Conv1d(channels, 1, 1)
        self.uncertainty = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.base_scale.weight)
        nn.init.constant_(self.base_scale.bias, math.atanh(0.05))
        for layer in (self.log_gain, self.correction, self.uncertainty):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.uncertainty.bias, -3.0)

    def load_tone_encoder(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        architecture = payload.get("architecture", {})
        if architecture.get("architecture") != "wet-log-spectrum-content-invariant-tone-encoder":
            raise ValueError("checkpoint is not an admitted Wet tone encoder")
        self.tone_encoder.load_state_dict(payload["state_dict"])
        self.tone_encoder_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    @staticmethod
    def _features(wet: torch.Tensor) -> torch.Tensor:
        mono = wet.unsqueeze(1)
        difference = F.pad(wet[:, 1:] - wet[:, :-1], (1, 0)).unsqueeze(1)
        envelope = mono.abs()
        return torch.cat((
            mono, envelope, difference,
            F.avg_pool1d(envelope, 65, 1, 32),
            F.avg_pool1d(envelope, 513, 1, 256),
            F.avg_pool1d(mono, 513, 1, 256),
        ), dim=1)

    def forward(
        self, wet: torch.Tensor, state: torch.Tensor | None = None,
        *, tone_reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if state is not None:
            raise ValueError("offline tone-conditioned expert accepts no graph or external state")
        reference = wet if tone_reference is None else tone_reference
        if wet.ndim != 2 or reference.ndim != 2:
            raise ValueError("tone-conditioned Amp+cab expects Wet [batch,time]")
        if wet.shape[0] != reference.shape[0] or not torch.isfinite(wet).all() or not torch.isfinite(reference).all():
            raise ValueError("Wet chunk and same-recording tone reference must be finite and batch-aligned")
        with torch.no_grad():
            tone = self.tone_encoder(reference)
        condition = self.condition_projection(tone)
        hidden = torch.tanh(self.stem(self._features(wet)))
        for block in self.blocks:
            hidden = block(hidden, condition)
        scale = 2.0 * torch.tanh(self.base_scale(condition))
        log_gain = 2.0 * torch.tanh(self.log_gain(hidden).squeeze(1))
        local_scale = F.avg_pool1d(wet.abs().unsqueeze(1), 129, 1, 64).squeeze(1).clamp_min(1.0e-4)
        correction = 0.50 * local_scale * torch.tanh(self.correction(hidden).squeeze(1))
        restored = (wet * scale * torch.exp(log_gain) + correction).clamp(
            -OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT
        )
        uncertainty = F.softplus(self.uncertainty(hidden).squeeze(1)) + 1.0e-5
        return restored, uncertainty, None

    def manifest(self) -> dict:
        return {
            "schema": 4,
            "architecture": "frozen-wet-tone-encoder-plus-film-gcn-inverse",
            "mechanism": "amp",
            "scope": "software Amp+cab preset to DI",
            "sample_rate": RATE,
            "channels": self.channels,
            "depth": self.depth,
            "condition_size": self.condition_size,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
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
