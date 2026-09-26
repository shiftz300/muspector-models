"""Wet-conditioned forward Amp verifier with explicit anti-leakage contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .egdb_pg_amp_data import RATE
from .egdb_pg_tone_encoder import WetToneEncoder


class _ConditionedResidualBlock(nn.Module):
    def __init__(self, channels: int, condition_size: int, dilation: int) -> None:
        super().__init__()
        self.filter = nn.Conv1d(
            channels, 2 * channels, 5, padding=2 * dilation, dilation=dilation
        )
        self.condition = nn.Linear(condition_size, 2 * channels)
        self.output = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        features = self.filter(value) + self.condition(condition).unsqueeze(-1)
        gate, candidate = features.chunk(2, dim=1)
        update = torch.sigmoid(gate) * torch.tanh(candidate)
        return value + self.output(update)


class WetConditionedAmpForwardVerifier(nn.Module):
    """Replay a candidate Clean through the effect inferred from current Wet.

    The Wet observation is reduced to one global tone embedding. It never enters
    the sample-rate path, so a zero or wrong candidate cannot be copied from Wet.
    This structural rule is still verified empirically before the model may score
    an inverse candidate.
    """

    mechanism = "amp"

    def __init__(self, channels: int = 32, depth: int = 10, condition_size: int = 64) -> None:
        super().__init__()
        if channels < 8 or depth < 4 or condition_size < 16:
            raise ValueError("invalid Amp forward-verifier geometry")
        self.channels = channels
        self.depth = depth
        self.condition_size = condition_size
        self.tone_encoder = WetToneEncoder(64)
        for parameter in self.tone_encoder.parameters():
            parameter.requires_grad_(False)
        self.tone_encoder_sha256: str | None = None
        self.condition = nn.Sequential(
            nn.Linear(64, condition_size), nn.GELU(),
            nn.Linear(condition_size, condition_size), nn.Tanh(),
        )
        self.input = nn.Conv1d(1, channels, 15, padding=7)
        self.blocks = nn.ModuleList(
            _ConditionedResidualBlock(channels, condition_size, 2 ** (index % 10))
            for index in range(depth)
        )
        self.output = nn.Conv1d(channels, 1, 15, padding=7)
        self.direct_gain = nn.Linear(condition_size, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.direct_gain.weight)
        nn.init.constant_(self.direct_gain.bias, 0.2554128119)  # 4*tanh(bias) = 1

    def load_tone_encoder(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        architecture = payload.get("architecture", {})
        if architecture.get("architecture") != "wet-log-spectrum-content-invariant-tone-encoder":
            raise ValueError("checkpoint is not an admitted Wet tone encoder")
        self.tone_encoder.load_state_dict(payload["state_dict"])
        self.tone_encoder_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def forward(
        self, candidate_clean: torch.Tensor, observed_wet: torch.Tensor,
        *, tone_reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if candidate_clean.ndim != 2 or observed_wet.ndim != 2:
            raise ValueError("forward verifier expects [batch,time] audio")
        if candidate_clean.shape != observed_wet.shape:
            raise ValueError("candidate Clean and observed Wet must share geometry")
        if not torch.isfinite(candidate_clean).all() or not torch.isfinite(observed_wet).all():
            raise ValueError("forward verifier expects finite audio")
        reference = observed_wet if tone_reference is None else tone_reference
        if reference.ndim != 2 or reference.shape[0] != candidate_clean.shape[0]:
            raise ValueError("Wet reference must be batch aligned")
        with torch.no_grad():
            tone = self.tone_encoder(reference)
        condition = self.condition(tone)
        value = self.input(candidate_clean.unsqueeze(1))
        for block in self.blocks:
            value = block(value, condition)
        residual = self.output(torch.tanh(value)).squeeze(1)
        gain = 4.0 * torch.tanh(self.direct_gain(condition))
        return gain * candidate_clean + residual

    def manifest(self) -> dict:
        receptive_field = 15 + 4 * sum(2 ** (index % 10) for index in range(self.depth)) + 14
        return {
            "schema": 18,
            "architecture": "wet-conditioned-global-latent-forward-amp-verifier",
            "sample_rate": RATE,
            "channels": self.channels,
            "depth": self.depth,
            "condition_size": self.condition_size,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "receptive_field_frames": receptive_field,
            "receptive_field_seconds": receptive_field / RATE,
            "tone_encoder_frozen": True,
            "tone_encoder_sha256": self.tone_encoder_sha256,
            "sample_rate_wet_path": False,
            "candidate_clean_input": True,
            "observed_wet_global_embedding_input": True,
            "profile_id_input": False,
            "gain_category_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
            "clean_or_oracle_required_at_inverse_inference": False,
            "purpose": "candidate verification and abstention; not an inverse model",
        }
