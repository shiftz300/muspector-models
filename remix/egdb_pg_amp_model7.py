"""Frozen complex restoration plus a long-context Wet-only waveform refiner."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .egdb_pg_amp_model import OUTPUT_PEAK_CONTRACT, RATE, _ConditionedBlock
from .egdb_pg_amp_model5 import WetToneComplexAmpCabInverse


class WetToneComplexTemporalAmpCabInverse(nn.Module):
    """Refine frozen magnitude/phase recovery with long-context waveform cues."""

    mechanism = "amp"

    def __init__(self, channels: int = 24, depth: int = 6, condition_size: int = 64) -> None:
        super().__init__()
        if channels < 12 or depth < 4 or condition_size < 16:
            raise ValueError("invalid complex temporal Amp+cab geometry")
        self.channels = channels
        self.depth = depth
        self.condition_size = condition_size
        self.temporal_depth = max(12, depth + 6)
        self.base = WetToneComplexAmpCabInverse(channels, depth, condition_size)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base_checkpoint_sha256 = None
        self.condition_projection = nn.Sequential(
            nn.Linear(64, condition_size), nn.GELU(),
            nn.Linear(condition_size, condition_size),
        )
        self.stem = nn.Conv1d(8, channels, 15, padding=7)
        self.dilations = tuple(2 ** index for index in range(self.temporal_depth))
        self.blocks = nn.ModuleList(
            _ConditionedBlock(channels, condition_size, dilation)
            for dilation in self.dilations
        )
        self.log_gain = nn.Conv1d(channels, 1, 1)
        self.correction = nn.Conv1d(channels, 1, 1)
        self.uncertainty = nn.Conv1d(channels, 1, 1)
        for layer in (self.log_gain, self.correction):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.uncertainty.weight)
        nn.init.constant_(self.uncertainty.bias, -3.0)

    def load_base(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        architecture = payload.get("architecture", {})
        if architecture.get("schema") != 5:
            raise ValueError("temporal stage requires a complex model schema 5 checkpoint")
        expected = (
            architecture.get("channels"), architecture.get("depth"),
            architecture.get("condition_size"),
        )
        if expected != (self.channels, self.depth, self.condition_size):
            raise ValueError(f"complex base geometry mismatch: {expected}")
        self.base.load_state_dict(payload["state_dict"])
        self.base.tone_encoder_sha256 = architecture.get("tone_encoder_sha256")
        self.base_checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    @staticmethod
    def _features(wet: torch.Tensor, restored: torch.Tensor) -> torch.Tensor:
        wet_diff = F.pad(wet[:, 1:] - wet[:, :-1], (1, 0))
        restored_diff = F.pad(restored[:, 1:] - restored[:, :-1], (1, 0))
        envelope = F.avg_pool1d(
            restored.abs().unsqueeze(1), 129, 1, 64
        ).squeeze(1)
        slow = F.avg_pool1d(
            restored.abs().unsqueeze(1), 2049, 1, 1024
        ).squeeze(1)
        return torch.stack((
            wet, wet.abs(), wet_diff,
            restored, restored_diff, envelope, slow,
            wet - restored,
        ), dim=1)

    def forward(
        self, wet: torch.Tensor, state: torch.Tensor | None = None,
        *, tone_reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if state is not None:
            raise ValueError("offline temporal Amp+cab expert accepts no graph or external state")
        reference = wet if tone_reference is None else tone_reference
        with torch.no_grad():
            base_restored, _, _ = self.base(wet, tone_reference=reference)
            tone = self.base.tone_encoder(reference)
        condition = self.condition_projection(tone)
        hidden = torch.tanh(self.stem(self._features(wet, base_restored)))
        for block in self.blocks:
            hidden = block(hidden, condition)
        log_gain = torch.tanh(self.log_gain(hidden).squeeze(1))
        local_scale = F.avg_pool1d(
            base_restored.abs().unsqueeze(1), 129, 1, 64
        ).squeeze(1).clamp_min(1.0e-4)
        correction = 0.75 * local_scale * torch.tanh(
            self.correction(hidden).squeeze(1)
        )
        restored = (base_restored * torch.exp(log_gain) + correction).clamp(
            -OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT
        )
        uncertainty = F.softplus(self.uncertainty(hidden).squeeze(1)) + 1.0e-5
        return restored, uncertainty, None

    def manifest(self) -> dict:
        receptive_field = 15 + 8 * sum(self.dilations)
        return {
            "schema": 7,
            "architecture": "frozen-complex-inverse-plus-long-context-waveform-tcn",
            "mechanism": "amp",
            "scope": "software Amp+cab preset to DI",
            "sample_rate": RATE,
            "channels": self.channels,
            "depth": self.depth,
            "temporal_depth": self.temporal_depth,
            "condition_size": self.condition_size,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "temporal_receptive_field_frames": receptive_field,
            "temporal_receptive_field_seconds": receptive_field / RATE,
            "base_complex_inverse_frozen": True,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "dynamics_policy": "bounded local gain plus waveform residual",
            "phase_policy": "generated by frozen complex base then waveform-refined",
            "tone_encoder_frozen": True,
            "tone_encoder_sha256": self.base.tone_encoder_sha256,
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
