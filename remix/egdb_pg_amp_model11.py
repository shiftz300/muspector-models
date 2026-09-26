"""Frozen phase-aware gray-box Amp inverse plus explicit dynamics correction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .egdb_pg_amp_model import OUTPUT_PEAK_CONTRACT, RATE
from .egdb_pg_amp_model6 import _EnvelopeBlock
from .egdb_pg_amp_model10 import WetTonePhaseGrayBoxAmpCabInverse


class WetTonePhaseGrayBoxDynamicsAmpCabInverse(nn.Module):
    """Learn bounded local crest restoration over a frozen phase gray box."""

    mechanism = "amp"

    def __init__(
        self, channels: int = 24, depth: int = 8, condition_size: int = 64
    ) -> None:
        super().__init__()
        if channels < 12 or depth < 4 or condition_size < 16:
            raise ValueError("invalid phase gray-box dynamics geometry")
        self.channels = channels
        self.depth = depth
        self.condition_size = condition_size
        self.envelope_hop = 128
        self.envelope_window = 512
        self.base = WetTonePhaseGrayBoxAmpCabInverse(channels, depth, condition_size)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base_checkpoint_sha256 = None
        self.condition_projection = nn.Sequential(
            nn.Linear(64, condition_size), nn.GELU(),
            nn.Linear(condition_size, condition_size),
        )
        self.stem = nn.Conv1d(6, channels, 5, padding=2)
        self.blocks = nn.ModuleList(
            _EnvelopeBlock(channels, condition_size, 2 ** (index % 6))
            for index in range(depth)
        )
        self.controls = nn.Conv1d(channels, 2, 1)
        self.uncertainty_logit = nn.Parameter(torch.tensor(-3.0))
        nn.init.zeros_(self.controls.weight)
        nn.init.zeros_(self.controls.bias)

    def load_base(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        architecture = payload.get("architecture", {})
        if architecture.get("schema") != 12:
            raise ValueError("dynamics stage requires phase gray-box schema 12")
        expected = (
            architecture.get("depth"), architecture.get("condition_size")
        )
        if expected != (self.depth, self.condition_size):
            raise ValueError(f"phase gray-box geometry mismatch: {expected}")
        self.base.load_state_dict(payload["state_dict"])
        self.base.tone_encoder_sha256 = architecture.get("tone_encoder_sha256")
        self.base_checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def _envelope_features(
        self, wet: torch.Tensor, restored: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def statistics(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            framed = value.abs().unsqueeze(1)
            rms = F.avg_pool1d(
                framed.square(), self.envelope_window, self.envelope_hop,
                padding=self.envelope_window // 2,
            ).add(1.0e-8).sqrt()
            peak = F.max_pool1d(
                framed, self.envelope_window, self.envelope_hop,
                padding=self.envelope_window // 2,
            )
            crest = peak / rms.clamp_min(1.0e-4)
            return (
                torch.log(rms + 1.0e-5),
                torch.log(peak + 1.0e-5),
                torch.log(crest + 1.0e-4),
            )

        wet_stats = statistics(wet)
        restored_stats = statistics(restored)
        return torch.cat((*wet_stats, *restored_stats), dim=1), restored_stats[0].exp()

    def forward(
        self,
        wet: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        tone_reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if state is not None:
            raise ValueError("phase gray-box dynamics accepts no graph or external state")
        reference = wet if tone_reference is None else tone_reference
        with torch.no_grad():
            base_restored, _, _ = self.base(wet, tone_reference=reference)
            tone = self.base.tone_encoder(reference)
        features, restored_rms = self._envelope_features(wet, base_restored)
        condition = self.condition_projection(tone)
        hidden = F.gelu(self.stem(features))
        for block in self.blocks:
            hidden = block(hidden, condition)
        controls = F.interpolate(
            self.controls(hidden), size=wet.shape[1], mode="linear", align_corners=False
        )
        drive = 0.75 * torch.tanh(controls[:, 0])
        level = 0.50 * torch.tanh(controls[:, 1])
        local_rms = F.interpolate(
            restored_rms, size=wet.shape[1], mode="linear", align_corners=False
        ).squeeze(1)
        relative_level = base_restored.abs() / local_rms.clamp_min(1.0e-4)
        crest_shape = torch.tanh(1.5 * (relative_level - 1.0))
        gain = torch.exp(level + drive * crest_shape)
        restored = (base_restored * gain).clamp(
            -OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT
        )
        uncertainty = F.softplus(self.uncertainty_logit).expand_as(restored) + 1.0e-5
        return restored, uncertainty, None

    def manifest(self) -> dict:
        return {
            "schema": 13,
            "architecture": "frozen-phase-gray-box-plus-explicit-envelope-compander",
            "mechanism": "amp",
            "scope": "software Amp+cab preset to DI",
            "sample_rate": RATE,
            "channels": self.channels,
            "depth": self.depth,
            "condition_size": self.condition_size,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "base_phase_gray_box_frozen": True,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "dynamics_policy": "bounded local crest companding and smooth level correction",
            "phase_policy": "generated by frozen non-symmetric gray-box LTI stages",
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
