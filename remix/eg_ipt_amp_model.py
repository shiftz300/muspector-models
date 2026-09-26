"""Fixed-device eight-stage Amp inverse for the EG-IPT physical chain."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .egdb_pg_amp_model import OUTPUT_PEAK_CONTRACT, RATE
from .open_riff_box_stage_model import (
    _LinearInverse,
    _LinearNonlinearInverse,
    _NonlinearInverse,
)


class EgIptFixedProfileAmpInverse(nn.Module):
    """One effect profile per model; no content-dependent profile inference."""

    mechanism = "amp"

    def __init__(self, depth: int = 8, condition_size: int = 64) -> None:
        super().__init__()
        if depth < 4 or condition_size < 16:
            raise ValueError("invalid fixed-profile Amp geometry")
        self.depth = depth
        self.condition_size = condition_size
        self.segments = max(8, depth * 2)
        self.device_condition = nn.Parameter(torch.zeros(condition_size))
        self.inverse_stages = nn.ModuleList((
            _LinearInverse(condition_size, 129, 1.0),
            _LinearNonlinearInverse(condition_size, 65, self.segments),
            _NonlinearInverse(condition_size, self.segments),
            _NonlinearInverse(condition_size, self.segments),
            _LinearInverse(condition_size, 129, 1.0),
            _LinearInverse(condition_size, 257, 1.0),
            _LinearNonlinearInverse(condition_size, 65, self.segments),
            _LinearInverse(condition_size, 129, 1.0),
        ))
        self.pretrain_sha256: str | None = None

    def load_stage_pretrain(
        self, checkpoint: Path, reference_wet: torch.Tensor
    ) -> None:
        from .rusty_amp_stage_model import RustyAmpStagewiseGrayBoxInverse

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        architecture = payload.get("architecture", {})
        if architecture.get("schema") != 17:
            raise ValueError("fixed-profile Amp requires schema-17 stage pretraining")
        source = RustyAmpStagewiseGrayBoxInverse(
            24, architecture.get("depth", 8), architecture.get("condition_size", 64)
        )
        source.load_state_dict(payload["state_dict"], strict=True)
        self.inverse_stages.load_state_dict(source.inverse_stages.state_dict(), strict=True)
        source.eval()
        with torch.inference_mode():
            condition = source.condition(source.tone_encoder(reference_wet)).mean(0)
        if condition.shape != self.device_condition.shape:
            raise ValueError("stage-pretrain condition geometry changed")
        with torch.no_grad():
            self.device_condition.copy_(condition)
        self.pretrain_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def stage_outputs(self, wet: torch.Tensor) -> list[torch.Tensor]:
        if wet.ndim != 2 or not torch.isfinite(wet).all():
            raise ValueError("fixed-profile Amp expects finite Wet [batch,time]")
        condition = self.device_condition[None].expand(wet.shape[0], -1)
        value = wet
        outputs = []
        for stage in self.inverse_stages:
            value = stage(value, condition)
            outputs.append(value)
        return outputs

    def forward(
        self,
        wet: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if state is not None:
            raise ValueError("fixed-profile Amp accepts no graph or external state")
        restored = self.stage_outputs(wet)[-1].clamp(
            -OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT
        )
        uncertainty = F.softplus(restored.new_tensor(-3.0)).expand_as(restored) + 1.0e-5
        return restored, uncertainty, None

    def manifest(self) -> dict:
        return {
            "schema": 19,
            "architecture": "fixed-device-eight-stage-gray-box-amp-inverse",
            "mechanism": "amp",
            "sample_rate": RATE,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "depth": self.depth,
            "condition_size": self.condition_size,
            "segments": self.segments,
            "device_profile": "EVH 5150 III 50W 6L6 + Mesa 4x12 V30 + close SM57",
            "device_profile_baked_into_weights": True,
            "content_dependent_profile_estimation": False,
            "stage_pretrain_sha256": self.pretrain_sha256,
            "internal_reverse_stages": [
                "output-and-presence", "speaker-load", "output-transformer",
                "power-amp", "voice-balance", "tone-stack", "preamp", "front-end",
            ],
            "profile_id_input": False,
            "gain_category_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
            "recurrent_state_input": False,
            "clean_or_oracle_input": False,
            "wet_only_inference": True,
            "causal": False,
            "output_peak_contract": OUTPUT_PEAK_CONTRACT,
        }


class _TransientBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.filter = nn.Conv1d(
            channels, 2 * channels, 5, padding=2 * dilation, dilation=dilation
        )
        self.output = nn.Conv1d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, candidate = self.filter(value).chunk(2, dim=1)
        return value + self.output(torch.sigmoid(gate) * torch.tanh(candidate))


class EgIptFixedProfileAmpTransientInverse(nn.Module):
    """Frozen v30 physical inverse plus a short-memory transient correction."""

    mechanism = "amp"

    def __init__(self, channels: int = 16, depth: int = 8) -> None:
        super().__init__()
        if channels < 8 or not 4 <= depth <= 9:
            raise ValueError("invalid fixed-profile transient geometry")
        self.base = EgIptFixedProfileAmpInverse()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.channels = channels
        self.depth = depth
        self.input = nn.Conv1d(4, channels, 9, padding=4)
        self.blocks = nn.ModuleList(
            _TransientBlock(channels, 2**index) for index in range(depth)
        )
        self.output = nn.Conv1d(channels, 1, 9, padding=4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.base_checkpoint_sha256: str | None = None

    def load_base(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("architecture", {}).get("schema") != 19:
            raise ValueError("transient refiner requires schema-19 fixed-profile base")
        self.base.load_state_dict(payload["state_dict"], strict=True)
        self.base.pretrain_sha256 = payload["architecture"].get("stage_pretrain_sha256")
        self.base_checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def forward(self, wet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        with torch.no_grad():
            base, _, _ = self.base(wet)
        wet_diff = F.pad(torch.diff(wet, dim=1), (1, 0))
        base_diff = F.pad(torch.diff(base, dim=1), (1, 0))
        features = torch.stack((wet, base, wet_diff, base_diff), dim=1)
        hidden = torch.tanh(self.input(features))
        for block in self.blocks:
            hidden = block(hidden)
        local_scale = F.avg_pool1d(base.abs().unsqueeze(1), 257, 1, 128).squeeze(1)
        correction = 0.75 * (local_scale + 1.0e-4) * torch.tanh(self.output(hidden).squeeze(1))
        restored = (base + correction).clamp(-OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT)
        uncertainty = F.softplus(restored.new_tensor(-3.0)).expand_as(restored) + 1.0e-5
        return restored, uncertainty, None

    def manifest(self) -> dict:
        receptive = 9 + 4 * sum(2**index for index in range(self.depth)) + 8
        base = self.base.manifest()
        return {
            **base,
            "schema": 20,
            "architecture": "fixed-device-eight-stage-gray-box-plus-transient-refiner",
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "base_parameters_frozen": True,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "transient_channels": self.channels,
            "transient_depth": self.depth,
            "transient_receptive_field_frames": receptive,
            "transient_receptive_field_seconds": receptive / RATE,
        }
