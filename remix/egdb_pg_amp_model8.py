"""Frozen complex restoration plus a compact waveform U-Net generator."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .egdb_pg_amp_model import OUTPUT_PEAK_CONTRACT, RATE
from .egdb_pg_amp_model5 import WetToneComplexAmpCabInverse


class _Down(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, 8, stride=4, padding=2),
            nn.GroupNorm(1, output_channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class _Up(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose1d(
                input_channels, output_channels, 8, stride=4, padding=2
            ),
            nn.GroupNorm(1, output_channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class WetToneComplexUNetAmpCabInverse(nn.Module):
    """Generate missing waveform detail after frozen complex restoration."""

    mechanism = "amp"

    def __init__(self, channels: int = 24, depth: int = 6, condition_size: int = 64) -> None:
        super().__init__()
        if channels < 12 or depth < 4 or condition_size < 16:
            raise ValueError("invalid complex U-Net Amp+cab geometry")
        self.channels = channels
        self.depth = depth
        self.condition_size = condition_size
        self.base = WetToneComplexAmpCabInverse(channels, depth, condition_size)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base_checkpoint_sha256 = None
        widths = (channels, channels * 2, channels * 4, channels * 8)
        self.down1 = _Down(2, widths[0])
        self.down2 = _Down(widths[0], widths[1])
        self.down3 = _Down(widths[1], widths[2])
        self.down4 = _Down(widths[2], widths[3])
        self.bottleneck = nn.ModuleList(
            nn.Conv1d(
                widths[3], widths[3], 5, dilation=dilation,
                padding=2 * dilation, groups=widths[3],
            )
            for dilation in (1, 2, 4, 8)
        )
        self.bottleneck_mix = nn.ModuleList(
            nn.Conv1d(widths[3], widths[3], 1) for _ in self.bottleneck
        )
        self.condition = nn.Linear(64, widths[3] * 2)
        self.up3 = _Up(widths[3], widths[2])
        self.up2 = _Up(widths[2], widths[1])
        self.up1 = _Up(widths[1], widths[0])
        self.up0 = _Up(widths[0], widths[0])
        self.controls = nn.Conv1d(widths[0], 2, 1)
        self.uncertainty = nn.Conv1d(widths[0], 1, 1)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)
        nn.init.zeros_(self.controls.weight)
        nn.init.zeros_(self.controls.bias)
        nn.init.zeros_(self.uncertainty.weight)
        nn.init.constant_(self.uncertainty.bias, -3.0)

    def load_base(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        architecture = payload.get("architecture", {})
        if architecture.get("schema") != 5:
            raise ValueError("waveform U-Net requires a complex schema 5 checkpoint")
        expected = (
            architecture.get("channels"), architecture.get("depth"),
            architecture.get("condition_size"),
        )
        if expected != (self.channels, self.depth, self.condition_size):
            raise ValueError(f"complex base geometry mismatch: {expected}")
        self.base.load_state_dict(payload["state_dict"])
        self.base.tone_encoder_sha256 = architecture.get("tone_encoder_sha256")
        self.base_checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def _base_forward(
        self, wet: torch.Tensor, reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            restored, _, _ = self.base(wet, tone_reference=reference)
            tone = self.base.tone_encoder(reference)
        return restored, tone

    def forward(
        self, wet: torch.Tensor, state: torch.Tensor | None = None,
        *, tone_reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if state is not None:
            raise ValueError("offline U-Net Amp+cab expert accepts no graph or external state")
        original_frames = wet.shape[1]
        padding = (-original_frames) % 256
        padded_wet = F.pad(wet, (0, padding)) if padding else wet
        reference = padded_wet if tone_reference is None else tone_reference
        base_restored, tone = self._base_forward(padded_wet, reference)
        value = torch.stack((padded_wet, base_restored), dim=1)
        down1 = self.down1(value)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        hidden = self.down4(down3)
        scale, bias = self.condition(tone).chunk(2, dim=1)
        hidden = hidden * (1.0 + 0.25 * torch.tanh(scale)[:, :, None])
        hidden = hidden + 0.25 * torch.tanh(bias)[:, :, None]
        for depthwise, mix in zip(self.bottleneck, self.bottleneck_mix, strict=True):
            hidden = hidden + 0.20 * mix(F.gelu(depthwise(hidden)))
        hidden = self.up3(hidden) + down3
        hidden = self.up2(hidden) + down2
        hidden = self.up1(hidden) + down1
        hidden = self.up0(hidden)
        log_gain, residual = self.controls(hidden).chunk(2, dim=1)
        local_scale = F.avg_pool1d(
            base_restored.abs().unsqueeze(1), 129, 1, 64
        ).clamp_min(1.0e-4)
        restored = (
            base_restored * torch.exp(torch.tanh(log_gain.squeeze(1)))
            + 0.75 * local_scale.squeeze(1) * torch.tanh(residual.squeeze(1))
        ).clamp(-OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT)
        uncertainty = F.softplus(self.uncertainty(hidden).squeeze(1)) + 1.0e-5
        return restored[:, :original_frames], uncertainty[:, :original_frames], None

    def manifest(self) -> dict:
        return {
            "schema": 8,
            "architecture": "frozen-complex-inverse-plus-waveform-unet-generator",
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
            "downsample_factor": 256,
            "base_complex_inverse_frozen": True,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "second_stage": "waveform U-Net local gain and residual generator",
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


class WetToneComplexUNetJointAmpCabInverse(WetToneComplexUNetAmpCabInverse):
    """Jointly fine-tune the complex decoder and second-stage waveform U-Net."""

    def __init__(self, channels: int = 24, depth: int = 6, condition_size: int = 64) -> None:
        super().__init__(channels, depth, condition_size)
        for parameter in self.base.parameters():
            parameter.requires_grad_(True)
        for parameter in self.base.tone_encoder.parameters():
            parameter.requires_grad_(False)

    def _base_forward(
        self, wet: torch.Tensor, reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        restored, _, _ = self.base(wet, tone_reference=reference)
        with torch.no_grad():
            tone = self.base.tone_encoder(reference)
        return restored, tone

    def manifest(self) -> dict:
        manifest = super().manifest()
        manifest.update({
            "schema": 9,
            "architecture": "joint-complex-inverse-plus-waveform-unet-generator",
            "base_complex_inverse_frozen": False,
            "complex_decoder_trainable": True,
            "tone_encoder_frozen": True,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters()
                if parameter.requires_grad
            ),
        })
        return manifest
