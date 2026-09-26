"""Joint complex and long-context Demucs-style Wet-only Amp+cab inverse."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .egdb_pg_amp_model import OUTPUT_PEAK_CONTRACT, RATE
from .egdb_pg_amp_model5 import WetToneComplexAmpCabInverse


class _Encoder(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv1d(input_channels, output_channels, 8, stride=4, padding=2)
        self.norm = nn.GroupNorm(1, output_channels)
        self.depthwise = nn.Conv1d(
            output_channels, output_channels, 5, padding=2, groups=output_channels
        )
        self.mix = nn.Conv1d(output_channels, output_channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.gelu(self.norm(self.down(value)))
        return value + 0.20 * self.mix(F.gelu(self.depthwise(value)))


class _Decoder(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(
            input_channels, output_channels, 8, stride=4, padding=2
        )
        self.norm = nn.GroupNorm(1, output_channels)
        self.depthwise = nn.Conv1d(
            output_channels, output_channels, 5, padding=2, groups=output_channels
        )
        self.mix = nn.Conv1d(output_channels, output_channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.gelu(self.norm(self.up(value)))
        return value + 0.20 * self.mix(F.gelu(self.depthwise(value)))


class _BottleneckBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
        )
        self.expand = nn.Conv1d(channels, channels * 2, 1)
        self.contract = nn.Conv1d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.expand(self.depthwise(self.norm(value)))
        hidden = F.glu(hidden, dim=1)
        return value + 0.20 * self.contract(hidden)


class WetToneComplexDemucsJointAmpCabInverse(nn.Module):
    """Jointly refine a complex inverse with a higher-capacity waveform path."""

    mechanism = "amp"

    def __init__(
        self, channels: int = 48, depth: int = 8, condition_size: int = 64
    ) -> None:
        super().__init__()
        if channels < 16 or depth < 4 or condition_size < 16:
            raise ValueError("invalid joint Demucs Amp+cab geometry")
        self.channels = channels
        self.depth = depth
        self.condition_size = condition_size
        self.base = WetToneComplexAmpCabInverse(
            max(12, channels // 2), max(4, depth - 2), condition_size
        )
        for parameter in self.base.tone_encoder.parameters():
            parameter.requires_grad_(False)
        self.base_checkpoint_sha256 = None

        widths = (channels, channels * 2, channels * 4, channels * 8, channels * 8)
        self.encoders = nn.ModuleList((
            _Encoder(2, widths[0]),
            _Encoder(widths[0], widths[1]),
            _Encoder(widths[1], widths[2]),
            _Encoder(widths[2], widths[3]),
            _Encoder(widths[3], widths[4]),
        ))
        dilations = tuple(2 ** (index % 6) for index in range(depth))
        self.bottleneck = nn.ModuleList(
            _BottleneckBlock(widths[-1], dilation) for dilation in dilations
        )
        self.condition = nn.Linear(64, widths[-1] * 2)
        self.decoders = nn.ModuleList((
            _Decoder(widths[4], widths[3]),
            _Decoder(widths[3], widths[2]),
            _Decoder(widths[2], widths[1]),
            _Decoder(widths[1], widths[0]),
            _Decoder(widths[0], widths[0]),
        ))
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
            raise ValueError("joint Demucs requires a complex schema 5 checkpoint")
        expected = (
            architecture.get("channels"),
            architecture.get("depth"),
            architecture.get("condition_size"),
        )
        actual = (self.base.channels, self.base.depth, self.base.condition_size)
        if expected != actual:
            raise ValueError(f"complex base geometry mismatch: {expected} != {actual}")
        self.base.load_state_dict(payload["state_dict"])
        self.base.tone_encoder_sha256 = architecture.get("tone_encoder_sha256")
        self.base_checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def forward(
        self,
        wet: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        tone_reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if state is not None:
            raise ValueError("offline Demucs Amp+cab expert accepts no graph or external state")
        if wet.ndim != 2 or not torch.isfinite(wet).all():
            raise ValueError("joint Demucs Amp+cab expects finite Wet [batch,time]")
        reference = wet if tone_reference is None else tone_reference
        if reference.ndim != 2 or reference.shape[0] != wet.shape[0]:
            raise ValueError("joint Demucs Amp+cab expects batch-aligned Wet reference")

        original_frames = wet.shape[1]
        padding = (-original_frames) % 1024
        padded_wet = F.pad(wet, (0, padding)) if padding else wet
        base_restored, _, _ = self.base(padded_wet, tone_reference=reference)
        with torch.no_grad():
            tone = self.base.tone_encoder(reference)

        hidden = torch.stack((padded_wet, base_restored), dim=1)
        skips = []
        for encoder in self.encoders:
            hidden = encoder(hidden)
            skips.append(hidden)
        scale, bias = self.condition(tone).chunk(2, dim=1)
        hidden = hidden * (1.0 + 0.25 * torch.tanh(scale)[:, :, None])
        hidden = hidden + 0.25 * torch.tanh(bias)[:, :, None]
        for block in self.bottleneck:
            hidden = block(hidden)
        for index, decoder in enumerate(self.decoders):
            hidden = decoder(hidden)
            if index < 4:
                hidden = hidden + skips[-2 - index]

        log_gain, residual = self.controls(hidden).chunk(2, dim=1)
        local_scale = F.avg_pool1d(
            base_restored.abs().unsqueeze(1), 257, 1, 128
        ).clamp_min(1.0e-4)
        restored = (
            base_restored * torch.exp(0.75 * torch.tanh(log_gain.squeeze(1)))
            + local_scale.squeeze(1) * torch.tanh(residual.squeeze(1))
        ).clamp(-OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT)
        uncertainty = F.softplus(self.uncertainty(hidden).squeeze(1)) + 1.0e-5
        return restored[:, :original_frames], uncertainty[:, :original_frames], None

    def manifest(self) -> dict:
        bottleneck_samples = 1 + 2 * sum(
            block.depthwise.dilation[0] for block in self.bottleneck
        )
        return {
            "schema": 10,
            "architecture": "joint-complex-inverse-plus-long-context-demucs-generator",
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
            "downsample_factor": 1024,
            "bottleneck_receptive_field_seconds": bottleneck_samples * 1024 / RATE,
            "base_complex_inverse_frozen": False,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "complex_decoder_trainable": True,
            "second_stage": "long-context Demucs-style waveform gain and residual generator",
            "phase_policy": "generated by jointly trained complex base then waveform-refined",
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
