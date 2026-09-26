"""Frozen phase gray box plus explicit multiband dynamics inversion."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .egdb_pg_amp_model import OUTPUT_PEAK_CONTRACT, RATE
from .egdb_pg_amp_model6 import _EnvelopeBlock
from .egdb_pg_amp_model10 import WetTonePhaseGrayBoxAmpCabInverse


class WetTonePhaseGrayBoxMultibandDynamicsAmpCabInverse(nn.Module):
    """Restore band-dependent crest and level over a frozen gray-box inverse."""

    mechanism = "amp"

    def __init__(
        self, channels: int = 24, depth: int = 8, condition_size: int = 64
    ) -> None:
        super().__init__()
        if channels < 12 or depth < 4 or condition_size < 16:
            raise ValueError("invalid multiband gray-box dynamics geometry")
        self.channels = channels
        self.depth = depth
        self.condition_size = condition_size
        self.n_fft = 512
        self.hop = 128
        self.band_edges_hz = (0.0, 250.0, 1200.0, 4000.0, RATE / 2)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        frequencies = torch.linspace(0.0, RATE / 2, self.n_fft // 2 + 1)
        masks = []
        for index, (left, right) in enumerate(zip(
            self.band_edges_hz[:-1], self.band_edges_hz[1:], strict=True
        )):
            if index == len(self.band_edges_hz) - 2:
                mask = (frequencies >= left) & (frequencies <= right)
            else:
                mask = (frequencies >= left) & (frequencies < right)
            masks.append(mask.to(torch.float32))
        self.register_buffer("band_masks", torch.stack(masks)[:, :, None], persistent=False)

        self.base = WetTonePhaseGrayBoxAmpCabInverse(channels, depth, condition_size)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base_checkpoint_sha256 = None
        self.condition_projection = nn.Sequential(
            nn.Linear(64, condition_size), nn.GELU(),
            nn.Linear(condition_size, condition_size),
        )
        self.stem = nn.Conv1d(12, channels, 5, padding=2)
        self.blocks = nn.ModuleList(
            _EnvelopeBlock(channels, condition_size, 2 ** (index % 6))
            for index in range(depth)
        )
        self.controls = nn.Conv1d(channels, 8, 1)
        self.uncertainty_logit = nn.Parameter(torch.tensor(-3.0))
        nn.init.zeros_(self.controls.weight)
        nn.init.zeros_(self.controls.bias)

    def load_base(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        architecture = payload.get("architecture", {})
        if architecture.get("schema") != 12:
            raise ValueError("multiband dynamics requires phase gray-box schema 12")
        expected = (architecture.get("depth"), architecture.get("condition_size"))
        if expected != (self.depth, self.condition_size):
            raise ValueError(f"phase gray-box geometry mismatch: {expected}")
        self.base.load_state_dict(payload["state_dict"])
        self.base.tone_encoder_sha256 = architecture.get("tone_encoder_sha256")
        self.base_checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def _bands(self, value: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(
            value,
            self.n_fft,
            self.hop,
            window=self.window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        batch = spectrum.shape[0]
        masked = (spectrum[:, None] * self.band_masks[None]).reshape(
            batch * len(self.band_edges_hz[:-1]), spectrum.shape[-2], spectrum.shape[-1]
        )
        bands = torch.istft(
            masked,
            self.n_fft,
            self.hop,
            window=self.window,
            center=True,
            length=value.shape[1],
        )
        return bands.reshape(batch, len(self.band_edges_hz[:-1]), value.shape[1])

    def _features(self, bands: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, band_count, frames = bands.shape
        flattened = bands.reshape(batch * band_count, 1, frames)
        rms = F.avg_pool1d(flattened.square(), 512, self.hop, padding=256).add(1.0e-8).sqrt()
        peak = F.max_pool1d(flattened.abs(), 512, self.hop, padding=256)
        crest = peak / rms.clamp_min(1.0e-4)
        feature_frames = rms.shape[-1]
        features = torch.cat((
            torch.log(rms + 1.0e-5).reshape(batch, band_count, feature_frames),
            torch.log(peak + 1.0e-5).reshape(batch, band_count, feature_frames),
            torch.log(crest + 1.0e-4).reshape(batch, band_count, feature_frames),
        ), dim=1)
        return features, rms.reshape(batch, band_count, feature_frames)

    def forward(
        self,
        wet: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        tone_reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if state is not None:
            raise ValueError("multiband dynamics accepts no graph or external state")
        reference = wet if tone_reference is None else tone_reference
        with torch.no_grad():
            base_restored, _, _ = self.base(wet, tone_reference=reference)
            tone = self.base.tone_encoder(reference)
        bands = self._bands(base_restored)
        features, band_rms = self._features(bands)
        condition = self.condition_projection(tone)
        hidden = F.gelu(self.stem(features))
        for block in self.blocks:
            hidden = block(hidden, condition)
        controls = F.interpolate(
            self.controls(hidden), size=wet.shape[1], mode="linear", align_corners=False
        ).reshape(wet.shape[0], 4, 2, wet.shape[1])
        drive = 0.75 * torch.tanh(controls[:, :, 0])
        level = 0.50 * torch.tanh(controls[:, :, 1])
        local_rms = F.interpolate(
            band_rms, size=wet.shape[1], mode="linear", align_corners=False
        )
        relative = bands.abs() / local_rms.clamp_min(1.0e-4)
        crest_shape = torch.tanh(1.5 * (relative - 1.0))
        gain = torch.exp(level + drive * crest_shape)
        restored = (bands * gain).sum(1).clamp(
            -OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT
        )
        uncertainty = F.softplus(self.uncertainty_logit).expand_as(restored) + 1.0e-5
        return restored, uncertainty, None

    def manifest(self) -> dict:
        return {
            "schema": 14,
            "architecture": "frozen-phase-gray-box-plus-explicit-four-band-envelope-inverse",
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
            "band_edges_hz": list(self.band_edges_hz),
            "dynamics_policy": "bounded independent per-band crest and level restoration",
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
