"""Profile-conditioned long-context Ambience v3 residual expert."""

from __future__ import annotations

import torch
from torch import nn

from .ambience2 import CONTROL_WIDTH
from .ambience_model2 import _LongBlock


class AmbienceProfileExpert(nn.Module):
    """Correct only bounded fallback profiles; preserve exact inverses bit-for-bit."""

    mechanism = "ambience"

    def __init__(self, channels: int = 8, depth: int = 8, n_fft: int = 1024, hop: int = 256) -> None:
        super().__init__()
        if channels < 4 or not 6 <= depth <= 9 or n_fft < 512 or hop < 64:
            raise ValueError("invalid ambience3 expert geometry")
        self.channels = channels
        self.depth = depth
        self.n_fft = n_fft
        self.hop = hop
        self.dilations = tuple(2**index for index in range(depth))
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.stem = nn.Conv2d(7 + CONTROL_WIDTH, channels, (5, 3), padding=(2, 1))
        self.blocks = nn.ModuleList(_LongBlock(channels, dilation) for dilation in self.dilations)
        self.correction = nn.Conv2d(channels, 2, 1)
        self.uncertainty = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        nn.init.zeros_(self.uncertainty.weight)
        nn.init.constant_(self.uncertainty.bias, -4.0)

    def forward(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        profile_base: torch.Tensor,
        profile_fallback: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if wet.ndim != 2 or controls.shape != (wet.shape[0], CONTROL_WIDTH):
            raise ValueError("ambience3 expects wet [batch,time] and controls [batch,3]")
        if profile_base.shape != wet.shape or profile_fallback.shape != (wet.shape[0],):
            raise ValueError("ambience3 profile inputs changed geometry")
        if not all(torch.isfinite(value).all() for value in (wet, controls, profile_base)):
            raise ValueError("ambience3 inputs must be finite")
        window = self.window.to(wet)
        wet_spectrum = torch.stft(
            wet, self.n_fft, self.hop, window=window, center=True,
            pad_mode="constant", return_complex=True,
        )
        base_spectrum = torch.stft(
            profile_base, self.n_fft, self.hop, window=window, center=True,
            pad_mode="constant", return_complex=True,
        )
        scale = wet_spectrum.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1.0e-6)
        features = torch.stack(
            (
                wet_spectrum.real / scale,
                wet_spectrum.imag / scale,
                torch.log1p(wet_spectrum.abs() / scale),
                base_spectrum.real / scale,
                base_spectrum.imag / scale,
                torch.log1p(base_spectrum.abs() / scale),
            ),
            dim=1,
        )
        condition = controls.mul(2.0).sub(1.0)[:, :, None, None].expand(
            -1, -1, features.shape[2], features.shape[3]
        )
        fallback_feature = profile_fallback.to(wet)[:, None, None, None].expand(
            -1, 1, features.shape[2], features.shape[3]
        )
        hidden = torch.tanh(self.stem(torch.cat((features, condition, fallback_feature), dim=1)))
        for block in self.blocks:
            hidden = block(hidden)
        residual = self.correction(hidden)
        estimate = base_spectrum + torch.complex(residual[:, 0], residual[:, 1]) * scale * 0.10
        candidate = torch.istft(
            estimate, self.n_fft, self.hop, window=window,
            center=True, length=wet.shape[1],
        )
        fallback = profile_fallback.to(wet)[:, None]
        restored = profile_base + fallback * (candidate - profile_base)
        frame_uncertainty = torch.nn.functional.softplus(
            self.uncertainty(hidden.mean(dim=2)).squeeze(1)
        ) + 1.0e-5
        uncertainty = torch.nn.functional.interpolate(
            frame_uncertainty.unsqueeze(1), size=wet.shape[1],
            mode="linear", align_corners=False,
        ).squeeze(1)
        return restored, uncertainty

    def manifest(self) -> dict:
        spectral_frames = 1 + 2 * sum(self.dilations)
        return {
            "schema": 2,
            "architecture": "exact-or-bounded-profile-inverse-plus-long-context-residual",
            "mechanism": self.mechanism,
            "channels": self.channels,
            "depth": self.depth,
            "n_fft": self.n_fft,
            "hop": self.hop,
            "control_width": CONTROL_WIDTH,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "spectral_context_frames": spectral_frames,
            "waveform_context_frames": spectral_frames * self.hop + self.n_fft,
            "normalization_scope": "bounded input window",
            "causal": False,
            "profile_required": True,
            "exact_profile_path_is_immutable": True,
            "uncertainty_output": True,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        }
