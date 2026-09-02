"""Compact control-conditioned spectral inverse stage."""

from __future__ import annotations

import torch
from torch import nn

from .net import Block, DILATIONS


class ConditionalNet(nn.Module):
    """Invert one effect family while conditioning on three normalized knobs."""

    def __init__(
        self,
        channels: int = 12,
        n_fft: int = 512,
        hop: int = 128,
        dilations: tuple[tuple[int, int], ...] = DILATIONS,
    ) -> None:
        super().__init__()
        self.channels, self.n_fft, self.hop = int(channels), int(n_fft), int(hop)
        self.dilations = tuple(tuple(map(int, value)) for value in dilations)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        # Real, imaginary, magnitude, and three spatially constant controls.
        self.stem = nn.Conv2d(6, channels, (5, 3), padding=(2, 1))
        self.blocks = nn.ModuleList(Block(channels, value) for value in self.dilations)
        self.head = nn.Conv2d(channels, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _controls(controls: torch.Tensor, batch: int) -> torch.Tensor:
        controls = torch.as_tensor(controls)
        if controls.shape != (batch, 3) or not torch.isfinite(controls).all():
            raise ValueError("controls must be finite [batch,3]")
        if bool((controls < 0.0).any()) or bool((controls > 1.0).any()):
            raise ValueError("controls must be normalized to [0,1]")
        return controls

    def core(self, features: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != 3:
            raise ValueError("spectral features must be [batch,3,frequency,time]")
        controls = self._controls(controls, features.shape[0]).to(features)
        planes = controls[:, :, None, None].expand(-1, -1, features.shape[2], features.shape[3])
        value = self.stem(torch.cat((features, planes), 1))
        for block in self.blocks:
            value = block(value)
        return self.head(torch.tanh(value))

    def forward(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        if wet.ndim != 2 or not torch.isfinite(wet).all():
            raise ValueError("wet audio must be finite [batch,time]")
        controls = self._controls(controls, wet.shape[0]).to(wet)
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError("restoration strength must be between zero and one")
        if strength == 0.0:
            return wet
        spectrum = torch.stft(
            wet, self.n_fft, self.hop, window=self.window.to(wet), return_complex=True
        )
        scale = spectrum.abs().mean((1, 2), keepdim=True).clamp_min(1.0e-6)
        features = torch.stack(
            (spectrum.real / scale, spectrum.imag / scale, torch.log1p(spectrum.abs() / scale)),
            1,
        )
        residual = self.core(features, controls)
        estimate = spectrum + torch.complex(residual[:, 0], residual[:, 1]) * scale * float(strength)
        return torch.istft(
            estimate, self.n_fft, self.hop, window=self.window.to(wet), length=wet.shape[1]
        )

