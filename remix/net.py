"""Small complex-STFT wet-to-clean network shared by training and runtime."""

from __future__ import annotations

import torch
from torch import nn


DILATIONS = ((1, 1), (2, 1), (4, 2), (8, 4), (16, 8))


class Block(nn.Module):
    def __init__(self, channels: int, dilation: tuple[int, int]):
        super().__init__()
        frequency, time = dilation
        self.filter = nn.Conv2d(
            channels,
            channels * 2,
            (5, 3),
            padding=(frequency * 2, time),
            dilation=dilation,
        )
        self.mix = nn.Conv2d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        left, right = self.filter(value).chunk(2, 1)
        return value + self.mix(torch.tanh(left) * torch.sigmoid(right))


class SpectralNet(nn.Module):
    """Complex-STFT residual restorer with exact strength-zero bypass."""

    def __init__(
        self,
        channels: int = 12,
        n_fft: int = 512,
        hop: int = 128,
        dilations: tuple[tuple[int, int], ...] = DILATIONS,
    ):
        super().__init__()
        self.channels, self.n_fft, self.hop = int(channels), int(n_fft), int(hop)
        self.dilations = tuple(tuple(map(int, value)) for value in dilations)
        if not self.dilations or any(left <= 0 or right <= 0 for left, right in self.dilations):
            raise ValueError("spectral dilations must be positive")
        self.minimum_halo = (2 + sum(value[1] for value in self.dilations)) * self.hop
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.stem = nn.Conv2d(3, channels, (5, 3), padding=(2, 1))
        self.blocks = nn.ModuleList(Block(channels, value) for value in self.dilations)
        self.head = nn.Conv2d(channels, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def core(self, features: torch.Tensor) -> torch.Tensor:
        """Run only the portable learned core on normalized STFT features."""

        if features.ndim != 4 or features.shape[1] != 3:
            raise ValueError("spectral features must be [batch,3,frequency,time]")
        value = self.stem(features)
        for block in self.blocks:
            value = block(value)
        return self.head(torch.tanh(value))

    def _forward(
        self,
        wet: torch.Tensor,
        strength: float,
        scale: torch.Tensor | None,
    ) -> torch.Tensor:
        if wet.ndim != 2 or not torch.isfinite(wet).all():
            raise ValueError("wet audio must be finite [batch,time]")
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError("restoration strength must be between zero and one")
        if strength == 0.0:
            return wet
        spectrum = torch.stft(
            wet, self.n_fft, self.hop, window=self.window.to(wet.dtype), return_complex=True
        )
        if scale is None:
            scale = spectrum.abs().mean((1, 2), keepdim=True)
        else:
            scale = torch.as_tensor(scale, dtype=wet.dtype, device=wet.device)
            if scale.ndim == 0:
                scale = scale.reshape(1, 1, 1).expand(wet.shape[0], -1, -1)
            if scale.shape != (wet.shape[0], 1, 1) or not torch.isfinite(scale).all():
                raise ValueError("spectral scale must be finite [batch,1,1]")
        scale = scale.clamp_min(1e-6)
        features = torch.stack(
            (spectrum.real / scale, spectrum.imag / scale, torch.log1p(spectrum.abs() / scale)),
            1,
        )
        residual = self.core(features)
        estimate = spectrum + torch.complex(residual[:, 0], residual[:, 1]) * scale * float(strength)
        return torch.istft(
            estimate, self.n_fft, self.hop, window=self.window.to(wet.dtype), length=wet.shape[1]
        )

    def forward(self, wet: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        return self._forward(wet, strength, None)

    def scaled(
        self,
        wet: torch.Tensor,
        scale: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Restore with a caller-supplied clip scale for bounded chunk parity."""

        return self._forward(wet, strength, scale)
