"""Regularized physics-first nonlinear inverse expert.

The repository-owned nonlinear renderer applies its stages as waveshaping,
low-pass filtering and output level.  This expert reverses those internal
stages without receiving graph order or neighbouring-effect context.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .product2 import CONTROL_WIDTH


RATE = 48_000.0
FIR_TAPS = 257
FIR_FFT = 4096
DEEQUALIZER_GAIN_DB = 12.0
CLEAN_PEAK_CONTRACT = 0.95
OUTPUT_PEAK_CONTRACT = 0.98


def _regularized_deequalize(
    audio: torch.Tensor,
    controls: torch.Tensor,
    *,
    taps: int = FIR_TAPS,
    fft_size: int = FIR_FFT,
    maximum_gain_db: float = DEEQUALIZER_GAIN_DB,
) -> torch.Tensor:
    """Apply a bounded zero-phase inverse of the known 2-pole low-pass.

    A Wiener denominator and a fixed gain ceiling prevent the inverse from
    exploding near Nyquist, where the forward filter has intentionally removed
    information.  One FIR is generated per stream because cutoff is a fixed
    control for the whole example.
    """
    if audio.ndim != 2 or controls.shape != (audio.shape[0], CONTROL_WIDTH):
        raise ValueError("nonlinear3 de-equalizer expects audio [batch,time] and fixed-width controls")
    if taps < 3 or taps % 2 != 1 or fft_size < taps or maximum_gain_db <= 0.0:
        raise ValueError("invalid nonlinear3 de-equalizer geometry")

    cutoff = 1800.0 * torch.exp(
        controls[:, 2] * math.log(11000.0 / 1800.0)
    )
    k = torch.tan(torch.pi * cutoff / RATE)
    normalizer = 1.0 / (1.0 + math.sqrt(2.0) * k + k.square())
    b0 = k.square() * normalizer
    b1 = 2.0 * b0
    b2 = b0
    a1 = 2.0 * (k.square() - 1.0) * normalizer
    a2 = (1.0 - math.sqrt(2.0) * k + k.square()) * normalizer

    omega = torch.linspace(
        0.0,
        torch.pi,
        fft_size // 2 + 1,
        dtype=audio.dtype,
        device=audio.device,
    )
    z1 = torch.polar(torch.ones_like(omega), -omega).unsqueeze(0)
    z2 = z1.square()
    numerator = b0[:, None] + b1[:, None] * z1 + b2[:, None] * z2
    denominator = 1.0 + a1[:, None] * z1 + a2[:, None] * z2
    response = numerator / denominator
    gain = 10.0 ** (maximum_gain_db / 20.0)
    inverse = response.conj() / (response.abs().square() + 1.0 / gain**2)
    inverse = inverse / inverse[:, :1].abs().clamp_min(1.0e-8)

    impulse = torch.fft.irfft(inverse, n=fft_size, dim=-1)
    impulse = torch.roll(impulse, fft_size // 2, dims=-1)
    center = fft_size // 2
    half = taps // 2
    kernels = impulse[:, center - half : center + half + 1]
    window = torch.hann_window(
        taps,
        periodic=False,
        dtype=audio.dtype,
        device=audio.device,
    )
    kernels = kernels * window
    kernels = kernels / kernels.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    # Grouped convolution applies one control-conditioned FIR to each stream.
    grouped = audio.unsqueeze(0)
    # conv1d computes cross-correlation, while kernels are impulse responses
    # intended for convolution.  Reverse them to preserve the inverse phase.
    restored = F.conv1d(
        grouped,
        kernels.flip(-1).unsqueeze(1),
        padding=half,
        groups=audio.shape[0],
    )
    return restored.squeeze(0)


def _inverse_shape(shaped: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
    drive = 1.8 + controls[:, 0:1] * (8.0 - 1.8)
    bias = -0.12 + controls[:, 1:2] * 0.24
    clean_low = torch.full_like(drive, -CLEAN_PEAK_CONTRACT)
    clean_high = torch.full_like(drive, CLEAN_PEAK_CONTRACT)
    tanh_offset = torch.tanh(bias)
    tanh_low = torch.tanh(clean_low * drive + bias) - tanh_offset
    tanh_high = torch.tanh(clean_high * drive + bias) - tanh_offset
    atan_offset = (2.0 / torch.pi) * torch.atan(bias * 1.8)
    atan_low = (2.0 / torch.pi) * torch.atan((clean_low * drive + bias) * 1.8) - atan_offset
    atan_high = (2.0 / torch.pi) * torch.atan((clean_high * drive + bias) * 1.8) - atan_offset

    def cubic_forward(clean: torch.Tensor) -> torch.Tensor:
        value = (clean * drive + bias).clamp(-1.5, 1.5)
        return value - value.pow(3) / 6.75

    cubic_low = cubic_forward(clean_low)
    cubic_high = cubic_forward(clean_high)
    shape = controls[:, 4:5]
    lower = torch.where(shape < 0.25, tanh_low, torch.where(shape < 0.75, atan_low, cubic_low))
    upper = torch.where(shape < 0.25, tanh_high, torch.where(shape < 0.75, atan_high, cubic_high))
    shaped = torch.maximum(torch.minimum(shaped, upper), lower)

    tanh_base = (
        torch.atanh((shaped + tanh_offset).clamp(-0.995, 0.995)) - bias
    ) / drive
    atan_base = (
        torch.tan(((shaped + atan_offset) * torch.pi / 2.0).clamp(-1.52, 1.52)) / 1.8 - bias
    ) / drive
    target = shaped.clamp(-1.0, 1.0)
    low = torch.full_like(target, -1.5)
    high = torch.full_like(target, 1.5)
    for _ in range(20):
        middle = (low + high) * 0.5
        value = middle - middle.pow(3) / 6.75
        low = torch.where(value < target, middle, low)
        high = torch.where(value >= target, middle, high)
    cubic_base = ((low + high) * 0.5 - bias) / drive
    return torch.where(shape < 0.25, tanh_base, torch.where(shape < 0.75, atan_base, cubic_base))


def _nonlinear3_preconditioner(wet: torch.Tensor, controls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    level = 0.45 + controls[:, 3:4] * 0.40
    deequalized = _regularized_deequalize(wet / level, controls)
    return _inverse_shape(deequalized, controls), deequalized


class NonlinearInverseV3(nn.Module):
    """Independent nonlinear expert with bounded physical preconditioning."""

    mechanism = "nonlinear"

    def __init__(self, hidden_size: int = 20, layers: int = 2) -> None:
        super().__init__()
        if hidden_size < 4 or layers < 1:
            raise ValueError("invalid nonlinear3 expert geometry")
        self.hidden_size = hidden_size
        self.layers = layers
        self.depth = max(5, layers * 2)
        self.dilations = tuple(2**index for index in range(self.depth))
        self.input = nn.Conv1d(5 + CONTROL_WIDTH, hidden_size, 1)
        self.temporal = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size * 2, 7, padding=3 * dilation, dilation=dilation)
            for dilation in self.dilations
        ])
        self.mix = nn.ModuleList([nn.Conv1d(hidden_size, hidden_size, 1) for _ in self.dilations])
        self.correction = nn.Conv1d(hidden_size, 1, 1)
        self.uncertainty = nn.Conv1d(hidden_size, 1, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        nn.init.zeros_(self.uncertainty.weight)
        nn.init.constant_(self.uncertainty.bias, -4.0)

    def forward(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if wet.ndim != 2 or controls.shape != (wet.shape[0], CONTROL_WIDTH):
            raise ValueError("nonlinear3 expects wet [batch,time] and fixed-width controls")
        if state is not None:
            raise ValueError("nonlinear3 local expert does not accept recurrent state")
        base, deequalized = _nonlinear3_preconditioner(wet, controls)
        condition = controls.mul(2.0).sub(1.0).unsqueeze(1).expand(-1, wet.shape[1], -1)
        base_difference = F.pad(base[:, 1:] - base[:, :-1], (1, 0))
        features = torch.cat(
            (
                wet.unsqueeze(-1),
                wet.abs().unsqueeze(-1),
                deequalized.unsqueeze(-1),
                base.unsqueeze(-1),
                base_difference.unsqueeze(-1),
                condition,
            ),
            dim=-1,
        ).transpose(1, 2)
        hidden = torch.tanh(self.input(features))
        for temporal, mix in zip(self.temporal, self.mix, strict=True):
            value, gate = temporal(hidden).chunk(2, dim=1)
            hidden = hidden + 0.25 * mix(torch.tanh(value) * torch.sigmoid(gate))
        correction_limit = torch.minimum(
            torch.full_like(base, 0.125),
            (OUTPUT_PEAK_CONTRACT - base.abs()).clamp_min(0.0),
        )
        correction = correction_limit * torch.tanh(self.correction(hidden).squeeze(1))
        uncertainty = F.softplus(self.uncertainty(hidden).squeeze(1)) + 1.0e-5
        return base + correction, uncertainty, None

    def manifest(self) -> dict:
        tcn_receptive_field = 1 + 6 * sum(self.dilations)
        return {
            "schema": 4,
            "architecture": "regularized-deeq-plus-analytic-shape-inverse-plus-local-tcn",
            "mechanism": self.mechanism,
            "hidden_size": self.hidden_size,
            "layers": self.layers,
            "control_width": CONTROL_WIDTH,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "state_floats_per_mono_stream": 0,
            "normalization_across_time": False,
            "causal": False,
            "uncertainty_output": True,
            "internal_forward_order": ["waveshape", "lowpass", "level"],
            "internal_inverse_order": ["level", "regularized-lowpass-inverse", "waveshape-inverse"],
            "graph_order_input": False,
            "neighbouring_effect_input": False,
            "deequalizer_taps": FIR_TAPS,
            "deequalizer_maximum_gain_db": DEEQUALIZER_GAIN_DB,
            "clean_peak_contract": CLEAN_PEAK_CONTRACT,
            "output_peak_contract": OUTPUT_PEAK_CONTRACT,
            "local_receptive_field_frames": FIR_TAPS + tcn_receptive_field - 1,
        }
