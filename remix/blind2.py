"""Compact Wet-only multi-label effect-presence model.

This package predicts which supported effect families are present.  It never
predicts chain order and is intentionally separate from controls and inverse
experts.
"""

from __future__ import annotations

import torch
from torch import nn


RATE = 44_100
WINDOW = RATE * 5
FFT = 2_048
HOP = 1_024
MELS = 128
FRAMES = 216
FMIN = 30.0
FMAX = 16_000.0
LABELS = ("nonlinear", "echo", "ambience", "unknown")


def _hz_to_mel(value: torch.Tensor) -> torch.Tensor:
    return 2_595.0 * torch.log10(1.0 + value / 700.0)


def _mel_to_hz(value: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, value / 2_595.0) - 1.0)


def mel_filters(*, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return the Rust-compatible HTK/Slaney triangular filter bank."""

    device = torch.device(device)
    # MPS does not implement float64.  Float32 also matches the Rust client,
    # whose filter bank is stored and accumulated as f32.
    lower = _hz_to_mel(torch.tensor(FMIN, dtype=torch.float32, device=device))
    upper = _hz_to_mel(torch.tensor(FMAX, dtype=torch.float32, device=device))
    points = _mel_to_hz(torch.linspace(lower, upper, MELS + 2, dtype=torch.float32, device=device))
    frequencies = torch.arange(FFT // 2 + 1, dtype=torch.float32, device=device) * RATE / FFT
    left = points[:-2, None]
    center = points[1:-1, None]
    right = points[2:, None]
    rise = (frequencies - left) / (center - left)
    fall = (right - frequencies) / (right - center)
    filters = torch.minimum(rise, fall).clamp_min(0.0) * (2.0 / (right - left))
    return filters.to(torch.float32)


def log_mel(audio: torch.Tensor) -> torch.Tensor:
    """Create [batch, 1, 128, 216] features matching the client frontend."""

    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.ndim != 2 or audio.shape[1] != WINDOW:
        raise ValueError(f"expected [batch, {WINDOW}] audio, got {tuple(audio.shape)}")
    window = torch.hann_window(FFT, periodic=True, dtype=audio.dtype, device=audio.device)
    spectrum = torch.stft(
        audio,
        n_fft=FFT,
        hop_length=HOP,
        win_length=FFT,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    if spectrum.shape[-1] != FRAMES:
        raise AssertionError(f"frontend frame contract changed: {spectrum.shape[-1]} != {FRAMES}")
    power = spectrum.abs().square()
    mel = torch.einsum("mf,bft->bmt", mel_filters(device=audio.device), power).clamp_min(1.0e-10)
    mel = 10.0 * torch.log10(mel)
    peak = mel.amax(dim=(-2, -1), keepdim=True)
    mel = torch.maximum(mel, peak - 80.0)
    mean = mel.mean(dim=(-2, -1), keepdim=True)
    deviation = mel.std(dim=(-2, -1), correction=1, keepdim=True).clamp_min(1.0e-5)
    return ((mel - mean) / deviation)[:, None]


class BasicBlock(nn.Module):
    def __init__(self, source: int, target: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(source, target, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(target),
            nn.ReLU(inplace=True),
            nn.Conv2d(target, target, 3, padding=1, bias=False),
            nn.BatchNorm2d(target),
        )
        self.skip = (
            nn.Identity()
            if source == target and stride == 1
            else nn.Sequential(
                nn.Conv2d(source, target, 1, stride=stride, bias=False),
                nn.BatchNorm2d(target),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(value) + self.skip(value))


class BlindPresence(nn.Module):
    def __init__(self, labels: tuple[str, ...] = LABELS) -> None:
        super().__init__()
        self.labels = labels
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        stages = []
        source = 16
        for index, target in enumerate((16, 32, 64, 128)):
            stride = 1 if index == 0 else 2
            stages.extend((BasicBlock(source, target, stride), BasicBlock(target, target)))
            source = target
        self.body = nn.Sequential(*stages)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(256, len(labels))

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return self.body(self.stem(value))

    def features(self, value: torch.Tensor) -> torch.Tensor:
        value = self.encode(value)
        average = value.mean(dim=(-2, -1))
        maximum = value.amax(dim=(-2, -1))
        return torch.cat((average, maximum), dim=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.features(value)))

    def manifest(self) -> dict:
        return {
            "architecture": "compact-audio-resnet18",
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "labels": list(self.labels),
            "input": [1, MELS, FRAMES],
            "sample_rate": RATE,
            "window_frames": WINDOW,
            "order_output": False,
            "controls_output": False,
        }


class BlindFamilyPresence(BlindPresence):
    """One-vs-rest family expert for a single order-independent gate."""

    def __init__(self, label: str) -> None:
        if label not in (*LABELS, "any"):
            raise ValueError(f"unsupported Blind family: {label}")
        self.family = label
        super().__init__((label,))

    def manifest(self) -> dict:
        result = super().manifest()
        result["architecture"] = "compact-audio-resnet18-family-expert"
        result["family"] = self.family
        return result


class BlindPresenceMultiAxis(BlindPresence):
    """Preserve temporal and spectral trajectories before final pooling."""

    def __init__(self, labels: tuple[str, ...] = LABELS) -> None:
        super().__init__(labels)
        self.temporal = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.spectral = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(768, len(labels))

    @staticmethod
    def _pool(value: torch.Tensor) -> torch.Tensor:
        return torch.cat((value.mean(dim=-1), value.amax(dim=-1)), dim=1)

    def features(self, value: torch.Tensor) -> torch.Tensor:
        value = self.encode(value)
        global_features = torch.cat(
            (value.mean(dim=(-2, -1)), value.amax(dim=(-2, -1))), dim=1
        )
        temporal_features = self._pool(self.temporal(value.mean(dim=-2)))
        spectral_features = self._pool(self.spectral(value.mean(dim=-1)))
        return torch.cat((global_features, temporal_features, spectral_features), dim=1)

    def manifest(self) -> dict:
        result = super().manifest()
        result["architecture"] = "multiaxis-audio-resnet18"
        result["parameters"] = sum(parameter.numel() for parameter in self.parameters())
        return result


def aggregate_windows(probabilities: torch.Tensor) -> torch.Tensor:
    """File score is the mean of the strongest two windows, never a raw max."""

    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise ValueError(f"expected [windows, labels], got {tuple(probabilities.shape)}")
    count = min(2, probabilities.shape[0])
    return probabilities.topk(count, dim=0).values.mean(dim=0)
