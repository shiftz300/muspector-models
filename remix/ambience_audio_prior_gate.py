"""Audio-prior gate for releasing a known-profile Reverb gray-box candidate."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .ambience2 import CONTROL_WIDTH
from .ambience5_profile_direct import PROFILE_CHANNELS, PROFILE_N_FFT


FEATURE_BINS = 128
FEATURE_FRAMES = 128


def candidate_audio_features(
    wet: torch.Tensor,
    candidate: torch.Tensor,
    profile: torch.Tensor,
) -> torch.Tensor:
    """Runtime features from Wet, candidate and current profile; no Clean."""
    if wet.ndim != 1 or candidate.shape != wet.shape:
        raise ValueError("audio-prior features expect matching mono Wet and candidate")
    if profile.shape != (PROFILE_CHANNELS, PROFILE_N_FFT // 2 + 1):
        raise ValueError("audio-prior profile geometry changed")
    if not all(torch.isfinite(value).all() for value in (wet, candidate, profile)):
        raise ValueError("audio-prior features require finite tensors")
    window = torch.hann_window(PROFILE_N_FFT, device=wet.device, dtype=wet.dtype)
    wet_spectrum = torch.stft(
        wet, PROFILE_N_FFT, 512, window=window, center=True,
        pad_mode="constant", return_complex=True,
    )
    candidate_spectrum = torch.stft(
        candidate, PROFILE_N_FFT, 512, window=window, center=True,
        pad_mode="constant", return_complex=True,
    )
    scale = wet_spectrum.abs().mean().clamp_min(1.0e-6)
    wet_log = torch.log1p(wet_spectrum.abs() / scale)
    candidate_log = torch.log1p(candidate_spectrum.abs() / scale)
    log_ratio = (candidate_log - wet_log).clamp(-3.0, 3.0)
    phase = (
        candidate_spectrum * wet_spectrum.conj()
        / (candidate_spectrum.abs() * wet_spectrum.abs()).clamp_min(1.0e-6)
    ).real
    audio = torch.stack((wet_log, candidate_log, log_ratio, phase), dim=0)
    audio = F.interpolate(
        audio[None], size=(FEATURE_BINS, FEATURE_FRAMES),
        mode="bilinear", align_corners=False,
    )[0]
    profile_map = F.interpolate(
        profile[:, :, None][None], size=(FEATURE_BINS, FEATURE_FRAMES),
        mode="bilinear", align_corners=False,
    )[0]
    return torch.cat((audio, profile_map), dim=0)


class ReverbCandidateAudioPriorGate(nn.Module):
    """Judge a gray-box candidate using dry-audio structure plus current RIR."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        if channels < 8:
            raise ValueError("audio-prior gate is too narrow")
        widths = (channels, channels * 2, channels * 3, channels * 4)
        layers = []
        incoming = 4 + PROFILE_CHANNELS
        for width in widths:
            layers.extend((
                nn.Conv2d(incoming, width, 5, stride=2, padding=2),
                nn.GroupNorm(4, width),
                nn.GELU(),
                nn.Conv2d(width, width, 3, padding=1),
                nn.GELU(),
            ))
            incoming = width
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(2 * widths[-1] + CONTROL_WIDTH + 1, 64),
            nn.GELU(), nn.Dropout(0.10), nn.Linear(64, 1),
        )
        self.channels = channels

    def forward(
        self,
        features: torch.Tensor,
        controls: torch.Tensor,
        exact_mode: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1:] != (
            4 + PROFILE_CHANNELS, FEATURE_BINS, FEATURE_FRAMES
        ):
            raise ValueError("audio-prior feature batch geometry changed")
        if controls.shape != (features.shape[0], CONTROL_WIDTH):
            raise ValueError("audio-prior control geometry changed")
        if exact_mode.shape != (features.shape[0], 1):
            raise ValueError("audio-prior mode geometry changed")
        hidden = self.encoder(features)
        pooled = torch.cat((hidden.mean(dim=(-2, -1)), hidden.amax(dim=(-2, -1))), dim=1)
        return self.head(torch.cat((pooled, controls, exact_mode), dim=1)).squeeze(1)

    def manifest(self) -> dict:
        return {
            "schema": 1,
            "architecture": "wet-candidate-profile-time-frequency-audio-prior-gate",
            "mechanism": "ambience",
            "channels": self.channels,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "feature_shape": [4 + PROFILE_CHANNELS, FEATURE_BINS, FEATURE_FRAMES],
            "inference_inputs": "Wet, analytic candidate, current RIR/profile and controls only",
            "clean_input_at_inference": False,
            "clean_used_for_fit_labels": True,
            "chain_order_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
            "output_contract": "hard bypass unless frozen score threshold admits candidate",
        }
