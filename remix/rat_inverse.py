"""RAT-specific inverse features with explicit gain-invariant spectral transfer."""

from __future__ import annotations

import torch


SPECTRAL_POINTS = 96
LEVEL_FEATURES = 12
FEATURES = SPECTRAL_POINTS * 3 + LEVEL_FEATURES


@torch.no_grad()
def rat_spectral_features(dry: torch.Tensor, wet: torch.Tensor) -> torch.Tensor:
    if dry.ndim != 2 or wet.shape != dry.shape:
        raise ValueError("RAT inverse features expect matching [batch,time] audio")
    window = torch.hann_window(4_096, device=dry.device, dtype=dry.dtype)
    spectra = []
    for audio in (dry, wet):
        magnitude = torch.stft(
            audio,
            n_fft=4_096,
            hop_length=2_048,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        ).abs().mean(dim=2)
        indices = torch.round(
            torch.logspace(
                torch.log10(torch.tensor(2.0)).item(),
                torch.log10(torch.tensor(1_706.0)).item(),
                SPECTRAL_POINTS,
                device=audio.device,
            )
        ).long()
        spectra.append(torch.log(magnitude[:, indices] + 1.0e-5))
    dry_spectrum, wet_spectrum = spectra
    transfer = wet_spectrum - dry_spectrum
    transfer_centered = transfer - transfer.mean(dim=1, keepdim=True)
    dry_centered = dry_spectrum - dry_spectrum.mean(dim=1, keepdim=True)
    wet_centered = wet_spectrum - wet_spectrum.mean(dim=1, keepdim=True)

    delta = wet - dry
    values = []
    for audio in (dry, wet, delta):
        rms = audio.square().mean(dim=1).sqrt()
        absolute_mean = audio.abs().mean(dim=1)
        peak = audio.abs().amax(dim=1)
        crest = peak / rms.clamp_min(1.0e-6)
        values.extend((rms, absolute_mean, peak, crest))
    levels = torch.stack(values, dim=1)
    levels[:, :3] = torch.log(levels[:, :3] + 1.0e-6)
    levels[:, 4:7] = torch.log(levels[:, 4:7] + 1.0e-6)
    levels[:, 8:11] = torch.log(levels[:, 8:11] + 1.0e-6)
    return torch.cat((transfer_centered, wet_centered, dry_centered, levels), dim=1)


class RatSpectralControlEstimator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(FEATURES),
            torch.nn.Linear(FEATURES, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
