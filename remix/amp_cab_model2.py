"""Long-profile-FIR Guitar-TECHS Amp+cab+mic inverse expert."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .amp_cab_model import AmpCabInverseExpert, OUTPUT_PEAK_CONTRACT, PROFILE_WIDTH, RATE


FIR_TAPS = 1025


class LongProfileFIR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.coefficients = nn.Parameter(torch.zeros(1 + PROFILE_WIDTH, FIR_TAPS))
        with torch.no_grad():
            self.coefficients[0, FIR_TAPS // 2] = 1.0

    def forward(self, wet: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        if wet.ndim != 2 or profile.shape != (wet.shape[0], PROFILE_WIDTH):
            raise ValueError("Amp-cab FIR expects Wet [batch,time] and profile [batch,2]")
        basis = torch.cat((torch.ones_like(profile[:, :1]), profile), dim=1)
        kernels = basis @ self.coefficients
        return F.conv1d(
            wet.unsqueeze(0), kernels.flip(-1).unsqueeze(1),
            padding=FIR_TAPS // 2, groups=wet.shape[0],
        ).squeeze(0)

    def initialize_profiles(self, filters: torch.Tensor) -> None:
        if filters.shape != (PROFILE_WIDTH, FIR_TAPS) or not torch.isfinite(filters).all():
            raise ValueError("profile FIR initialization must be finite [2,1025]")
        identity = torch.zeros(FIR_TAPS, dtype=filters.dtype, device=filters.device)
        identity[FIR_TAPS // 2] = 1.0
        with torch.no_grad():
            self.coefficients[0].copy_(identity)
            self.coefficients[1:].copy_(filters - identity.unsqueeze(0))


class AmpCabInverseExpert2(AmpCabInverseExpert):
    def __init__(self, hidden_size: int = 32, depth: int = 1) -> None:
        super().__init__(hidden_size, depth)
        self.profile_fir = LongProfileFIR()

    def manifest(self) -> dict:
        return {
            **super().manifest(),
            "architecture": "fit-spectral-long-profile-fir-plus-bidirectional-frame-dynamics",
            "profile_fir_taps": FIR_TAPS,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "output_peak_contract": OUTPUT_PEAK_CONTRACT,
            "sample_rate": RATE,
        }
