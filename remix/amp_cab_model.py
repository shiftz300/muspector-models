"""Named-profile Guitar-TECHS Amp+cab+mic inverse expert."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


RATE = 48_000
PROFILE_WIDTH = 2
FIR_TAPS = 257
OUTPUT_PEAK_CONTRACT = 0.98


class ProfileFIR(nn.Module):
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


class AmpCabInverseExpert(nn.Module):
    mechanism = "amp"

    def __init__(self, hidden_size: int = 48, depth: int = 2) -> None:
        super().__init__()
        if hidden_size < 8 or depth < 1:
            raise ValueError("invalid Amp-cab inverse geometry")
        self.hidden_size = hidden_size
        self.depth = depth
        self.profile_fir = ProfileFIR()
        self.frame_size = 240
        self.frame_hop = 120
        self.recurrent = nn.GRU(
            6 + PROFILE_WIDTH, hidden_size, depth,
            batch_first=True, bidirectional=True,
        )
        self.dynamics = nn.Linear(hidden_size * 2, 2)
        self.uncertainty = nn.Linear(hidden_size * 2, 1)
        nn.init.zeros_(self.dynamics.weight); nn.init.zeros_(self.dynamics.bias)
        nn.init.zeros_(self.uncertainty.weight); nn.init.constant_(self.uncertainty.bias, -3.0)

    def forward(
        self, wet: torch.Tensor, profile: torch.Tensor, state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if state is not None:
            raise ValueError("offline Amp-cab inverse accepts no graph or external recurrent state")
        if wet.ndim != 2 or profile.shape != (wet.shape[0], PROFILE_WIDTH):
            raise ValueError("Amp-cab inverse expects Wet [batch,time] and profile [batch,2]")
        if wet.shape[1] < self.frame_size or not torch.isfinite(wet).all():
            raise ValueError("invalid Amp-cab Wet audio")
        base = self.profile_fir(wet, profile)
        base_frames = base.unfold(1, self.frame_size, self.frame_hop)
        wet_frames = wet.unfold(1, self.frame_size, self.frame_hop)
        base_rms = base_frames.square().mean(-1).add(1.0e-8).sqrt()
        wet_rms = wet_frames.square().mean(-1).add(1.0e-8).sqrt()
        base_peak = base_frames.abs().amax(-1)
        wet_peak = wet_frames.abs().amax(-1)
        crest = base_peak / base_rms.clamp_min(1.0e-5)
        attack = F.pad(torch.relu(torch.diff(torch.log(base_rms + 1.0e-6), dim=1)), (1, 0))
        condition = profile.unsqueeze(1).expand(-1, base_rms.shape[1], -1)
        features = torch.cat((
            torch.log(base_rms + 1.0e-6).unsqueeze(-1),
            torch.log(wet_rms + 1.0e-6).unsqueeze(-1),
            base_peak.unsqueeze(-1), wet_peak.unsqueeze(-1),
            crest.unsqueeze(-1), attack.unsqueeze(-1), condition,
        ), dim=-1)
        hidden, _ = self.recurrent(features)
        dynamics = F.interpolate(
            self.dynamics(hidden).transpose(1, 2), size=wet.shape[1],
            mode="linear", align_corners=False,
        )
        log_gain = 1.5 * torch.tanh(dynamics[:, 0])
        transient_mix = torch.tanh(dynamics[:, 1])
        smooth = F.avg_pool1d(base.unsqueeze(1), 33, 1, 16).squeeze(1)
        restored = (base * torch.exp(log_gain) + transient_mix * (base - smooth)).clamp(
            -OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT
        )
        uncertainty = F.softplus(F.interpolate(
            self.uncertainty(hidden).transpose(1, 2), size=wet.shape[1],
            mode="linear", align_corners=False,
        ).squeeze(1)) + 1.0e-5
        return restored, uncertainty, None

    def manifest(self) -> dict:
        return {
            "schema": 1,
            "architecture": "named-profile-fir-plus-bidirectional-frame-dynamics",
            "mechanism": "amp",
            "scope": "Amp+cab+mic to DI",
            "sample_rate": RATE,
            "hidden_size": self.hidden_size,
            "depth": self.depth,
            "profile_ids": ["P1-orange-cr60-sm57", "P2-yamaha-yb15-at2020"],
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "graph_order_input": False,
            "neighbor_effect_input": False,
            "recurrent_state_input": False,
            "p3_input_or_profile": False,
            "causal": False,
            "whole_chunk_context": True,
            "output_peak_contract": OUTPUT_PEAK_CONTRACT,
        }
