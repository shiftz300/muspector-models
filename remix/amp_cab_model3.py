"""Long-profile-FIR plus local-TCN Guitar-TECHS Amp+cab+mic inverse."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .amp_cab_model import OUTPUT_PEAK_CONTRACT, PROFILE_WIDTH, RATE
from .amp_cab_model2 import FIR_TAPS, LongProfileFIR


class AmpCabInverseExpert3(nn.Module):
    mechanism = "amp"

    def __init__(self, hidden_size: int = 24, depth: int = 6) -> None:
        super().__init__()
        if hidden_size < 8 or depth < 2:
            raise ValueError("invalid Amp-cab local TCN geometry")
        self.hidden_size = hidden_size
        self.depth = depth
        self.profile_fir = LongProfileFIR()
        self.dilations = tuple(2 ** index for index in range(depth))
        self.input = nn.Conv1d(5 + PROFILE_WIDTH, hidden_size, 1)
        self.temporal = nn.ModuleList(
            nn.Conv1d(hidden_size, hidden_size * 2, 7, padding=3 * dilation, dilation=dilation)
            for dilation in self.dilations
        )
        self.mix = nn.ModuleList(nn.Conv1d(hidden_size, hidden_size, 1) for _ in self.dilations)
        self.correction = nn.Conv1d(hidden_size, 1, 1)
        self.uncertainty = nn.Conv1d(hidden_size, 1, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        nn.init.zeros_(self.uncertainty.weight)
        nn.init.constant_(self.uncertainty.bias, -3.0)

    def forward(
        self, wet: torch.Tensor, profile: torch.Tensor, state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if state is not None:
            raise ValueError("offline Amp-cab local TCN accepts no graph or external state")
        if wet.ndim != 2 or profile.shape != (wet.shape[0], PROFILE_WIDTH):
            raise ValueError("Amp-cab local TCN expects Wet [batch,time] and profile [batch,2]")
        if not torch.isfinite(wet).all():
            raise ValueError("invalid Amp-cab Wet audio")
        base = self.profile_fir(wet, profile)
        difference = F.pad(base[:, 1:] - base[:, :-1], (1, 0))
        envelope = F.avg_pool1d(base.abs().unsqueeze(1), 129, 1, 64).squeeze(1)
        condition = profile.mul(2.0).sub(1.0).unsqueeze(1).expand(-1, wet.shape[1], -1)
        features = torch.cat((
            wet.unsqueeze(-1), wet.abs().unsqueeze(-1), base.unsqueeze(-1),
            difference.unsqueeze(-1), envelope.unsqueeze(-1), condition,
        ), dim=-1).transpose(1, 2)
        hidden = torch.tanh(self.input(features))
        for temporal, mix in zip(self.temporal, self.mix, strict=True):
            value, gate = temporal(hidden).chunk(2, dim=1)
            hidden = hidden + 0.25 * mix(torch.tanh(value) * torch.sigmoid(gate))
        correction = 0.25 * torch.tanh(self.correction(hidden).squeeze(1))
        restored = (base + correction).clamp(-OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT)
        uncertainty = F.softplus(self.uncertainty(hidden).squeeze(1)) + 1.0e-5
        return restored, uncertainty, None

    def manifest(self) -> dict:
        local_frames = 1 + 6 * sum(self.dilations)
        return {
            "schema": 1,
            "architecture": "fit-spectral-long-profile-fir-plus-local-tcn",
            "mechanism": "amp",
            "scope": "Amp+cab+mic to DI",
            "sample_rate": RATE,
            "hidden_size": self.hidden_size,
            "depth": self.depth,
            "profile_ids": ["P1-orange-cr60-sm57", "P2-yamaha-yb15-at2020"],
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "profile_fir_taps": FIR_TAPS,
            "local_receptive_field_frames": FIR_TAPS + local_frames - 1,
            "graph_order_input": False,
            "neighbor_effect_input": False,
            "recurrent_state_input": False,
            "p3_input_or_profile": False,
            "causal": False,
            "whole_chunk_context": False,
            "output_peak_contract": OUTPUT_PEAK_CONTRACT,
        }
