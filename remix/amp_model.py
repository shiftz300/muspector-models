"""Order-independent profile-conditioned Amp inverse expert."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


RATE = 44_100
CONTROL_WIDTH = 4
CONTROL_BASIS_WIDTH = 15
FIR_TAPS = 257
OUTPUT_PEAK_CONTRACT = 0.98


def control_basis(controls: torch.Tensor) -> torch.Tensor:
    if controls.ndim != 2 or controls.shape[1] != CONTROL_WIDTH:
        raise ValueError("Amp controls must be [batch,4] Bass/Mid/Treble/Gain")
    if not torch.isfinite(controls).all() or torch.any(controls < 0.0) or torch.any(controls > 1.0):
        raise ValueError("Amp controls must be finite and normalized to 0..1")
    bass, mid, treble, gain = controls.unbind(dim=1)
    return torch.stack(
        (
            torch.ones_like(bass),
            bass,
            mid,
            treble,
            gain,
            bass.square(),
            mid.square(),
            treble.square(),
            gain.square(),
            bass * mid,
            bass * treble,
            mid * treble,
            bass * gain,
            mid * gain,
            treble * gain,
        ),
        dim=1,
    )


class ControlFIR(nn.Module):
    """Smooth control-conditioned offline FIR initialized as exact identity."""

    def __init__(self, taps: int = FIR_TAPS) -> None:
        super().__init__()
        if taps < 3 or taps % 2 != 1:
            raise ValueError("Amp FIR must have an odd number of at least three taps")
        self.taps = taps
        self.coefficients = nn.Parameter(torch.zeros(CONTROL_BASIS_WIDTH, taps))
        with torch.no_grad():
            self.coefficients[0, taps // 2] = 1.0

    def forward(self, wet: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        if wet.ndim != 2 or controls.shape != (wet.shape[0], CONTROL_WIDTH):
            raise ValueError("Amp FIR expects Wet [batch,time] and controls [batch,4]")
        kernels = control_basis(controls) @ self.coefficients
        grouped = wet.unsqueeze(0)
        restored = F.conv1d(
            grouped,
            kernels.flip(-1).unsqueeze(1),
            padding=self.taps // 2,
            groups=wet.shape[0],
        )
        return restored.squeeze(0)


class AmpInverseExpert(nn.Module):
    """JVM-compatible Amp subdomain: tone/profile FIR plus bounded local TCN."""

    mechanism = "amp"

    def __init__(self, hidden_size: int = 24, depth: int = 7, fir_taps: int = FIR_TAPS) -> None:
        super().__init__()
        if hidden_size < 4 or depth < 1:
            raise ValueError("invalid Amp inverse geometry")
        self.hidden_size = hidden_size
        self.depth = depth
        self.profile = ControlFIR(fir_taps)
        self.dilations = tuple(2**index for index in range(depth))
        self.input = nn.Conv1d(5 + CONTROL_WIDTH, hidden_size, 1)
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
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if wet.ndim != 2 or controls.shape != (wet.shape[0], CONTROL_WIDTH):
            raise ValueError("Amp inverse expects Wet [batch,time] and controls [batch,4]")
        if state is not None:
            raise ValueError("offline Amp inverse does not accept graph or recurrent state")
        if not torch.isfinite(wet).all():
            raise ValueError("Amp Wet contains non-finite samples")
        base = self.profile(wet, controls)
        difference = F.pad(base[:, 1:] - base[:, :-1], (1, 0))
        envelope = F.avg_pool1d(base.abs().unsqueeze(1), 129, 1, 64).squeeze(1)
        condition = controls.mul(2.0).sub(1.0).unsqueeze(1).expand(-1, wet.shape[1], -1)
        features = torch.cat(
            (
                wet.unsqueeze(-1),
                wet.abs().unsqueeze(-1),
                base.unsqueeze(-1),
                difference.unsqueeze(-1),
                envelope.unsqueeze(-1),
                condition,
            ),
            dim=-1,
        ).transpose(1, 2)
        hidden = torch.tanh(self.input(features))
        for temporal, mix in zip(self.temporal, self.mix, strict=True):
            value, gate = temporal(hidden).chunk(2, dim=1)
            hidden = hidden + 0.25 * mix(torch.tanh(value) * torch.sigmoid(gate))
        correction = 0.5 * torch.tanh(self.correction(hidden).squeeze(1))
        restored = (base + correction).clamp(-OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT)
        uncertainty = F.softplus(self.uncertainty(hidden).squeeze(1)) + 1.0e-5
        return restored, uncertainty, None

    def manifest(self) -> dict:
        local_tcn_frames = 1 + 6 * sum(self.dilations)
        return {
            "schema": 1,
            "architecture": "control-conditioned-fir-plus-local-tcn",
            "mechanism": self.mechanism,
            "sample_rate": RATE,
            "control_names": ["bass", "mid", "treble", "gain"],
            "control_width": CONTROL_WIDTH,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "profile_fir_taps": self.profile.taps,
            "local_receptive_field_frames": self.profile.taps + local_tcn_frames - 1,
            "normalization_across_time": False,
            "causal": False,
            "uncertainty_output": True,
            "graph_order_input": False,
            "neighbor_effect_input": False,
            "recurrent_state_input": False,
            "output_peak_contract": OUTPUT_PEAK_CONTRACT,
            "current_device_scope": "Marshall JVM410H OD1; B/M/T and released Gain controls",
            "gain_fit_domain": [0.0, 2.0, 4.0, 5.0, 10.0],
            "gain_development_holdouts": [1.0, 8.0],
            "quarantined_control": "Gain=6 naming contradiction in source archive",
            "cabinet_scope": "speaker-output only; no cabinet or microphone removal",
        }
