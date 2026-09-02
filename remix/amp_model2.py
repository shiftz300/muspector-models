"""Order-independent Amp inverse with explicit dynamics restoration."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .amp_model import CONTROL_WIDTH, OUTPUT_PEAK_CONTRACT, RATE, ControlFIR


class AmpDynamicsInverseExpert(nn.Module):
    """Tone/profile FIR followed by bounded multiplicative dynamics recovery."""

    mechanism = "amp"

    def __init__(self, hidden_size: int = 32, depth: int = 10) -> None:
        super().__init__()
        if hidden_size < 8 or depth < 3:
            raise ValueError("invalid Amp dynamics inverse geometry")
        self.hidden_size = hidden_size
        self.depth = depth
        self.profile = ControlFIR()
        self.dilations = tuple(2**index for index in range(depth))
        feature_width = 8 + CONTROL_WIDTH
        self.input = nn.Conv1d(feature_width, hidden_size, 1)
        self.temporal = nn.ModuleList(
            nn.Conv1d(hidden_size, hidden_size * 2, 7, padding=3 * dilation, dilation=dilation)
            for dilation in self.dilations
        )
        self.mix = nn.ModuleList(nn.Conv1d(hidden_size, hidden_size, 1) for _ in self.dilations)
        self.log_gain = nn.Conv1d(hidden_size, 1, 1)
        self.correction = nn.Conv1d(hidden_size, 1, 1)
        self.uncertainty = nn.Conv1d(hidden_size, 1, 1)
        nn.init.zeros_(self.log_gain.weight)
        nn.init.zeros_(self.log_gain.bias)
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
            raise ValueError("Amp dynamics inverse expects Wet [batch,time] and controls [batch,4]")
        if state is not None:
            raise ValueError("offline Amp dynamics inverse does not accept graph or recurrent state")
        if not torch.isfinite(wet).all():
            raise ValueError("Amp Wet contains non-finite samples")
        base = self.profile(wet, controls)
        difference = F.pad(base[:, 1:] - base[:, :-1], (1, 0))
        absolute = base.abs().unsqueeze(1)
        fast_envelope = F.avg_pool1d(absolute, 129, 1, 64).squeeze(1)
        slow_envelope = F.avg_pool1d(absolute, 2049, 1, 1024).squeeze(1)
        log_fast = torch.log(fast_envelope + 1.0e-5)
        compression_hint = log_fast - torch.log(slow_envelope + 1.0e-5)
        condition = controls.mul(2.0).sub(1.0).unsqueeze(1).expand(-1, wet.shape[1], -1)
        features = torch.cat(
            (
                wet.unsqueeze(-1),
                wet.abs().unsqueeze(-1),
                base.unsqueeze(-1),
                difference.unsqueeze(-1),
                fast_envelope.unsqueeze(-1),
                slow_envelope.unsqueeze(-1),
                log_fast.unsqueeze(-1),
                compression_hint.unsqueeze(-1),
                condition,
            ),
            dim=-1,
        ).transpose(1, 2)
        hidden = torch.tanh(self.input(features))
        for temporal, mix in zip(self.temporal, self.mix, strict=True):
            value, gate = temporal(hidden).chunk(2, dim=1)
            hidden = hidden + 0.20 * mix(torch.tanh(value) * torch.sigmoid(gate))
        log_gain = 1.5 * torch.tanh(self.log_gain(hidden).squeeze(1))
        correction = 0.20 * torch.tanh(self.correction(hidden).squeeze(1))
        restored = (base * torch.exp(log_gain) + correction).clamp(
            -OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT
        )
        uncertainty = F.softplus(self.uncertainty(hidden).squeeze(1)) + 1.0e-5
        return restored, uncertainty, None

    def manifest(self) -> dict:
        local_tcn_frames = 1 + 6 * sum(self.dilations)
        return {
            "schema": 2,
            "architecture": "control-fir-plus-envelope-conditioned-multiplicative-local-tcn",
            "mechanism": self.mechanism,
            "sample_rate": RATE,
            "control_names": ["bass", "mid", "treble", "gain"],
            "control_width": CONTROL_WIDTH,
            "hidden_size": self.hidden_size,
            "depth": self.depth,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "profile_fir_taps": self.profile.taps,
            "fast_envelope_frames": 129,
            "slow_envelope_frames": 2049,
            "local_receptive_field_frames": self.profile.taps + 2049 + local_tcn_frames - 2,
            "normalization_across_time": False,
            "causal": False,
            "uncertainty_output": True,
            "multiplicative_dynamics_path": True,
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
