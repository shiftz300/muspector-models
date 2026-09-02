"""Frame-dynamics Amp inverse expert for attack and crest restoration."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .amp_model import CONTROL_WIDTH, OUTPUT_PEAK_CONTRACT, RATE, ControlFIR


class AmpFrameDynamicsInverseExpert(nn.Module):
    """Profile FIR plus offline frame-rate dynamics and transient reconstruction."""

    mechanism = "amp"

    def __init__(self, hidden_size: int = 48, depth: int = 2) -> None:
        super().__init__()
        if hidden_size < 8 or depth < 1:
            raise ValueError("invalid Amp frame-dynamics geometry")
        self.hidden_size = hidden_size
        self.depth = depth
        self.profile = ControlFIR()
        self.frame_size = 240
        self.frame_hop = 120
        self.recurrent = nn.GRU(
            6 + CONTROL_WIDTH,
            hidden_size,
            depth,
            batch_first=True,
            bidirectional=True,
        )
        self.dynamics = nn.Linear(hidden_size * 2, 2)
        self.uncertainty = nn.Linear(hidden_size * 2, 1)
        nn.init.zeros_(self.dynamics.weight)
        nn.init.zeros_(self.dynamics.bias)
        nn.init.zeros_(self.uncertainty.weight)
        nn.init.constant_(self.uncertainty.bias, -3.0)

    def forward(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if wet.ndim != 2 or controls.shape != (wet.shape[0], CONTROL_WIDTH):
            raise ValueError("Amp frame dynamics expects Wet [batch,time] and controls [batch,4]")
        if state is not None:
            raise ValueError("offline Amp frame dynamics does not accept graph or external recurrent state")
        if wet.shape[1] < self.frame_size or not torch.isfinite(wet).all():
            raise ValueError("Amp frame dynamics received invalid Wet audio")
        base = self.profile(wet, controls)
        base_frames = base.unfold(1, self.frame_size, self.frame_hop)
        wet_frames = wet.unfold(1, self.frame_size, self.frame_hop)
        base_rms = base_frames.square().mean(dim=-1).add(1.0e-8).sqrt()
        wet_rms = wet_frames.square().mean(dim=-1).add(1.0e-8).sqrt()
        base_peak = base_frames.abs().amax(dim=-1)
        wet_peak = wet_frames.abs().amax(dim=-1)
        base_crest = base_peak / base_rms.clamp_min(1.0e-5)
        base_attack = F.pad(
            torch.relu(torch.diff(torch.log(base_rms + 1.0e-6), dim=1)),
            (1, 0),
        )
        condition = controls.mul(2.0).sub(1.0).unsqueeze(1).expand(-1, base_rms.shape[1], -1)
        features = torch.cat(
            (
                torch.log(base_rms + 1.0e-6).unsqueeze(-1),
                torch.log(wet_rms + 1.0e-6).unsqueeze(-1),
                base_peak.unsqueeze(-1),
                wet_peak.unsqueeze(-1),
                base_crest.unsqueeze(-1),
                base_attack.unsqueeze(-1),
                condition,
            ),
            dim=-1,
        )
        hidden, _ = self.recurrent(features)
        frame_dynamics = self.dynamics(hidden).transpose(1, 2)
        sample_dynamics = F.interpolate(
            frame_dynamics,
            size=wet.shape[1],
            mode="linear",
            align_corners=False,
        )
        log_gain = 1.5 * torch.tanh(sample_dynamics[:, 0])
        transient_mix = torch.tanh(sample_dynamics[:, 1])
        smooth = F.avg_pool1d(base.unsqueeze(1), 33, 1, 16).squeeze(1)
        highpass = base - smooth
        restored = (base * torch.exp(log_gain) + transient_mix * highpass).clamp(
            -OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT
        )
        frame_uncertainty = self.uncertainty(hidden).transpose(1, 2)
        uncertainty = F.softplus(F.interpolate(
            frame_uncertainty,
            size=wet.shape[1],
            mode="linear",
            align_corners=False,
        ).squeeze(1)) + 1.0e-5
        return restored, uncertainty, None

    def manifest(self) -> dict:
        return {
            "schema": 3,
            "architecture": "control-fir-plus-bidirectional-frame-dynamics",
            "mechanism": self.mechanism,
            "sample_rate": RATE,
            "control_names": ["bass", "mid", "treble", "gain"],
            "control_width": CONTROL_WIDTH,
            "hidden_size": self.hidden_size,
            "depth": self.depth,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "profile_fir_taps": self.profile.taps,
            "frame_size": self.frame_size,
            "frame_hop": self.frame_hop,
            "normalization_across_time": False,
            "causal": False,
            "whole_chunk_context": True,
            "uncertainty_output": True,
            "multiplicative_dynamics_path": True,
            "transient_highpass_path": True,
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
