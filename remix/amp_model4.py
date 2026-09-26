"""Structured grey-box inverse for the Marshall speaker-output Amp domain.

The internal stage order mirrors the known amplifier signal path in reverse,
but the expert itself remains independent of any surrounding effect order.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .amp_model import (
    CONTROL_BASIS_WIDTH,
    CONTROL_WIDTH,
    OUTPUT_PEAK_CONTRACT,
    RATE,
    ControlFIR,
    control_basis,
)


class MonotonicDynamicInverse(nn.Module):
    """Control-conditioned monotonic waveshaper with explicit envelope memory."""

    def __init__(self, segments: int = 16) -> None:
        super().__init__()
        if segments < 4:
            raise ValueError("Amp waveshaper needs at least four monotonic segments")
        self.segments = segments
        # Separate positive and negative slopes permit tube-like asymmetry.  Log
        # slopes start at zero, making the stage an exact identity.
        self.log_slopes = nn.Parameter(torch.zeros(CONTROL_BASIS_WIDTH, 2, segments))
        self.log_gain = nn.Parameter(torch.zeros(CONTROL_BASIS_WIDTH))
        self.dynamic_strength = nn.Parameter(torch.zeros(CONTROL_BASIS_WIDTH))

    def forward(self, value: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2 or controls.shape != (value.shape[0], CONTROL_WIDTH):
            raise ValueError("Amp monotonic stage expects audio [batch,time] and controls [batch,4]")
        basis = control_basis(controls)
        slopes = torch.exp(torch.einsum("bk,kps->bps", basis, self.log_slopes).clamp(-2.5, 2.5))
        side = (value < 0.0).long()
        magnitude = value.abs()
        scaled = magnitude * self.segments
        bucket = torch.floor(scaled).long().clamp(0, self.segments - 1)
        selected_slopes = slopes.gather(1, side.unsqueeze(-1).expand(-1, -1, self.segments))
        selected_slope = selected_slopes.gather(2, bucket.unsqueeze(-1)).squeeze(-1)
        starts = torch.cumsum(selected_slopes / self.segments, dim=-1) - selected_slopes / self.segments
        selected_start = starts.gather(2, bucket.unsqueeze(-1)).squeeze(-1)
        local = magnitude - bucket.to(value.dtype) / self.segments
        shaped = torch.sign(value) * (selected_start + selected_slope * local)

        absolute = shaped.abs().unsqueeze(1)
        fast = F.avg_pool1d(absolute, 129, 1, 64).squeeze(1)
        slow = F.avg_pool1d(absolute, 2049, 1, 1024).squeeze(1)
        envelope_delta = (torch.log(fast + 1.0e-5) - torch.log(slow + 1.0e-5)).clamp(-2.0, 2.0)
        static_gain = basis @ self.log_gain
        dynamics = torch.tanh(basis @ self.dynamic_strength)
        gain = torch.exp((static_gain.unsqueeze(1) + dynamics.unsqueeze(1) * envelope_delta).clamp(-3.5, 3.5))
        return shaped * gain


class AmpStructuredInverseExpert(nn.Module):
    """Output profile -> power stage -> tone stack -> preamp stage inverse."""

    mechanism = "amp"

    def __init__(self, hidden_size: int = 0, depth: int = 0) -> None:
        super().__init__()
        # Kept in the constructor for the shared trainer CLI; this architecture
        # deliberately has no opaque hidden network or depth hyperparameter.
        del hidden_size, depth
        self.output_profile_inverse = ControlFIR(129)
        self.power_inverse = MonotonicDynamicInverse(16)
        self.tone_stack_inverse = ControlFIR(257)
        self.preamp_inverse = MonotonicDynamicInverse(16)
        self.uncertainty_logit = nn.Parameter(torch.tensor(-3.0))

    @property
    def profile(self) -> ControlFIR:
        """Compatibility hook for fit-only tone-profile warm starts."""
        return self.tone_stack_inverse

    def forward(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if wet.ndim != 2 or controls.shape != (wet.shape[0], CONTROL_WIDTH):
            raise ValueError("structured Amp inverse expects Wet [batch,time] and controls [batch,4]")
        if state is not None:
            raise ValueError("structured Amp inverse accepts no graph or external recurrent state")
        if not torch.isfinite(wet).all():
            raise ValueError("structured Amp inverse received non-finite Wet audio")
        value, _ = self.forward_stages(wet, controls)
        restored = value.clamp(-OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT)
        uncertainty = F.softplus(self.uncertainty_logit).expand_as(restored) + 1.0e-5
        return restored, uncertainty, None

    def forward_stages(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final input estimate and the supervised preamp-tap estimate."""
        value = self.output_profile_inverse(wet, controls)
        value = self.power_inverse(value, controls)
        preamp = self.tone_stack_inverse(value, controls)
        restored = self.preamp_inverse(preamp, controls)
        return restored, preamp

    def manifest(self) -> dict:
        return {
            "schema": 4,
            "architecture": "structured-output-profile-power-tone-stack-preamp-inverse",
            "mechanism": self.mechanism,
            "sample_rate": RATE,
            "control_names": ["bass", "mid", "treble", "gain"],
            "control_width": CONTROL_WIDTH,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "internal_reverse_stages": [
                "output-profile-inverse",
                "monotonic-dynamic-power-inverse",
                "tone-stack-inverse",
                "monotonic-dynamic-preamp-inverse",
            ],
            "output_profile_fir_taps": self.output_profile_inverse.taps,
            "tone_stack_fir_taps": self.tone_stack_inverse.taps,
            "monotonic_segments_per_polarity": self.power_inverse.segments,
            "explicit_envelope_memory": True,
            "opaque_temporal_network": False,
            "normalization_across_time": False,
            "causal": False,
            "uncertainty_output": True,
            "graph_order_input": False,
            "neighbor_effect_input": False,
            "recurrent_state_input": False,
            "output_peak_contract": OUTPUT_PEAK_CONTRACT,
            "current_device_scope": "Marshall JVM410H OD1 speaker output only",
            "structural_identifiability_claim": False,
            "supports_preamp_tap_supervision": True,
            "cabinet_scope": "speaker-output only; no cabinet or microphone removal",
        }
