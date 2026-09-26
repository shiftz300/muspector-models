"""Stage-supervised structured inverse for the Marshall Amp domain."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .amp_model import CONTROL_WIDTH, OUTPUT_PEAK_CONTRACT, RATE, ControlFIR
from .amp_model4 import MonotonicDynamicInverse


class AmpStageSupervisedInverseExpert(nn.Module):
    """Two identified inverse sections joined at the measured preamp tap."""

    mechanism = "amp"

    def __init__(self) -> None:
        super().__init__()
        self.output_profile_inverse = ControlFIR(129)
        self.power_inverse = MonotonicDynamicInverse(16)
        self.tone_stack_inverse = ControlFIR(257)
        self.preamp_inverse = MonotonicDynamicInverse(16)
        self.input_profile_inverse = ControlFIR(257)
        self.uncertainty_logit = nn.Parameter(torch.tensor(-3.0))

    def inverse_power_tone(self, wet: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        value = self.output_profile_inverse(wet, controls)
        value = self.power_inverse(value, controls)
        return self.tone_stack_inverse(value, controls)

    def inverse_preamp(self, preamp: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        value = self.preamp_inverse(preamp, controls)
        return self.input_profile_inverse(value, controls)

    def forward_stages(self, wet: torch.Tensor, controls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        preamp = self.inverse_power_tone(wet, controls)
        return self.inverse_preamp(preamp, controls), preamp

    def forward(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if wet.ndim != 2 or controls.shape != (wet.shape[0], CONTROL_WIDTH):
            raise ValueError("stage-supervised Amp inverse expects Wet [batch,time] and controls [batch,4]")
        if state is not None:
            raise ValueError("stage-supervised Amp inverse accepts no graph or external recurrent state")
        if not torch.isfinite(wet).all():
            raise ValueError("stage-supervised Amp inverse received non-finite Wet audio")
        restored, _ = self.forward_stages(wet, controls)
        restored = restored.clamp(-OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT)
        uncertainty = F.softplus(self.uncertainty_logit).expand_as(restored) + 1.0e-5
        return restored, uncertainty, None

    def stage_parameters(self, stage: str):
        modules = {
            "power-tone": (self.output_profile_inverse, self.power_inverse, self.tone_stack_inverse),
            "preamp": (self.preamp_inverse, self.input_profile_inverse),
        }
        if stage not in modules:
            raise ValueError(f"unknown Amp stage: {stage}")
        for module in modules[stage]:
            yield from module.parameters()

    def manifest(self) -> dict:
        return {
            "schema": 5,
            "architecture": "stage-supervised-output-power-tone-preamp-input-inverse",
            "mechanism": self.mechanism,
            "sample_rate": RATE,
            "control_names": ["bass", "mid", "treble", "gain"],
            "control_width": CONTROL_WIDTH,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "measured_intermediate": "preamp",
            "internal_reverse_stages": [
                "output-profile-inverse",
                "monotonic-dynamic-power-inverse",
                "tone-stack-inverse",
                "measured-preamp-boundary",
                "monotonic-dynamic-preamp-inverse",
                "input-profile-inverse",
            ],
            "stagewise_training_required": True,
            "opaque_temporal_network": False,
            "normalization_across_time": False,
            "causal": False,
            "uncertainty_output": True,
            "graph_order_input": False,
            "neighbor_effect_input": False,
            "recurrent_state_input": False,
            "output_peak_contract": OUTPUT_PEAK_CONTRACT,
            "current_device_scope": "Marshall JVM410H OD1 speaker output only",
            "cabinet_scope": "speaker-output only; no cabinet or microphone removal",
            "structural_identifiability_claim": "bounded-by-measured-preamp-tap",
        }
