"""Eight-stage product gray-box Amp inverse for rusty-amp supervision."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .egdb_pg_amp_model import OUTPUT_PEAK_CONTRACT, RATE
from .egdb_pg_amp_model10 import _ToneCondition
from .egdb_pg_tone_encoder import WetToneEncoder
from .open_riff_box_stage_model import (
    _LinearInverse,
    _LinearNonlinearInverse,
    _NonlinearInverse,
)


class RustyAmpStagewiseGrayBoxInverse(nn.Module):
    mechanism = "amp"

    def __init__(self, channels: int = 24, depth: int = 8, condition_size: int = 64) -> None:
        super().__init__()
        del channels
        if depth < 4 or condition_size < 16:
            raise ValueError("invalid rusty-amp gray-box geometry")
        self.depth = depth
        self.condition_size = condition_size
        self.segments = max(8, depth * 2)
        self.tone_encoder = WetToneEncoder(64)
        self.condition = _ToneCondition(condition_size)
        self.inverse_stages = nn.ModuleList((
            _LinearInverse(condition_size, 129, 1.0),
            _LinearNonlinearInverse(condition_size, 65, self.segments),
            _NonlinearInverse(condition_size, self.segments),
            _NonlinearInverse(condition_size, self.segments),
            _LinearInverse(condition_size, 129, 1.0),
            _LinearInverse(condition_size, 257, 1.0),
            _LinearNonlinearInverse(condition_size, 65, self.segments),
            _LinearInverse(condition_size, 129, 1.0),
        ))

    def stage_outputs(
        self, wet: torch.Tensor, *, tone_reference: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        if wet.ndim != 2 or not torch.isfinite(wet).all():
            raise ValueError("rusty-amp gray box expects finite Wet [batch,time]")
        reference = wet if tone_reference is None else tone_reference
        condition = self.condition(self.tone_encoder(reference))
        value = wet
        outputs = []
        for stage in self.inverse_stages:
            value = stage(value, condition)
            outputs.append(value)
        return outputs

    def inverse_output_to_preamp(self, wet: torch.Tensor) -> torch.Tensor:
        condition = self.condition(self.tone_encoder(wet))
        value = wet
        for stage in self.inverse_stages[:6]:
            value = stage(value, condition)
        return value

    def inverse_preamp_to_clean(
        self, preamp: torch.Tensor, tone_reference: torch.Tensor
    ) -> torch.Tensor:
        condition = self.condition(self.tone_encoder(tone_reference))
        value = preamp
        for stage in self.inverse_stages[6:]:
            value = stage(value, condition)
        return value

    def forward_stages(
        self, wet: torch.Tensor, controls: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del controls
        outputs = self.stage_outputs(wet)
        return outputs[-1].clamp(-OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT), outputs[5]

    def stage_parameters(self, phase: str):
        if phase == "output-preamp":
            modules = (self.tone_encoder, self.condition, *self.inverse_stages[:6])
        elif phase == "preamp-input":
            modules = (*self.inverse_stages[6:],)
        elif phase == "joint":
            modules = (self,)
        else:
            raise ValueError(f"unknown rusty-amp phase: {phase}")
        for module in modules:
            yield from module.parameters()

    def forward(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        *,
        tone_reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if state is not None:
            raise ValueError("rusty-amp gray box accepts no graph or external state")
        if controls is not None and controls.shape != (wet.shape[0], 4):
            raise ValueError("optional audit controls must be [batch,4]")
        restored = self.stage_outputs(wet, tone_reference=tone_reference)[-1]
        restored = restored.clamp(-OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT)
        uncertainty = F.softplus(restored.new_tensor(-3.0)).expand_as(restored) + 1.0e-5
        return restored, uncertainty, None

    def manifest(self) -> dict:
        return {
            "schema": 17,
            "architecture": "eight-stage-product-supervised-gray-box-inverse",
            "mechanism": "amp",
            "scope": "no-cabinet Amp pretraining followed by product audio fine-tuning",
            "sample_rate": RATE,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "condition_size": self.condition_size,
            "depth": self.depth,
            "segments": self.segments,
            "internal_reverse_stages": [
                "output-and-presence", "speaker-load", "output-transformer",
                "power-amp", "voice-balance", "tone-stack", "preamp", "front-end",
            ],
            "deep_supervision_pretraining": True,
            "product_pretraining_source": "rusty-amp-stage-renderer",
            "profile_id_input": False,
            "gain_category_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
            "recurrent_state_input": False,
            "clean_or_oracle_input": False,
            "wet_only_inference": True,
            "audit_controls_ignored": True,
            "causal": False,
            "output_peak_contract": OUTPUT_PEAK_CONTRACT,
        }
