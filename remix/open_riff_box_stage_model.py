"""Nine-stage gray-box inverse for isolated architecture selection."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .egdb_pg_amp_model import OUTPUT_PEAK_CONTRACT, RATE
from .egdb_pg_amp_model10 import _ToneCondition, _ToneFIR, _ToneMonotonicInverse
from .egdb_pg_tone_encoder import WetToneEncoder


class _LinearInverse(nn.Module):
    def __init__(self, condition_size: int, taps: int, initial_scale: float = 1.0) -> None:
        super().__init__()
        self.filter = _ToneFIR(condition_size, taps, initial_scale, symmetric=False)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.filter(value, condition)


class _NonlinearInverse(nn.Module):
    def __init__(self, condition_size: int, segments: int) -> None:
        super().__init__()
        self.shape = _ToneMonotonicInverse(condition_size, segments)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.shape(value, condition)


class _LinearNonlinearInverse(nn.Module):
    def __init__(self, condition_size: int, taps: int, segments: int) -> None:
        super().__init__()
        self.filter = _ToneFIR(condition_size, taps, 1.0, symmetric=False)
        self.shape = _ToneMonotonicInverse(condition_size, segments)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.shape(self.filter(value, condition), condition)


class OpenRiffBoxStagewiseGrayBoxInverse(nn.Module):
    """Reverse transformer-to-input stages with an intermediate target per block."""

    mechanism = "amp"

    def __init__(
        self,
        channels: int = 24,
        depth: int = 8,
        condition_size: int = 64,
        *,
        segments: int | None = None,
        research_probe: bool = False,
    ) -> None:
        super().__init__()
        del channels
        if depth < 4 or condition_size < 16:
            raise ValueError("invalid stagewise gray-box geometry")
        segments = max(8, depth * 2) if segments is None else segments
        self.depth = depth
        self.condition_size = condition_size
        self.segments = segments
        self.research_probe = research_probe
        self.tone_encoder = WetToneEncoder(64)
        self.condition = _ToneCondition(condition_size)
        self.inverse_stages = nn.ModuleList((
            _LinearInverse(
                condition_size, 129, 1.0 if research_probe else 0.10
            ),                                                # output profile/transformer
            _NonlinearInverse(condition_size, segments),      # power amp -> tone stack
            _LinearInverse(condition_size, 257),              # tone stack -> cathode follower
            _NonlinearInverse(condition_size, segments),      # cathode follower -> V3A
            _LinearNonlinearInverse(condition_size, 65, segments),
            _LinearNonlinearInverse(condition_size, 65, segments),
            _NonlinearInverse(condition_size, segments),
            _LinearNonlinearInverse(condition_size, 65, segments),
            _LinearNonlinearInverse(condition_size, 65, segments),
        ))

    def stage_outputs(
        self, wet: torch.Tensor, *, tone_reference: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        if wet.ndim != 2 or not torch.isfinite(wet).all():
            raise ValueError("stagewise gray box expects finite Wet [batch,time]")
        reference = wet if tone_reference is None else tone_reference
        condition = self.condition(self.tone_encoder(reference))
        value = wet
        outputs = []
        for stage in self.inverse_stages:
            value = stage(value, condition)
            outputs.append(value)
        return outputs

    def forward(
        self,
        wet: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        tone_reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if state is not None:
            raise ValueError("stagewise gray box accepts no graph or external state")
        restored = self.stage_outputs(wet, tone_reference=tone_reference)[-1]
        restored = restored.clamp(-OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT)
        uncertainty = F.softplus(restored.new_tensor(-3.0)).expand_as(restored) + 1.0e-5
        return restored, uncertainty, None

    def manifest(self) -> dict:
        return {
            "schema": 15,
            "architecture": "nine-stage-deeply-supervised-gray-box-inverse",
            "mechanism": "amp",
            "scope": (
                "architecture probe only; product model must be reinitialized"
                if self.research_probe
                else "software Amp+cab preset to DI; product-only reinitialization"
            ),
            "sample_rate": RATE,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "condition_size": self.condition_size,
            "depth": self.depth,
            "channels": 0,
            "segments": self.segments,
            "internal_reverse_stages": [
                "output-transformer", "power-amp", "tone-stack", "cathode-follower",
                "v3a", "v2b", "v2a", "v1b", "v1a-and-input",
            ],
            "deep_supervision": True,
            "research_probe": self.research_probe,
            "research_pretraining_weights_reusable": False,
            "profile_id_input": False,
            "gain_category_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
            "recurrent_state_input": False,
            "clean_or_oracle_input": False,
            "wet_only_inference": True,
            "causal": False,
            "output_peak_contract": OUTPUT_PEAK_CONTRACT,
        }


class OpenRiffBoxStagewiseTransientGrayBoxInverse(OpenRiffBoxStagewiseGrayBoxInverse):
    """Same stage factorization trained with the frozen attack-aligned objective."""

    def manifest(self) -> dict:
        manifest = super().manifest()
        manifest.update({
            "schema": 16,
            "architecture": "nine-stage-transient-emphasis-gray-box-inverse",
            "product_loss_policy": "dynamic plus transient and positive-attack emphasis",
        })
        return manifest
