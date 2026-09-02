"""Learned residual correction for filter-robust Delay Feedback and Mix."""

from __future__ import annotations

import torch

from .model import Estimate
from .physics import (
    DeconvolutionContext,
    deconvolution_delay_controls,
    deconvolution_delay_hint,
)
from .reverb_model import FEATURES as IMPULSE_FEATURES
from .reverb_model import reverb_features


FEATURES = IMPULSE_FEATURES + 3


@torch.no_grad()
def delay_features(
    dry: torch.Tensor,
    wet: torch.Tensor,
    impulse_features: torch.Tensor | None = None,
    context: DeconvolutionContext | None = None,
) -> torch.Tensor:
    time = deconvolution_delay_hint(dry, wet, context)
    feedback, mix = deconvolution_delay_controls(dry, wet, context)
    analytic = torch.stack((time, feedback, mix), dim=1)
    impulse = (
        reverb_features(dry, wet, context)
        if impulse_features is None
        else impulse_features
    )
    return torch.cat((impulse, analytic), dim=1)


class DelayControlEstimator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(FEATURES),
            torch.nn.Linear(FEATURES, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def apply_delay_residual(
    estimate: Estimate,
    dry: torch.Tensor,
    wet: torch.Tensor,
    model: DelayControlEstimator,
    impulse_features: torch.Tensor | None = None,
    context: DeconvolutionContext | None = None,
) -> Estimate:
    controls = estimate.control_logits.clone()
    controls[:, 4:6] = model(
        delay_features(dry, wet, impulse_features, context)
    )
    return Estimate(estimate.order_logits, controls)
