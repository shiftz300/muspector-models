"""Compact Reverb-control head over a deconvolved multiscale energy curve."""

from __future__ import annotations

import math

import torch

from .model import Estimate
from .physics import ANALYSIS_RATE, DeconvolutionContext, deconvolved_impulse


FEATURES = 160


def _temporal_features(impulse: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Pool a deconvolved impulse into the original 160 energy features."""

    energy = impulse.double().square()
    length = energy.shape[1]
    duration = length / ANALYSIS_RATE
    logarithmic = torch.unique(
        torch.round(
            torch.logspace(
                math.log10(0.002),
                math.log10(duration),
                97,
                device=energy.device,
                dtype=energy.dtype,
            )
            * ANALYSIS_RATE
        ).to(torch.int64)
    ).clamp(0, length)
    linear = torch.linspace(
        0, length, 65, device=energy.device, dtype=energy.dtype
    ).to(torch.int64)
    pooled = []
    for edges in (logarithmic, linear):
        pooled.extend(
            energy[:, int(left) : int(right)].mean(dim=1)
            for left, right in zip(edges[:-1], edges[1:])
        )
    value = torch.stack(pooled, dim=1)
    if value.shape[1] != FEATURES:
        raise ValueError(f"expected {FEATURES} Reverb features, got {value.shape[1]}")
    reference = value.amax(dim=1, keepdim=True).clamp_min(1.0e-20)
    decibels = 10.0 * torch.log10((value / reference).clamp_min(1.0e-12))
    return decibels.clamp(-100.0, 10.0).to(dtype) / 100.0


@torch.no_grad()
def reverb_features(
    dry: torch.Tensor,
    wet: torch.Tensor,
    context: DeconvolutionContext | None = None,
) -> torch.Tensor:
    """Extract scale-invariant early and late impulse-response energies."""

    impulse = deconvolved_impulse(dry, wet, 1.0e-5, context)
    return _temporal_features(impulse, dry.dtype)


class ReverbControlEstimator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(FEATURES),
            torch.nn.Linear(FEATURES, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def apply_reverb_estimate(
    estimate: Estimate,
    dry: torch.Tensor,
    wet: torch.Tensor,
    model: ReverbControlEstimator,
    features: torch.Tensor | None = None,
) -> Estimate:
    controls = estimate.control_logits.clone()
    value = reverb_features(dry, wet) if features is None else features
    controls[:, 6:9] = model(value)
    return Estimate(estimate.order_logits, controls)
