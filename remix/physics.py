"""Paired signal-processing estimators that outperform learned controls."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .model import Estimate


SOURCE_RATE = 44_100
ANALYSIS_RATE = 11_025
DELAY_CONTROL = 3
DELAY_FEEDBACK_CONTROL = 4
DELAY_MIX_CONTROL = 5


@dataclass(frozen=True)
class DeconvolutionContext:
    dry_spectrum: torch.Tensor
    wet_spectrum: torch.Tensor
    power: torch.Tensor
    output_length: int
    padded_length: int


def deconvolution_context(dry: torch.Tensor, wet: torch.Tensor) -> DeconvolutionContext:
    """Compute paired spectra once for all regularized hybrid estimators."""

    if dry.ndim != 2 or wet.shape != dry.shape:
        raise ValueError(f"expected matching [batch,time] audio, got {dry.shape}/{wet.shape}")
    dry_small = torch.nn.functional.avg_pool1d(dry.unsqueeze(1), 4, 4).squeeze(1)
    wet_small = torch.nn.functional.avg_pool1d(wet.unsqueeze(1), 4, 4).squeeze(1)
    required = dry_small.shape[1] * 2
    padded = 1 << math.ceil(math.log2(required))
    dry_spectrum = torch.fft.rfft(dry_small, n=padded)
    wet_spectrum = torch.fft.rfft(wet_small, n=padded)
    return DeconvolutionContext(
        dry_spectrum,
        wet_spectrum,
        dry_spectrum.abs().square(),
        dry_small.shape[1],
        padded,
    )


def deconvolved_impulse(
    dry: torch.Tensor,
    wet: torch.Tensor,
    regularization: float,
    context: DeconvolutionContext | None = None,
) -> torch.Tensor:
    value = deconvolution_context(dry, wet) if context is None else context
    power = value.power
    regularizer = (power.mean(dim=1, keepdim=True) * regularization).clamp_min(
        torch.finfo(power.dtype).eps
    )
    transfer = (
        value.wet_spectrum * value.dry_spectrum.conj() / (power + regularizer)
    )
    return torch.fft.irfft(transfer, n=value.padded_length)[
        :, : value.output_length
    ]


def _delay_samples(impulse: torch.Tensor) -> torch.Tensor:
    minimum = round(0.040 * ANALYSIS_RATE)
    maximum = min(round(1.000 * ANALYSIS_RATE), impulse.shape[1] - 1)
    return impulse[:, minimum:maximum].abs().argmax(dim=1) + minimum


def deconvolution_delay_hint(
    dry: torch.Tensor,
    wet: torch.Tensor,
    context: DeconvolutionContext | None = None,
) -> torch.Tensor:
    """Estimate normalized 40-1000 ms delay time from aligned Clean/Wet pairs."""

    impulse = deconvolved_impulse(dry, wet, 1.0e-3, context)
    delay_samples = _delay_samples(impulse)
    delay_ms = delay_samples.to(dry.dtype) * (1_000.0 / ANALYSIS_RATE)
    return (torch.log(delay_ms / 40.0) / math.log(25.0)).clamp(0.0, 1.0)


def deconvolution_delay_controls(
    dry: torch.Tensor,
    wet: torch.Tensor,
    context: DeconvolutionContext | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate normalized Feedback and Mix from consecutive echo energies."""

    impulse = deconvolved_impulse(dry, wet, 1.0e-4, context)
    delay_samples = _delay_samples(impulse)
    window = round(0.004 * ANALYSIS_RATE)
    offsets = torch.arange(window, device=impulse.device).unsqueeze(0)
    first = torch.gather(impulse, 1, delay_samples.unsqueeze(1) + offsets)
    second = torch.gather(impulse, 1, 2 * delay_samples.unsqueeze(1) + offsets)
    direct_energy = torch.linalg.vector_norm(impulse[:, :window], dim=1)
    first_energy = torch.linalg.vector_norm(first, dim=1)
    second_energy = torch.linalg.vector_norm(second, dim=1)
    epsilon = torch.finfo(impulse.dtype).eps
    mix = first_energy / (direct_energy + first_energy + epsilon)
    feedback = second_energy / (first_energy + epsilon)
    return (feedback.clamp(0.0, 0.9) / 0.9, mix.clamp(0.0, 0.7) / 0.7)


def apply_delay_hints(
    estimate: Estimate,
    dry: torch.Tensor,
    wet: torch.Tensor,
    context: DeconvolutionContext | None = None,
) -> Estimate:
    controls = estimate.control_logits.clone()
    time = deconvolution_delay_hint(dry, wet, context)
    feedback, mix = deconvolution_delay_controls(dry, wet, context)
    for index, hint in (
        (DELAY_CONTROL, time),
        (DELAY_FEEDBACK_CONTROL, feedback),
        (DELAY_MIX_CONTROL, mix),
    ):
        hint = hint.clamp(1.0e-4, 1.0 - 1.0e-4)
        controls[:, index] = torch.log(hint / (1.0 - hint))
    return Estimate(estimate.order_logits, controls)
