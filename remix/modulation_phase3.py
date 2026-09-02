"""High-confidence hidden-phase Tremolo inverse with explicit abstention."""

from __future__ import annotations

import math

import torch
from torch import nn

from .foundation_data import RATE
from .modulation3 import CONTROL_WIDTH, DEPTH_MAX, DEPTH_MIN, RATE_MAX, RATE_MIN


FRAME = 480
HOP = 120
PHASE_CANDIDATES = 128
MINIMUM_RATE_HZ = 3.2
DEFAULT_RESIDUAL_THRESHOLD = 0.16


def _waveform(cycles: torch.Tensor, triangle: torch.Tensor) -> torch.Tensor:
    sine = 0.5 + 0.5 * torch.sin(2.0 * torch.pi * cycles)
    fraction = torch.remainder(cycles, 1.0)
    triangular = 1.0 - 2.0 * torch.abs(fraction - 0.5)
    return torch.where(triangle, triangular, sine)


class TremoloPhaseInverseV3(nn.Module):
    """Estimate LFO phase from Wet harmonics; pass through ambiguous inputs."""

    mechanism = "modulation"
    family = "tremolo"

    def __init__(self, residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD) -> None:
        super().__init__()
        if not 0.0 < residual_threshold < 1.0:
            raise ValueError("invalid Tremolo residual threshold")
        self.residual_threshold = float(residual_threshold)

    def forward(
        self, wet: torch.Tensor, controls: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if wet.ndim != 2 or controls.shape != (wet.shape[0], CONTROL_WIDTH):
            raise ValueError("Tremolo phase inverse expects Wet [batch,time] and controls [batch,5]")
        if wet.shape[1] < FRAME or not torch.isfinite(wet).all() or not torch.isfinite(controls).all():
            raise ValueError("invalid Tremolo phase inverse input")
        rate = RATE_MIN * torch.exp(controls[:, 0] * math.log(RATE_MAX / RATE_MIN))
        depth = DEPTH_MIN + controls[:, 1] * (DEPTH_MAX - DEPTH_MIN)
        triangle = controls[:, 2] >= 0.5
        frames = wet.unfold(1, FRAME, HOP)
        log_envelope = torch.log(frames.square().mean(dim=-1).add(1.0e-10).sqrt())
        count = log_envelope.shape[1]
        normalized_time = torch.linspace(-1.0, 1.0, count, dtype=wet.dtype, device=wet.device)
        design = torch.stack(
            (torch.ones_like(normalized_time), normalized_time, normalized_time.square(), normalized_time.pow(3)),
            dim=1,
        )
        basis_q = torch.linalg.qr(design, mode="reduced").Q
        detrended = log_envelope - (log_envelope @ basis_q) @ basis_q.transpose(0, 1)
        frame_time = (torch.arange(count, dtype=wet.dtype, device=wet.device) * HOP + FRAME / 2.0) / RATE
        phases = torch.arange(PHASE_CANDIDATES, dtype=wet.dtype, device=wet.device)
        phases = phases * (2.0 * torch.pi / PHASE_CANDIDATES)
        cycles = rate[:, None, None] * frame_time[None, None, :] + phases[None, :, None] / (2.0 * torch.pi)
        lfo = _waveform(cycles, triangle[:, None, None])
        candidate_inverse = -torch.log((1.0 - depth[:, None, None] * (1.0 - lfo)).clamp_min(1.0e-4))
        corrected = detrended[:, None, :] + candidate_inverse
        scores = corrected.new_zeros(corrected.shape[:2])
        base_score = detrended.new_zeros(detrended.shape[0])
        for harmonic in (1.0, 2.0, 3.0, 4.0):
            angle = 2.0 * torch.pi * harmonic * rate[:, None] * frame_time[None, :]
            cosine, sine = torch.cos(angle), torch.sin(angle)
            scores = scores + (corrected * cosine[:, None, :]).sum(dim=2).square()
            scores = scores + (corrected * sine[:, None, :]).sum(dim=2).square()
            base_score = base_score + (detrended * cosine).sum(dim=1).square()
            base_score = base_score + (detrended * sine).sum(dim=1).square()
        selected_score, selected_index = scores.min(dim=1)
        residual_ratio = selected_score / base_score.clamp_min(1.0e-12)
        selected_phase = phases[selected_index]
        full_time = torch.arange(wet.shape[1], dtype=wet.dtype, device=wet.device) / RATE
        full_cycles = rate[:, None] * full_time[None, :] + selected_phase[:, None] / (2.0 * torch.pi)
        full_lfo = _waveform(full_cycles, triangle[:, None])
        inverse_log_gain = -torch.log((1.0 - depth[:, None] * (1.0 - full_lfo)).clamp_min(1.0e-4))
        accepted = (rate >= MINIMUM_RATE_HZ) & (residual_ratio <= self.residual_threshold)
        admitted_gain = torch.where(accepted[:, None], inverse_log_gain, torch.zeros_like(inverse_log_gain))
        restored = wet * torch.exp(admitted_gain)
        return restored, residual_ratio, accepted, admitted_gain

    def manifest(self) -> dict:
        return {
            "schema": 1,
            "architecture": "wet-harmonic-hidden-lfo-phase-search-with-abstention",
            "mechanism": self.mechanism,
            "family": self.family,
            "parameters": 0,
            "control_width": CONTROL_WIDTH,
            "hidden_forward_state": "lfo_start_phase",
            "hidden_forward_state_is_inference_input": False,
            "phase_candidates": PHASE_CANDIDATES,
            "harmonics": [1, 2, 3, 4],
            "minimum_admitted_rate_hz": MINIMUM_RATE_HZ,
            "residual_threshold": self.residual_threshold,
            "ambiguous_input_behavior": "abstain-and-pass-through",
            "causal": False,
            "bounded_context": True,
            "graph_order_input": False,
            "neighbouring_effect_input": False,
            "clean_or_oracle_input": False,
            "physical_device_claim": False,
        }
