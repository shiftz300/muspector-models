"""Wet-conditioned Wiener-Hammerstein gray-box Amp+cab inverse."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .egdb_pg_amp_model import OUTPUT_PEAK_CONTRACT, RATE
from .egdb_pg_tone_encoder import WetToneEncoder


class _ToneCondition(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size
        self.network = nn.Sequential(
            nn.Linear(64, size),
            nn.GELU(),
            nn.Linear(size, size),
            nn.Tanh(),
        )

    def forward(self, tone: torch.Tensor) -> torch.Tensor:
        return self.network(tone)


class _ToneFIR(nn.Module):
    """Symmetric, tone-conditioned LTI inverse initialized to a scaled identity."""

    def __init__(
        self,
        condition_size: int,
        taps: int,
        initial_scale: float,
        *,
        symmetric: bool = True,
    ) -> None:
        super().__init__()
        if taps < 3 or taps % 2 != 1:
            raise ValueError("gray-box FIR taps must be odd and at least three")
        self.taps = taps
        self.symmetric = symmetric
        coefficients = taps // 2 + 1 if symmetric else taps
        self.base_half = nn.Parameter(torch.zeros(coefficients))
        self.delta_half = nn.Linear(condition_size, coefficients, bias=False)
        with torch.no_grad():
            self.base_half[-1 if symmetric else taps // 2] = initial_scale
            self.delta_half.weight.zero_()

    def _kernel(self, condition: torch.Tensor) -> torch.Tensor:
        half = self.base_half.unsqueeze(0) + (
            0.25 / self.taps**0.5
        ) * torch.tanh(self.delta_half(condition))
        if self.symmetric:
            return torch.cat((half[:, :-1], half.flip(1)), dim=1)
        return half

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        kernels = self._kernel(condition)
        grouped = value.unsqueeze(0)
        restored = F.conv1d(
            grouped,
            kernels.flip(-1).unsqueeze(1),
            padding=self.taps // 2,
            groups=value.shape[0],
        )
        return restored.squeeze(0)


class _ToneMonotonicInverse(nn.Module):
    """Asymmetric monotonic waveshaper with explicit fast/slow envelope memory."""

    def __init__(self, condition_size: int, segments: int) -> None:
        super().__init__()
        if segments < 8:
            raise ValueError("gray-box waveshaper requires at least eight segments")
        self.segments = segments
        self.log_slopes = nn.Linear(condition_size, 2 * segments, bias=True)
        self.static_gain = nn.Linear(condition_size, 1, bias=True)
        self.dynamic_strength = nn.Linear(condition_size, 1, bias=True)
        for layer in (self.log_slopes, self.static_gain, self.dynamic_strength):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        slopes = torch.exp(
            self.log_slopes(condition).reshape(-1, 2, self.segments).clamp(-2.5, 2.5)
        )
        side = (value < 0.0).long()
        magnitude = value.abs()
        scaled = magnitude * self.segments
        bucket = torch.floor(scaled).long().clamp(0, self.segments - 1)
        selected = slopes.gather(1, side.unsqueeze(-1).expand(-1, -1, self.segments))
        local_slope = selected.gather(2, bucket.unsqueeze(-1)).squeeze(-1)
        starts = torch.cumsum(selected / self.segments, dim=-1) - selected / self.segments
        local_start = starts.gather(2, bucket.unsqueeze(-1)).squeeze(-1)
        local = magnitude - bucket.to(value.dtype) / self.segments
        shaped = torch.sign(value) * (local_start + local_slope * local)

        absolute = shaped.abs().unsqueeze(1)
        fast = F.avg_pool1d(absolute, 129, 1, 64).squeeze(1)
        slow = F.avg_pool1d(absolute, 2049, 1, 1024).squeeze(1)
        envelope_delta = (
            torch.log(fast + 1.0e-5) - torch.log(slow + 1.0e-5)
        ).clamp(-2.0, 2.0)
        static = self.static_gain(condition)
        dynamic = torch.tanh(self.dynamic_strength(condition))
        gain = torch.exp((static + dynamic * envelope_delta).clamp(-3.5, 3.5))
        return shaped * gain


class WetToneGrayBoxAmpCabInverse(nn.Module):
    """Estimate internal inverse-stage parameters from Wet audio only."""

    mechanism = "amp"

    def __init__(
        self, channels: int = 24, depth: int = 8, condition_size: int = 64
    ) -> None:
        super().__init__()
        del channels
        if depth < 4 or condition_size < 16:
            raise ValueError("invalid gray-box Amp+cab geometry")
        self.depth = depth
        self.condition_size = condition_size
        self.tone_encoder = WetToneEncoder(64)
        for parameter in self.tone_encoder.parameters():
            parameter.requires_grad_(False)
        self.tone_encoder_sha256 = None
        self.condition = _ToneCondition(condition_size)
        segments = max(8, depth * 2)
        self.output_profile_inverse = _ToneFIR(condition_size, 129, 0.10)
        self.power_inverse = _ToneMonotonicInverse(condition_size, segments)
        self.tone_stack_inverse = _ToneFIR(condition_size, 257, 1.0)
        self.preamp_inverse = _ToneMonotonicInverse(condition_size, segments)
        self.input_profile_inverse = _ToneFIR(condition_size, 129, 1.0)
        self.uncertainty_logit = nn.Parameter(torch.tensor(-3.0))

    def load_tone_encoder(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        architecture = payload.get("architecture", {})
        if architecture.get("architecture") != "wet-log-spectrum-content-invariant-tone-encoder":
            raise ValueError("checkpoint is not an admitted Wet tone encoder")
        self.tone_encoder.load_state_dict(payload["state_dict"])
        self.tone_encoder_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def forward(
        self,
        wet: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        tone_reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if state is not None:
            raise ValueError("gray-box Amp+cab inverse accepts no graph or external state")
        if wet.ndim != 2 or not torch.isfinite(wet).all():
            raise ValueError("gray-box Amp+cab inverse expects finite Wet [batch,time]")
        reference = wet if tone_reference is None else tone_reference
        if reference.ndim != 2 or reference.shape[0] != wet.shape[0]:
            raise ValueError("gray-box Amp+cab inverse expects batch-aligned Wet reference")
        with torch.no_grad():
            tone = self.tone_encoder(reference)
        condition = self.condition(tone)
        restored = self.forward_with_condition(wet, condition)
        uncertainty = F.softplus(self.uncertainty_logit).expand_as(restored) + 1.0e-5
        return restored, uncertainty, None

    def forward_with_condition(
        self, wet: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """Run explicit inverse stages from a supplied diagnostic condition.

        Production inference must use :meth:`forward`. This entry point exists
        solely for clean-assisted, split-isolated processor-capacity audits.
        """
        if wet.ndim != 2 or condition.shape != (wet.shape[0], self.condition_size):
            raise ValueError("gray-box diagnostic condition shape mismatch")
        if not torch.isfinite(wet).all() or not torch.isfinite(condition).all():
            raise ValueError("gray-box diagnostic input must be finite")
        value = self.output_profile_inverse(wet, condition)
        value = self.power_inverse(value, condition)
        value = self.tone_stack_inverse(value, condition)
        value = self.preamp_inverse(value, condition)
        value = self.input_profile_inverse(value, condition)
        return value.clamp(-OUTPUT_PEAK_CONTRACT, OUTPUT_PEAK_CONTRACT)

    def manifest(self) -> dict:
        return {
            "schema": 11,
            "architecture": "wet-parameter-estimated-wiener-hammerstein-gray-box-inverse",
            "mechanism": "amp",
            "scope": "software Amp+cab preset to DI",
            "sample_rate": RATE,
            "channels": 0,
            "depth": self.depth,
            "condition_size": self.condition_size,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "internal_reverse_stages": [
                "output-profile-LTI-inverse",
                "monotonic-dynamic-power-inverse",
                "tone-stack-LTI-inverse",
                "monotonic-dynamic-preamp-inverse",
                "input-profile-LTI-inverse",
            ],
            "symmetric_fir_taps": [129, 257, 129],
            "monotonic_segments_per_polarity": max(8, self.depth * 2),
            "explicit_envelope_memory": True,
            "opaque_temporal_network": False,
            "tone_encoder_frozen": True,
            "tone_encoder_sha256": self.tone_encoder_sha256,
            "tone_reference_source": "another 3-second span of the same Wet recording",
            "parameter_estimator_input": "Wet tone embedding only",
            "profile_id_input": False,
            "gain_category_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
            "recurrent_state_input": False,
            "clean_or_oracle_input": False,
            "causal": False,
            "offline_chunk_conditioning": True,
            "output_peak_contract": OUTPUT_PEAK_CONTRACT,
        }


class WetTonePhaseGrayBoxAmpCabInverse(WetToneGrayBoxAmpCabInverse):
    """Gray-box inverse whose LTI stages can reconstruct non-zero phase."""

    def __init__(
        self, channels: int = 24, depth: int = 8, condition_size: int = 64
    ) -> None:
        super().__init__(channels, depth, condition_size)
        self.output_profile_inverse = _ToneFIR(
            condition_size, 513, 0.10, symmetric=False
        )
        self.tone_stack_inverse = _ToneFIR(
            condition_size, 257, 1.0, symmetric=False
        )
        self.input_profile_inverse = _ToneFIR(
            condition_size, 129, 1.0, symmetric=False
        )

    def manifest(self) -> dict:
        manifest = super().manifest()
        manifest.update({
            "schema": 12,
            "architecture": "wet-parameter-estimated-phase-aware-wiener-hammerstein-gray-box-inverse",
            "symmetric_fir_taps": [],
            "arbitrary_phase_fir_taps": [513, 257, 129],
            "phase_policy": "tone-conditioned non-symmetric LTI inverse stages",
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters()
                if parameter.requires_grad
            ),
        })
        return manifest
