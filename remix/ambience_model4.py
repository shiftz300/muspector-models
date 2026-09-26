"""Order-independent gray-box dereverberation with bounded learned gating."""

from __future__ import annotations

import torch
from torch import nn

from .ambience2 import CONTROL_WIDTH
from .ambience4 import (
    DECAY_BANK_VARIANTS,
    PROFILE_SHAPING_STRENGTHS,
    multiband_decay_suppression,
)
from .ambience_model2 import _LongBlock, ambience_loss


class _FrequencyTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.evidence = nn.Conv2d(
            channels,
            channels * 2,
            (5, 3),
            padding=(2, dilation),
            dilation=(1, dilation),
        )
        self.mix = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        content, gate = self.evidence(self.norm(value)).chunk(2, dim=1)
        update = self.mix(torch.nn.functional.gelu(content) * torch.sigmoid(gate))
        return value + 0.2 * update


class AmbienceGrayboxExpert(nn.Module):
    """Learn where to apply a deterministic late-suppression candidate."""

    def __init__(
        self,
        channels: int = 8,
        depth: int = 8,
        n_fft: int = 1024,
        hop: int = 256,
    ) -> None:
        super().__init__()
        if channels < 4 or not 6 <= depth <= 9 or n_fft < 512 or hop < 64:
            raise ValueError("invalid ambience4 gray-box geometry")
        self.channels = channels
        self.depth = depth
        self.n_fft = n_fft
        self.hop = hop
        self.dilations = tuple(2**index for index in range(depth))
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.stem = nn.Conv2d(6 + CONTROL_WIDTH, channels, (5, 3), padding=(2, 1))
        self.blocks = nn.ModuleList(_LongBlock(channels, dilation) for dilation in self.dilations)
        self.mask = nn.Conv2d(channels, 1, 1)
        self.uncertainty = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.mask.weight)
        nn.init.zeros_(self.mask.bias)
        nn.init.zeros_(self.uncertainty.weight)
        nn.init.constant_(self.uncertainty.bias, -4.0)

    def _forward_components(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        late_base: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if wet.ndim != 2 or controls.shape != (wet.shape[0], CONTROL_WIDTH):
            raise ValueError("ambience4 expects wet [batch,time] and controls [batch,3]")
        if late_base.shape != wet.shape:
            raise ValueError("ambience4 physical candidate must match Wet geometry")
        if not all(torch.isfinite(value).all() for value in (wet, controls, late_base)):
            raise ValueError("ambience4 inputs must be finite")
        window = self.window.to(wet)
        spectrum = torch.stft(
            wet, self.n_fft, self.hop, window=window, center=True,
            pad_mode="constant", return_complex=True,
        )
        candidate = torch.stft(
            late_base, self.n_fft, self.hop, window=window, center=True,
            pad_mode="constant", return_complex=True,
        )
        scale = spectrum.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1.0e-6)

        def features(value: torch.Tensor) -> torch.Tensor:
            return torch.stack((
                value.real / scale,
                value.imag / scale,
                torch.log1p(value.abs() / scale),
            ), dim=1)

        evidence = torch.cat((features(spectrum), features(candidate)), dim=1)
        condition = controls.mul(2.0).sub(1.0)[:, :, None, None].expand(
            -1, -1, evidence.shape[2], evidence.shape[3]
        )
        hidden = torch.tanh(self.stem(torch.cat((evidence, condition), dim=1)))
        for block in self.blocks:
            hidden = block(hidden)
        learned_gate = torch.sigmoid(self.mask(hidden)).squeeze(1)
        envelope = torch.nn.functional.avg_pool1d(
            wet[:, None].square(), 480, 120, padding=240
        ).clamp_min(1.0e-12).sqrt()
        recent_peak = torch.nn.functional.max_pool1d(
            torch.nn.functional.pad(envelope, (80, 0)), 81, 1
        )
        relative_level = envelope / recent_peak.clamp_min(1.0e-6)
        temporal_prior = torch.sigmoid((0.60 - relative_level) / 0.08)
        temporal_prior = temporal_prior * (
            recent_peak >= recent_peak.amax(dim=2, keepdim=True) * 0.03
        )
        temporal_prior = torch.nn.functional.interpolate(
            temporal_prior,
            size=spectrum.shape[-1],
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        frequencies = torch.linspace(
            0.0, 24_000.0, spectrum.shape[-2], device=wet.device, dtype=wet.dtype
        )
        spectral_prior = torch.sigmoid((2_000.0 - frequencies) / 400.0)[None, :, None]
        gate_upper = temporal_prior[:, None, :] * spectral_prior
        gate = learned_gate * gate_upper
        estimate = spectrum + gate * (candidate - spectrum)
        restored = torch.istft(
            estimate, self.n_fft, self.hop, window=window, center=True, length=wet.shape[1]
        )
        frame_uncertainty = torch.nn.functional.softplus(
            self.uncertainty(hidden.mean(dim=2)).squeeze(1)
        ) + 1.0e-5
        uncertainty = torch.nn.functional.interpolate(
            frame_uncertainty.unsqueeze(1),
            size=wet.shape[1],
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        return restored, uncertainty, gate, gate_upper, spectrum, candidate

    def forward(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        late_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        restored, uncertainty, *_ = self._forward_components(wet, controls, late_base)
        return restored, uncertainty

    def training_loss(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        late_base: torch.Tensor,
        clean: torch.Tensor,
        target_start: int,
        uncertainty_weight: float,
        tail_weight: float,
        gate_oracle_weight: float,
    ) -> tuple[torch.Tensor, dict]:
        if not 0.0 <= gate_oracle_weight <= 10.0:
            raise ValueError("ambience4 oracle weight must be in [0, 10]")
        restored, uncertainty, gate, gate_upper, wet_spectrum, candidate = (
            self._forward_components(wet, controls, late_base)
        )
        loss, parts = ambience_loss(
            restored,
            uncertainty,
            wet,
            clean,
            target_start,
            uncertainty_weight,
            tail_weight,
        )
        clean_spectrum = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            window=self.window.to(clean),
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        direction = candidate - wet_spectrum
        target = clean_spectrum - wet_spectrum
        projection = (
            target.real * direction.real + target.imag * direction.imag
        ) / direction.abs().square().add(1.0e-8)
        oracle = torch.minimum(projection.clamp_min(0.0), gate_upper).detach()
        weights = direction.abs().square().detach()
        weights = weights / weights.mean(dim=(1, 2), keepdim=True).clamp_min(1.0e-8)
        active = gate_upper >= 0.05
        predicted_fraction = gate / gate_upper.clamp_min(1.0e-6)
        oracle_fraction = (oracle / gate_upper.clamp_min(1.0e-6)).detach()
        point_loss = torch.nn.functional.smooth_l1_loss(
            predicted_fraction, oracle_fraction, reduction="none"
        )
        positive = active & (oracle_fraction >= 0.05)
        negative = active & ~positive

        def weighted_mean(mask: torch.Tensor) -> torch.Tensor:
            selected = weights * mask.to(weights.dtype)
            return (point_loss * selected).sum() / selected.sum().clamp_min(1.0)

        # Balance sparse useful bins against the many bins where abstention is
        # correct. Otherwise the trivial all-zero gate minimizes the oracle term.
        oracle_loss = weighted_mean(positive) + 0.25 * weighted_mean(negative)
        loss = loss + gate_oracle_weight * oracle_loss
        parts["gate_oracle"] = float(oracle_loss.detach())
        parts["gate_mean"] = float(gate.detach().mean())
        parts["gate_oracle_mean"] = float(oracle.mean())
        parts["gate_oracle_positive_fraction"] = float(positive.float().mean())
        return loss, parts

    def manifest(self) -> dict:
        spectral_frames = 1 + 2 * sum(self.dilations)
        return {
            "schema": 1,
            "architecture": "bounded-physical-late-suppression-mask-tcn",
            "mechanism": "ambience",
            "channels": self.channels,
            "depth": self.depth,
            "n_fft": self.n_fft,
            "hop": self.hop,
            "control_width": CONTROL_WIDTH,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "spectral_context_frames": spectral_frames,
            "waveform_context_frames": spectral_frames * self.hop + self.n_fft,
            "normalization_scope": "bounded input window",
            "physical_candidate": "blind-delayed-linear-late-suppression",
            "physical_candidate_strength": 0.375,
            "wet_tail_ratio_prior": 0.60,
            "wet_tail_recent_window_ms": 200.0,
            "spectral_prior_cutoff_hz": 2_000.0,
            "learned_output": "time-frequency interpolation gate in [0,1]",
            "training_only_clean_oracle": True,
            "wet_identity_reachable": True,
            "candidate_extrapolation": False,
            "causal": False,
            "uncertainty_output": True,
            "chain_order_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        }


class AmbienceSparseMaskExpert(AmbienceGrayboxExpert):
    """Higher-capacity gate with explicit cross-frequency harmonic context."""

    def __init__(
        self,
        channels: int = 16,
        depth: int = 8,
        n_fft: int = 1024,
        hop: int = 256,
    ) -> None:
        super().__init__(channels, depth, n_fft, hop)
        self.stem = nn.Conv2d(6 + CONTROL_WIDTH, channels, (7, 5), padding=(3, 2))
        self.blocks = nn.ModuleList(
            _FrequencyTemporalBlock(channels, dilation) for dilation in self.dilations
        )
        self.mask = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        nn.init.zeros_(self.mask[-1].weight)
        nn.init.zeros_(self.mask[-1].bias)

    def manifest(self) -> dict:
        result = super().manifest()
        result.update({
            "architecture": "bounded-physical-sparse-frequency-temporal-mask-network",
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "cross_frequency_context": True,
            "gate_estimator": "residual-gated-frequency-temporal-convolution",
        })
        return result


class AmbienceDecayBankExpert(nn.Module):
    """Select a convex Wet/decay-candidate mixture independently per TF bin."""

    candidate_variants = DECAY_BANK_VARIANTS

    def __init__(
        self,
        channels: int = 10,
        depth: int = 8,
        n_fft: int = 1024,
        hop: int = 256,
    ) -> None:
        super().__init__()
        if channels < 4 or not 6 <= depth <= 9 or n_fft < 512 or hop < 64:
            raise ValueError("invalid ambience4 decay-bank geometry")
        self.channels = channels
        self.depth = depth
        self.n_fft = n_fft
        self.hop = hop
        self.candidate_count = len(self.candidate_variants)
        self.dilations = tuple(2**index for index in range(depth))
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        input_channels = 3 * (1 + self.candidate_count) + CONTROL_WIDTH
        self.stem = nn.Conv2d(input_channels, channels, (7, 5), padding=(3, 2))
        self.blocks = nn.ModuleList(
            _FrequencyTemporalBlock(channels, dilation) for dilation in self.dilations
        )
        self.logits = nn.Conv2d(channels, 1 + self.candidate_count, 1)
        self.uncertainty = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)
        with torch.no_grad():
            self.logits.bias[0] = 3.0
        nn.init.zeros_(self.uncertainty.weight)
        nn.init.constant_(self.uncertainty.bias, -4.0)

    def _runtime_candidates(self, wet: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.stack([
                multiband_decay_suppression(
                    wet,
                    half_life_ms=half_life_ms,
                    ratio_threshold=ratio_threshold,
                    strength=strength,
                    n_fft=self.n_fft,
                    hop=self.hop,
                )
                for half_life_ms, ratio_threshold, strength in self.candidate_variants
            ], dim=1)

    def _forward_components(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        candidate_bank: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if wet.ndim != 2 or controls.shape != (wet.shape[0], CONTROL_WIDTH):
            raise ValueError("decay-bank expert expects Wet [batch,time] and controls [batch,3]")
        if candidate_bank.ndim == 2:
            if candidate_bank.shape != wet.shape:
                raise ValueError("decay-bank runtime placeholder must match Wet geometry")
            candidate_bank = self._runtime_candidates(wet)
        expected = (wet.shape[0], self.candidate_count, wet.shape[1])
        if candidate_bank.shape != expected:
            raise ValueError(f"decay candidate bank must have geometry {expected}")
        if not all(torch.isfinite(value).all() for value in (wet, controls, candidate_bank)):
            raise ValueError("decay-bank inputs must be finite")
        waveforms = torch.cat((wet[:, None], candidate_bank), dim=1)
        flat = waveforms.flatten(0, 1)
        spectra = torch.stft(
            flat, self.n_fft, self.hop, window=self.window.to(wet), center=True,
            pad_mode="constant", return_complex=True,
        ).reshape(wet.shape[0], 1 + self.candidate_count, self.n_fft // 2 + 1, -1)
        scale = spectra[:, 0].abs().mean(dim=(1, 2), keepdim=True).clamp_min(1.0e-6)
        normalized = spectra / scale[:, None]
        evidence = torch.stack((
            normalized.real,
            normalized.imag,
            torch.log1p(normalized.abs()),
        ), dim=2).flatten(1, 2)
        condition = controls.mul(2.0).sub(1.0)[:, :, None, None].expand(
            -1, -1, evidence.shape[2], evidence.shape[3]
        )
        hidden = torch.tanh(self.stem(torch.cat((evidence, condition), dim=1)))
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.logits(hidden)
        weights = torch.softmax(logits, dim=1)
        estimate = torch.sum(weights * spectra, dim=1)
        restored = torch.istft(
            estimate, self.n_fft, self.hop, window=self.window.to(wet), center=True,
            length=wet.shape[1],
        )
        frame_uncertainty = torch.nn.functional.softplus(
            self.uncertainty(hidden.mean(dim=2)).squeeze(1)
        ) + 1.0e-5
        uncertainty = torch.nn.functional.interpolate(
            frame_uncertainty[:, None], size=wet.shape[1],
            mode="linear", align_corners=False,
        )[:, 0]
        return restored, uncertainty, logits, spectra

    def forward(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        candidate_bank: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        restored, uncertainty, *_ = self._forward_components(wet, controls, candidate_bank)
        return restored, uncertainty

    def training_loss(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        candidate_bank: torch.Tensor,
        clean: torch.Tensor,
        target_start: int,
        uncertainty_weight: float,
        tail_weight: float,
        gate_oracle_weight: float,
    ) -> tuple[torch.Tensor, dict]:
        restored, uncertainty, logits, spectra = self._forward_components(
            wet, controls, candidate_bank
        )
        loss, parts = ambience_loss(
            restored, uncertainty, wet, clean, target_start,
            uncertainty_weight, tail_weight,
        )
        clean_spectrum = torch.stft(
            clean, self.n_fft, self.hop, window=self.window.to(clean), center=True,
            pad_mode="constant", return_complex=True,
        )
        wet_spectrum = spectra[:, 0]
        directions = spectra[:, 1:] - wet_spectrum[:, None]
        target = clean_spectrum - wet_spectrum
        projections = (
            target[:, None].real * directions.real
            + target[:, None].imag * directions.imag
        ) / directions.abs().square().add(1.0e-8)
        projections = projections.clamp(0.0, 1.0)
        projected = wet_spectrum[:, None] + projections * directions
        projected_error = (projected - clean_spectrum[:, None]).abs().square()
        candidate_index = projected_error.argmin(dim=1)
        candidate_fraction = torch.gather(
            projections, 1, candidate_index[:, None]
        )[:, 0]
        best_error = torch.gather(
            projected_error, 1, candidate_index[:, None]
        )[:, 0]
        useful = best_error < (wet_spectrum - clean_spectrum).abs().square()
        candidate_fraction = (candidate_fraction * useful).detach()
        oracle = torch.zeros_like(logits).detach()
        oracle[:, 0] = 1.0 - candidate_fraction
        oracle.scatter_(
            1, candidate_index[:, None] + 1, candidate_fraction[:, None]
        )
        point_loss = -(oracle * torch.log_softmax(logits, dim=1)).sum(dim=1)
        positive = candidate_fraction >= 0.05

        def selected_mean(mask: torch.Tensor) -> torch.Tensor:
            return (point_loss * mask).sum() / mask.sum().clamp_min(1)

        oracle_loss = selected_mean(positive) + 0.25 * selected_mean(~positive)
        loss = loss + gate_oracle_weight * oracle_loss
        parts["candidate_oracle"] = float(oracle_loss.detach())
        parts["candidate_oracle_positive_fraction"] = float(positive.float().mean())
        parts["candidate_oracle_mean_fraction"] = float(candidate_fraction.mean())
        parts["wet_selection_mean"] = float(torch.softmax(logits, dim=1)[:, 0].mean().detach())
        return loss, parts

    def manifest(self) -> dict:
        spectral_frames = 1 + 2 * sum(self.dilations)
        return {
            "schema": 1,
            "architecture": "bounded-multiband-exponential-decay-candidate-selector",
            "mechanism": "ambience",
            "channels": self.channels,
            "depth": self.depth,
            "n_fft": self.n_fft,
            "hop": self.hop,
            "control_width": CONTROL_WIDTH,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "candidate_count": self.candidate_count,
            "candidate_variants": [
                {
                    "half_life_ms": half_life_ms,
                    "ratio_threshold": ratio_threshold,
                    "strength": strength,
                }
                for half_life_ms, ratio_threshold, strength in self.candidate_variants
            ],
            "spectral_context_frames": spectral_frames,
            "waveform_context_frames": spectral_frames * self.hop + self.n_fft,
            "physical_candidate": "wet-only-causal-multiband-exponential-decay-state",
            "learned_output": "convex time-frequency selector including Wet identity",
            "training_only_clean_oracle": True,
            "wet_identity_reachable": True,
            "candidate_extrapolation": False,
            "causal": False,
            "uncertainty_output": True,
            "chain_order_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        }


class AmbienceProfileBankExpert(AmbienceDecayBankExpert):
    """Select only among Wet and known-profile response-shortening candidates."""

    candidate_variants = PROFILE_SHAPING_STRENGTHS
    requires_profile_candidate_bank = True

    def _runtime_candidates(self, wet: torch.Tensor) -> torch.Tensor:
        raise ValueError("profile-bank runtime requires candidates from the current RIR profile")

    def _forward_components(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        candidate_bank: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        restored, uncertainty, logits, spectra = super()._forward_components(
            wet, controls, candidate_bank
        )
        exact = (
            (candidate_bank[:, 1:] - candidate_bank[:, :1])
            .abs()
            .amax(dim=(1, 2))
            <= 1.0e-7
        )
        # Stable exact inversion is immutable. Learning is used only after its
        # analytic stability veto, so a checkpoint cannot degrade exact cases.
        restored = torch.where(exact[:, None], candidate_bank[:, 0], restored)
        return restored, uncertainty, logits, spectra

    def manifest(self) -> dict:
        spectral_frames = 1 + 2 * sum(self.dilations)
        return {
            "schema": 2,
            "architecture": "known-profile-response-shortening-bank-selector",
            "mechanism": "ambience",
            "channels": self.channels,
            "depth": self.depth,
            "n_fft": self.n_fft,
            "hop": self.hop,
            "control_width": CONTROL_WIDTH,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "candidate_count": self.candidate_count,
            "candidate_variants": [
                {
                    "early_ms": 25.0,
                    "target_rt60_ms": 40.0,
                    "maximum_gain": 4.0,
                    "low_band_strength": strength,
                    "high_band_strength": 0.0,
                    "transition_band_hz": [2_000.0, 3_000.0],
                }
                for strength in self.candidate_variants
            ],
            "physical_candidate": "known-current-profile-bounded-response-shortening",
            "profile_required": True,
            "exact_profile_path_is_immutable": True,
            "runtime_candidate_input": "current RIR plus current mix and room gain only",
            "learned_output": "convex time-frequency selector including Wet identity",
            "spectral_context_frames": spectral_frames,
            "waveform_context_frames": spectral_frames * self.hop + self.n_fft,
            "training_only_clean_oracle": True,
            "wet_identity_reachable": True,
            "candidate_extrapolation": False,
            "causal": False,
            "uncertainty_output": True,
            "chain_order_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        }


class AmbienceFrequencyProfileBankExpert(AmbienceProfileBankExpert):
    """Select Wet or a frequency-dependent known-profile shortening candidate."""

    from .ambience4_frequency_shaping import FREQUENCY_SHAPING_CANDIDATES

    candidate_variants = FREQUENCY_SHAPING_CANDIDATES

    def manifest(self) -> dict:
        spectral_frames = 1 + 2 * sum(self.dilations)
        return {
            "schema": 3,
            "architecture": "known-profile-frequency-decay-bank-selector",
            "mechanism": "ambience",
            "channels": self.channels,
            "depth": self.depth,
            "n_fft": self.n_fft,
            "hop": self.hop,
            "control_width": CONTROL_WIDTH,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "candidate_count": self.candidate_count,
            "candidate_variants": list(self.candidate_variants),
            "physical_candidate": "known-current-profile-frequency-bin-specific-faded-response",
            "frequency_decay_estimator": "per-bin Schroeder T30 extrapolated to T60",
            "profile_required": True,
            "exact_profile_path_is_immutable": True,
            "runtime_candidate_input": "current RIR plus current mix and room gain only",
            "learned_output": "convex time-frequency selector including Wet identity",
            "spectral_context_frames": spectral_frames,
            "waveform_context_frames": spectral_frames * self.hop + self.n_fft,
            "training_only_clean_oracle": True,
            "wet_identity_reachable": True,
            "candidate_extrapolation": False,
            "causal": False,
            "uncertainty_output": True,
            "chain_order_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        }
