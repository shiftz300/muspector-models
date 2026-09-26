"""Direct profile-conditioned Reverb inverse with physical replay training."""

from __future__ import annotations

import torch
from torch import nn

from .ambience2 import CONTROL_WIDTH
from .ambience5_profile_direct import PROFILE_CHANNELS, PROFILE_N_FFT
from .ambience_model2 import ambience_loss
from .ambience_model4 import _FrequencyTemporalBlock


class AmbienceProfileDirectExpert(nn.Module):
    """Predict a bounded complex correction outside the rejected candidate hull."""

    def __init__(
        self,
        channels: int = 12,
        depth: int = 8,
        n_fft: int = PROFILE_N_FFT,
        hop: int = 256,
        maximum_complex_correction: float = 2.0,
    ) -> None:
        super().__init__()
        if channels < 6 or not 6 <= depth <= 9 or n_fft != PROFILE_N_FFT or hop < 64:
            raise ValueError("invalid profile-direct geometry")
        if not 0.5 <= maximum_complex_correction <= 4.0:
            raise ValueError("invalid profile-direct correction bound")
        self.channels = channels
        self.depth = depth
        self.n_fft = n_fft
        self.hop = hop
        self.maximum_complex_correction = maximum_complex_correction
        self.dilations = tuple(2**index for index in range(depth))
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.stem = nn.Conv2d(
            6 + PROFILE_CHANNELS + CONTROL_WIDTH,
            channels,
            (7, 5),
            padding=(3, 2),
        )
        self.blocks = nn.ModuleList(
            _FrequencyTemporalBlock(channels, dilation) for dilation in self.dilations
        )
        self.correction = nn.Conv2d(channels, 2, 1)
        self.uncertainty = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        nn.init.zeros_(self.uncertainty.weight)
        nn.init.constant_(self.uncertainty.bias, -4.0)

    def _forward_components(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        profile_features: torch.Tensor,
        analytic_base: torch.Tensor,
        exact_base: torch.Tensor,
        exact_available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, frames = wet.shape
        expected_profile = (batch, PROFILE_CHANNELS, self.n_fft // 2 + 1)
        if controls.shape != (batch, CONTROL_WIDTH) or profile_features.shape != expected_profile:
            raise ValueError("profile-direct inputs have incompatible geometry")
        if analytic_base.shape != wet.shape or exact_base.shape != wet.shape or exact_available.shape != (batch,):
            raise ValueError("profile-direct analytic path geometry changed")
        if not all(torch.isfinite(value).all() for value in (wet, controls, profile_features, analytic_base, exact_base)):
            raise ValueError("profile-direct inputs must be finite")
        spectrum = torch.stft(
            wet,
            self.n_fft,
            self.hop,
            window=self.window.to(wet),
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        analytic_spectrum = torch.stft(
            analytic_base,
            self.n_fft,
            self.hop,
            window=self.window.to(wet),
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        scale = spectrum.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1.0e-6)
        def audio_features(value: torch.Tensor) -> torch.Tensor:
            return torch.stack((
                value.real / scale,
                value.imag / scale,
                torch.log1p(value.abs() / scale),
            ), dim=1)
        wet_features = audio_features(spectrum)
        analytic_features = audio_features(analytic_spectrum)
        profile = profile_features[:, :, :, None].expand(-1, -1, -1, spectrum.shape[-1])
        condition = controls.mul(2.0).sub(1.0)[:, :, None, None].expand(
            -1, -1, spectrum.shape[-2], spectrum.shape[-1]
        )
        hidden = torch.tanh(self.stem(torch.cat(
            (wet_features, analytic_features, profile, condition), dim=1
        )))
        for block in self.blocks:
            hidden = block(hidden)
        components = torch.tanh(self.correction(hidden))
        local_bound = torch.maximum(spectrum.abs(), analytic_spectrum.abs()) + 0.05 * scale
        correction = torch.complex(components[:, 0], components[:, 1])
        estimate = analytic_spectrum + self.maximum_complex_correction * local_bound * correction
        learned = torch.istft(
            estimate,
            self.n_fft,
            self.hop,
            window=self.window.to(wet),
            center=True,
            length=frames,
        )
        restored = torch.where(exact_available[:, None], exact_base, learned)
        frame_uncertainty = torch.nn.functional.softplus(
            self.uncertainty(hidden.mean(dim=2)).squeeze(1)
        ) + 1.0e-5
        uncertainty = torch.nn.functional.interpolate(
            frame_uncertainty[:, None], size=frames, mode="linear", align_corners=False
        )[:, 0]
        return restored, uncertainty, learned

    def forward(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        profile_features: torch.Tensor,
        analytic_base: torch.Tensor,
        exact_base: torch.Tensor,
        exact_available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        restored, uncertainty, _ = self._forward_components(
            wet, controls, profile_features, analytic_base, exact_base, exact_available
        )
        return restored, uncertainty

    @staticmethod
    def _replay(restored: torch.Tensor, transfer: torch.Tensor) -> torch.Tensor:
        fft_frames = 1 << (restored.shape[1] + transfer.shape[1] - 2).bit_length()
        return torch.fft.irfft(
            torch.fft.rfft(restored, n=fft_frames)
            * torch.fft.rfft(transfer, n=fft_frames),
            n=fft_frames,
        )[:, : restored.shape[1]]

    def training_loss(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        profile_features: torch.Tensor,
        analytic_base: torch.Tensor,
        exact_base: torch.Tensor,
        exact_available: torch.Tensor,
        transfer: torch.Tensor,
        clean: torch.Tensor,
        target_start: int,
        *,
        physical_weight: float = 0.25,
        tail_weight: float = 3.0,
    ) -> tuple[torch.Tensor, dict]:
        if not 0.0 <= physical_weight <= 1.0:
            raise ValueError("physical replay weight must be in [0,1]")
        restored, uncertainty, _ = self._forward_components(
            wet, controls, profile_features, analytic_base, exact_base, exact_available
        )
        loss, parts = ambience_loss(
            restored, uncertainty, wet, clean, target_start, 0.0, tail_weight
        )
        replay = self._replay(restored, transfer)
        scale = wet[:, target_start:].abs().mean().clamp_min(1.0e-4)
        replay_loss = torch.nn.functional.l1_loss(
            replay[:, target_start:], wet[:, target_start:]
        ) / scale
        loss = loss + physical_weight * replay_loss
        parts["physical_replay"] = float(replay_loss.detach())
        parts["exact_fraction"] = float(exact_available.float().mean())
        return loss, parts

    def manifest(self) -> dict:
        spectral_frames = 1 + 2 * sum(self.dilations)
        return {
            "schema": 1,
            "architecture": "profile-conditioned-direct-complex-residual-graybox",
            "mechanism": "ambience",
            "channels": self.channels,
            "depth": self.depth,
            "n_fft": self.n_fft,
            "hop": self.hop,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "profile_features": [
                "exact-transfer-log-magnitude",
                "exact-transfer-unit-real",
                "exact-transfer-unit-imag",
                "frequency-decay-t60",
            ],
            "physical_training_constraint": "reconvolve restored with current transfer to reproduce Wet",
            "analytic_initialization": "exact stable inverse else bounded regularized profile inverse else Wet",
            "learned_output": "bounded direct complex STFT residual",
            "maximum_complex_correction": self.maximum_complex_correction,
            "fixed_candidate_convex_hull": False,
            "exact_stable_inverse_immutable": True,
            "wet_identity_reachable": True,
            "spectral_context_frames": spectral_frames,
            "waveform_context_frames": spectral_frames * self.hop + self.n_fft,
            "inference_inputs": "Wet plus current RIR/profile and current controls only",
            "clean_input_at_runtime": False,
            "chain_order_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        }
