"""Independent long-context ambience inverse with explicit early/late evidence."""

from __future__ import annotations

import torch
from torch import nn

from .ambience2 import CONTROL_WIDTH
from .inverse2 import _pre_emphasis, _spectral_loss


class _LongBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(
            channels,
            channels * 2,
            (3, 3),
            padding=(1, dilation),
            dilation=(1, dilation),
            groups=channels,
        )
        self.mix = nn.Conv2d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        content, gate = self.temporal(value).chunk(2, dim=1)
        return value + 0.25 * self.mix(torch.tanh(content) * torch.sigmoid(gate))


class AmbienceExpert(nn.Module):
    """Bounded 2.7-second spectral context; no chain or neighboring-effect input."""

    def __init__(self, channels: int = 8, depth: int = 8, n_fft: int = 1024, hop: int = 256) -> None:
        super().__init__()
        if channels < 4 or not 6 <= depth <= 9 or n_fft < 512 or hop < 64:
            raise ValueError("invalid ambience2 expert geometry")
        self.channels = channels
        self.depth = depth
        self.n_fft = n_fft
        self.hop = hop
        self.dilations = tuple(2**index for index in range(depth))
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.stem = nn.Conv2d(6 + CONTROL_WIDTH, channels, (5, 3), padding=(2, 1))
        self.blocks = nn.ModuleList(_LongBlock(channels, dilation) for dilation in self.dilations)
        self.correction = nn.Conv2d(channels, 2, 1)
        self.uncertainty = nn.Conv1d(channels, 1, 1)
        nn.init.normal_(self.correction.weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.correction.bias)
        nn.init.zeros_(self.uncertainty.weight)
        nn.init.constant_(self.uncertainty.bias, -4.0)

    def forward(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        late_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if wet.ndim != 2 or controls.shape != (wet.shape[0], CONTROL_WIDTH):
            raise ValueError("ambience2 expects wet [batch,time] and controls [batch,3]")
        if not torch.isfinite(wet).all() or not torch.isfinite(controls).all():
            raise ValueError("ambience2 inputs must be finite")
        if late_base.shape != wet.shape or not torch.isfinite(late_base).all():
            raise ValueError("ambience2 late base must match Wet geometry")
        spectrum = torch.stft(
            wet,
            self.n_fft,
            self.hop,
            window=self.window.to(wet),
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        scale = spectrum.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1.0e-6)
        wet_features = torch.stack(
            (
                spectrum.real / scale,
                spectrum.imag / scale,
                torch.log1p(spectrum.abs() / scale),
            ),
            dim=1,
        )
        base_spectrum = torch.stft(
            late_base,
            self.n_fft,
            self.hop,
            window=self.window.to(wet),
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        base_features = torch.stack(
            (
                base_spectrum.real / scale,
                base_spectrum.imag / scale,
                torch.log1p(base_spectrum.abs() / scale),
            ),
            dim=1,
        )
        features = torch.cat((wet_features, base_features), dim=1)
        condition = controls.mul(2.0).sub(1.0)[:, :, None, None].expand(
            -1, -1, features.shape[2], features.shape[3]
        )
        hidden = torch.tanh(self.stem(torch.cat((features, condition), dim=1)))
        for block in self.blocks:
            hidden = block(hidden)
        residual = self.correction(hidden)
        estimate = base_spectrum + torch.complex(residual[:, 0], residual[:, 1]) * scale * 0.25
        restored = torch.istft(
            estimate,
            self.n_fft,
            self.hop,
            window=self.window.to(wet),
            center=True,
            length=wet.shape[1],
        )
        frame_uncertainty = torch.nn.functional.softplus(
            self.uncertainty(hidden.mean(dim=2)).squeeze(1)
        ) + 1.0e-5
        uncertainty = torch.nn.functional.interpolate(
            frame_uncertainty.unsqueeze(1), size=wet.shape[1], mode="linear", align_corners=False
        ).squeeze(1)
        return restored, uncertainty

    def manifest(self) -> dict:
        spectral_frames = 1 + 2 * sum(self.dilations)
        return {
            "schema": 1,
            "architecture": "blind-late-prediction-plus-early-residual-tcn-with-uncertainty",
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
            "causal": False,
            "uncertainty_output": True,
            "blind_late_suppression_strength": 0.25,
            "chain_order_input": False,
            "neighbor_effect_input": False,
        }


def _frame_rms(audio: torch.Tensor, frame: int = 480, hop: int = 120) -> torch.Tensor:
    return audio.unfold(1, frame, hop).square().mean(dim=-1).add(1.0e-10).sqrt()


def ambience_loss(
    restored: torch.Tensor,
    uncertainty: torch.Tensor,
    wet: torch.Tensor,
    clean: torch.Tensor,
    target_start: int,
) -> tuple[torch.Tensor, dict]:
    restored = restored[:, target_start:]
    uncertainty = uncertainty[:, target_start:]
    wet = wet[:, target_start:]
    clean = clean[:, target_start:]
    scale = clean.abs().mean().clamp_min(1.0e-5)
    waveform = torch.nn.functional.l1_loss(restored, clean) / scale
    emphasized = torch.nn.functional.l1_loss(
        _pre_emphasis(restored), _pre_emphasis(clean)
    ) / _pre_emphasis(clean).abs().mean().clamp_min(1.0e-5)
    spectral = _spectral_loss(restored, clean)
    clean_rms = _frame_rms(clean)
    wet_rms = _frame_rms(wet)
    restored_rms = _frame_rms(restored)
    envelope = torch.nn.functional.l1_loss(restored_rms, clean_rms) / clean_rms.mean().clamp_min(1.0e-5)
    clean_attacks = torch.relu(torch.diff(torch.log(clean_rms + 1.0e-6), dim=1))
    restored_attacks = torch.relu(torch.diff(torch.log(restored_rms + 1.0e-6), dim=1))
    attack = torch.nn.functional.l1_loss(restored_attacks, clean_attacks) / clean_attacks.mean().clamp_min(1.0e-5)
    tail_losses = []
    for batch_index in range(clean.shape[0]):
        active = clean_rms[batch_index] > torch.quantile(clean_rms[batch_index].detach(), 0.75)
        recent = torch.nn.functional.conv1d(
            torch.nn.functional.pad(active.float()[None, None], (39, 0)),
            torch.ones(1, 1, 40, device=clean.device),
        )[0, 0] > 0
        quiet = clean_rms[batch_index] <= torch.quantile(clean_rms[batch_index].detach(), 0.35)
        mask = quiet & recent
        if mask.any():
            baseline = (wet_rms[batch_index, mask] - clean_rms[batch_index, mask]).abs().mean().clamp_min(1.0e-5)
            tail_losses.append(
                (restored_rms[batch_index, mask] - clean_rms[batch_index, mask]).abs().mean() / baseline
            )
    tail = torch.stack(tail_losses).mean() if tail_losses else waveform * 0.0
    uncertainty_target = (restored - clean).abs().detach()
    uncertainty_loss = torch.nn.functional.l1_loss(uncertainty, uncertainty_target) / scale
    loss = waveform + 0.3 * emphasized + 0.1 * spectral + 0.5 * envelope + 0.5 * attack + tail + 0.05 * uncertainty_loss
    return loss, {
        "waveform": float(waveform.detach()),
        "preemphasis": float(emphasized.detach()),
        "spectral": float(spectral.detach()),
        "envelope": float(envelope.detach()),
        "attack": float(attack.detach()),
        "tail": float(tail.detach()),
        "uncertainty": float(uncertainty_loss.detach()),
    }
