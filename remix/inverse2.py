"""Independent control-conditioned recurrent inverse experts."""

from __future__ import annotations

import torch
from torch import nn

from .product2 import CONTROL_WIDTH


TRAINABLE_MECHANISMS = ("nonlinear", "dynamics")


def _pre_emphasis(audio: torch.Tensor, coefficient: float = 0.85) -> torch.Tensor:
    return torch.cat((audio[:, :1], audio[:, 1:] - coefficient * audio[:, :-1]), dim=1)


def _spectral_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for fft_size in (256, 512, 1024):
        if prediction.shape[1] < fft_size:
            continue
        window = torch.hann_window(fft_size, device=prediction.device, dtype=prediction.dtype)
        predicted = torch.stft(
            prediction, fft_size, fft_size // 4, window=window,
            center=True, pad_mode="constant", return_complex=True,
        ).abs()
        expected = torch.stft(
            target, fft_size, fft_size // 4, window=window,
            center=True, pad_mode="constant", return_complex=True,
        ).abs()
        log_distance = torch.nn.functional.l1_loss(torch.log1p(predicted), torch.log1p(expected))
        convergence = torch.linalg.vector_norm(predicted - expected, dim=(-2, -1)) / torch.linalg.vector_norm(
            expected, dim=(-2, -1)
        ).clamp_min(1.0e-8)
        losses.append(log_distance + convergence.mean())
    return torch.stack(losses).mean()


def _nonlinear_preconditioner(wet: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
    drive = 1.8 + controls[:, 0:1] * (8.0 - 1.8)
    bias = -0.12 + controls[:, 1:2] * 0.24
    level = 0.45 + controls[:, 3:4] * 0.40
    shaped = wet / level
    tanh_base = (
        torch.atanh((shaped + torch.tanh(bias)).clamp(-0.995, 0.995)) - bias
    ) / drive
    atan_offset = (2.0 / torch.pi) * torch.atan(bias * 1.8)
    atan_base = (
        torch.tan(((shaped + atan_offset) * torch.pi / 2.0).clamp(-1.52, 1.52)) / 1.8 - bias
    ) / drive
    target = shaped.clamp(-1.0, 1.0)
    low = torch.full_like(target, -1.5)
    high = torch.full_like(target, 1.5)
    for _ in range(20):
        middle = (low + high) * 0.5
        value = middle - middle.pow(3) / 6.75
        low = torch.where(value < target, middle, low)
        high = torch.where(value >= target, middle, high)
    cubic_base = ((low + high) * 0.5 - bias) / drive
    shape = controls[:, 4:5]
    return torch.where(shape < 0.25, tanh_base, torch.where(shape < 0.75, atan_base, cubic_base))


class InverseExpert(nn.Module):
    def __init__(self, mechanism: str, hidden_size: int = 20, layers: int = 2) -> None:
        super().__init__()
        if mechanism not in TRAINABLE_MECHANISMS or hidden_size < 4 or layers < 1:
            raise ValueError("invalid inverse2 expert geometry")
        self.mechanism = mechanism
        self.hidden_size = hidden_size
        self.layers = layers
        if mechanism == "nonlinear":
            self.depth = max(6, layers * 3)
            self.input = nn.Conv1d(4 + CONTROL_WIDTH, hidden_size, 1)
            dilations = tuple(2 ** index for index in range(self.depth))
            self.dilations = dilations
            self.temporal = nn.ModuleList([
                nn.Conv1d(hidden_size, hidden_size * 2, 7, padding=3 * dilation, dilation=dilation)
                for dilation in dilations
            ])
            self.mix = nn.ModuleList([nn.Conv1d(hidden_size, hidden_size, 1) for _ in dilations])
            self.correction = nn.Conv1d(hidden_size, 1, 1)
            self.uncertainty = nn.Conv1d(hidden_size, 1, 1)
            nn.init.normal_(self.correction.weight, mean=0.0, std=1.0e-4)
            nn.init.zeros_(self.correction.bias)
            nn.init.zeros_(self.uncertainty.weight)
            nn.init.constant_(self.uncertainty.bias, -4.0)
        else:
            self.recurrent = nn.GRU(4 + CONTROL_WIDTH, hidden_size, layers, batch_first=True)
            self.correction = nn.Linear(hidden_size, 1)
            self.direct_controls = nn.Linear(CONTROL_WIDTH, 1)
            self.uncertainty = nn.Linear(hidden_size, 1)
            nn.init.normal_(self.correction.weight, mean=0.0, std=1.0e-4)
            nn.init.zeros_(self.correction.bias)
            nn.init.zeros_(self.direct_controls.weight)
            nn.init.zeros_(self.direct_controls.bias)
            with torch.no_grad():
                self.direct_controls.weight[0, 4] = -8.0 * torch.log(torch.tensor(10.0)) / 20.0
            nn.init.zeros_(self.uncertainty.weight)
            nn.init.constant_(self.uncertainty.bias, -4.0)

    def forward(
        self,
        wet: torch.Tensor,
        controls: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if wet.ndim != 2 or controls.shape != (wet.shape[0], CONTROL_WIDTH):
            raise ValueError("inverse2 expects wet [batch,time] and fixed-width controls")
        condition = controls.mul(2.0).sub(1.0).unsqueeze(1).expand(-1, wet.shape[1], -1)
        difference = torch.nn.functional.pad(wet[:, 1:] - wet[:, :-1], (1, 0))
        if self.mechanism == "nonlinear":
            if state is not None:
                raise ValueError("nonlinear local expert does not accept recurrent state")
            base = _nonlinear_preconditioner(wet, controls)
            base_difference = torch.nn.functional.pad(base[:, 1:] - base[:, :-1], (1, 0))
            features = torch.cat(
                (
                    wet.unsqueeze(-1),
                    wet.abs().unsqueeze(-1),
                    base.unsqueeze(-1),
                    base_difference.unsqueeze(-1),
                    condition,
                ),
                dim=-1,
            ).transpose(1, 2)
            hidden = torch.tanh(self.input(features))
            for temporal, mix in zip(self.temporal, self.mix, strict=True):
                value, gate = temporal(hidden).chunk(2, dim=1)
                hidden = hidden + 0.25 * mix(torch.tanh(value) * torch.sigmoid(gate))
            correction = 0.25 * torch.tanh(self.correction(hidden).squeeze(1))
            uncertainty = torch.nn.functional.softplus(self.uncertainty(hidden).squeeze(1)) + 1.0e-5
            return base + correction, uncertainty, None
        log_level = torch.log(wet.abs().clamp_min(1.0e-5)).unsqueeze(-1)
        features = torch.cat(
            (wet.unsqueeze(-1), wet.abs().unsqueeze(-1), difference.unsqueeze(-1), log_level, condition), dim=-1
        )
        hidden, next_state = self.recurrent(features, state)
        log_correction = self.correction(hidden).squeeze(-1) + self.direct_controls(controls).expand(-1, wet.shape[1])
        restored = wet * torch.exp(log_correction.clamp(-2.0, 2.0))
        uncertainty = torch.nn.functional.softplus(self.uncertainty(hidden).squeeze(-1)) + 1.0e-5
        return restored, uncertainty, next_state

    def manifest(self) -> dict:
        return {
            "schema": 3,
            "architecture": (
                "analytic-shape-inverse-plus-local-tcn-with-uncertainty"
                if self.mechanism == "nonlinear"
                else "control-conditioned-causal-gru-gain-with-uncertainty"
            ),
            "mechanism": self.mechanism,
            "hidden_size": self.hidden_size,
            "layers": self.layers,
            "control_width": CONTROL_WIDTH,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "state_floats_per_mono_stream": (
                self.layers * self.hidden_size if self.mechanism == "dynamics" else 0
            ),
            "normalization_across_time": False,
            "causal": self.mechanism == "dynamics",
            "uncertainty_output": True,
            "local_receptive_field_frames": (
                1 + 6 * sum(self.dilations) if self.mechanism == "nonlinear" else None
            ),
        }


def inverse_loss(
    restored: torch.Tensor,
    uncertainty: torch.Tensor,
    clean: torch.Tensor,
    target_start: int,
    *,
    wet: torch.Tensor | None = None,
    inverse_log_gain: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    restored = restored[:, target_start:]
    uncertainty = uncertainty[:, target_start:]
    clean = clean[:, target_start:]
    scale = clean.abs().mean().clamp_min(1.0e-5)
    waveform = torch.nn.functional.l1_loss(restored, clean) / scale
    restored_difference = restored[:, 1:] - restored[:, :-1]
    clean_difference = clean[:, 1:] - clean[:, :-1]
    transient_scale = clean_difference.abs().mean().clamp_min(1.0e-5)
    transient = torch.nn.functional.l1_loss(restored_difference, clean_difference) / transient_scale
    clean_frames = clean.unfold(1, 240, 120)
    restored_frames = restored.unfold(1, 240, 120)
    clean_rms = clean_frames.square().mean(dim=-1).add(1.0e-10).sqrt()
    restored_rms = restored_frames.square().mean(dim=-1).add(1.0e-10).sqrt()
    clean_attacks = torch.relu(torch.diff(torch.log(clean_rms + 1.0e-6), dim=1))
    restored_attacks = torch.relu(torch.diff(torch.log(restored_rms + 1.0e-6), dim=1))
    attack = torch.nn.functional.l1_loss(restored_attacks, clean_attacks) / clean_attacks.abs().mean().clamp_min(1.0e-5)
    emphasized_clean = _pre_emphasis(clean)
    emphasized_restored = _pre_emphasis(restored)
    emphasized = torch.nn.functional.l1_loss(emphasized_restored, emphasized_clean) / emphasized_clean.abs().mean().clamp_min(1.0e-5)
    spectral = _spectral_loss(restored, clean)
    clean_envelope = torch.nn.functional.avg_pool1d(clean.abs().unsqueeze(1), 257, 1, 128)
    restored_envelope = torch.nn.functional.avg_pool1d(restored.abs().unsqueeze(1), 257, 1, 128)
    envelope = torch.nn.functional.l1_loss(restored_envelope, clean_envelope) / clean_envelope.mean().clamp_min(1.0e-5)
    uncertainty_target = (restored - clean).abs().detach()
    uncertainty_loss = torch.nn.functional.l1_loss(uncertainty, uncertainty_target) / scale
    loss = (
        waveform
        + 0.20 * transient
        + 0.50 * emphasized
        + 0.10 * spectral
        + 0.25 * envelope
        + 2.0 * attack
        + 0.05 * uncertainty_loss
    )
    parts = {
        "waveform": float(waveform.detach()),
        "transient": float(transient.detach()),
        "attack": float(attack.detach()),
        "preemphasis": float(emphasized.detach()),
        "spectral": float(spectral.detach()),
        "envelope": float(envelope.detach()),
        "uncertainty": float(uncertainty_loss.detach()),
    }
    if inverse_log_gain is not None:
        if wet is None:
            raise ValueError("gain supervision requires wet audio")
        wet = wet[:, target_start:]
        inverse_log_gain = inverse_log_gain[:, target_start:]
        predicted_log_gain = torch.log(restored.abs().add(1.0e-5)) - torch.log(wet.abs().add(1.0e-5))
        active = wet.abs() >= 1.0e-4
        gain = torch.nn.functional.l1_loss(
            predicted_log_gain[active], inverse_log_gain[active]
        ) if active.any() else predicted_log_gain.square().mean() * 0.0
        loss = loss + 0.75 * gain
        parts["gain"] = float(gain.detach())
    return loss, parts
