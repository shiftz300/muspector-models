"""Control-conditioned causal forward renderer for Drive research."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import soundfile
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .pedalboard_renderer import render_pedalboard_chain
from .quality import checked_audio, validate_render
from .render import render_chain
from .spec import ChainSpec, Drive


FORWARD_RATE = 48_000
ForwardDomain = Literal["reference", "alternate", "stress", "challenge", "pedalboard"]


class DriveForwardRenderer(torch.nn.Module):
    """One-layer residual LSTM conditioned on normalized physical controls."""

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.recurrent = torch.nn.LSTM(3, hidden_size, batch_first=True)
        self.control_gain = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 1),
        )
        self.output = torch.nn.Linear(hidden_size, 1)
        for name, parameter in self.recurrent.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(parameter)
                parameter.requires_grad_(False)
        torch.nn.init.zeros_(self.control_gain[-1].weight)
        torch.nn.init.zeros_(self.control_gain[-1].bias)
        torch.nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-3)
        torch.nn.init.zeros_(self.output.bias)
        self.output.bias.requires_grad_(False)

    def forward(
        self,
        dry: torch.Tensor,
        controls: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Shape/range admission stays at the Python/runtime boundary. Tracing
        # cannot preserve Python exceptions, so keep them out of the ONNX graph.
        if not torch.jit.is_tracing():
            if dry.ndim != 2:
                raise ValueError(f"expected dry [batch,time], got {tuple(dry.shape)}")
            if controls.ndim != 2 or controls.shape != (dry.shape[0], 3):
                raise ValueError(
                    f"expected controls [batch,3] for {dry.shape[0]} examples, "
                    f"got {tuple(controls.shape)}"
                )
            if not torch.isfinite(dry).all() or not torch.isfinite(controls).all():
                raise ValueError("forward renderer input contains non-finite values")
            if torch.any(controls < 0.0) or torch.any(controls > 1.0):
                raise ValueError("forward renderer controls must be normalized to [0,1]")
        nonlinear_controls = controls[:, :2]
        centered_controls = nonlinear_controls.mul(2.0).sub(1.0).unsqueeze(1)
        recurrent_input = torch.cat(
            (
                dry.unsqueeze(-1),
                dry.unsqueeze(-1) * centered_controls,
            ),
            dim=-1,
        )
        hidden, next_state = self.recurrent(recurrent_input, state)
        learned_gain = torch.exp(self.control_gain(nonlinear_controls).clamp(-2.0, 4.0))
        level_db = controls[:, 2:3] * 30.0 - 18.0
        physical_level_gain = torch.pow(10.0, level_db / 20.0)
        rendered = torch.tanh(
            (dry + self.output(hidden).squeeze(-1)) * learned_gain
        ) * physical_level_gain
        if not torch.jit.is_tracing() and not torch.isfinite(rendered).all():
            raise ValueError("forward renderer produced non-finite audio")
        return rendered, next_state


def error_to_signal_ratio(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(f"ESR expects matching [batch,time], got {prediction.shape}/{target.shape}")
    error_energy = (prediction - target).square().mean(dim=1)
    target_energy = target.square().mean(dim=1).clamp_min(1.0e-8)
    return (error_energy / target_energy).mean()


def pre_emphasis(audio: torch.Tensor, coefficient: float = 0.85) -> torch.Tensor:
    if audio.ndim != 2:
        raise ValueError(f"pre-emphasis expects [batch,time], got {audio.shape}")
    first = audio[:, :1]
    remainder = audio[:, 1:] - coefficient * audio[:, :-1]
    return torch.cat((first, remainder), dim=1)


def multiresolution_spectral_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    fft_sizes: Sequence[int] = (256, 512, 1_024),
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("spectral loss expects matching audio")
    losses = []
    for fft_size in fft_sizes:
        if prediction.shape[1] < fft_size:
            continue
        window = torch.hann_window(fft_size, device=prediction.device, dtype=prediction.dtype)
        prediction_magnitude = torch.stft(
            prediction,
            n_fft=fft_size,
            hop_length=fft_size // 4,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        ).abs()
        target_magnitude = torch.stft(
            target,
            n_fft=fft_size,
            hop_length=fft_size // 4,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        ).abs()
        log_distance = torch.nn.functional.l1_loss(
            torch.log1p(prediction_magnitude),
            torch.log1p(target_magnitude),
        )
        spectral_convergence = torch.linalg.vector_norm(
            prediction_magnitude - target_magnitude,
            dim=(-2, -1),
        ) / torch.linalg.vector_norm(target_magnitude, dim=(-2, -1)).clamp_min(1.0e-8)
        losses.append(log_distance + spectral_convergence.mean())
    if not losses:
        return prediction.square().mean() * 0.0
    return torch.stack(losses).mean()


def forward_loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    waveform = error_to_signal_ratio(prediction, target)
    emphasized = error_to_signal_ratio(pre_emphasis(prediction), pre_emphasis(target))
    spectral = multiresolution_spectral_loss(prediction, target)
    normalized_l1 = torch.mean(torch.abs(prediction - target)) / torch.mean(
        torch.abs(target)
    ).clamp_min(1.0e-6)
    relative_peak = torch.mean(
        torch.abs(prediction.abs().amax(dim=1) - target.abs().amax(dim=1))
        / target.abs().amax(dim=1).clamp_min(1.0e-5)
    )
    amplitude_overshoot = torch.mean(
        torch.relu(torch.abs(prediction) - torch.abs(target))
    ) / torch.mean(torch.abs(target)).clamp_min(1.0e-6)
    total = (
        waveform
        + 0.5 * emphasized
        + 0.1 * spectral
        + 0.25 * normalized_l1
        + 0.5 * relative_peak
        + 0.25 * amplitude_overshoot
    )
    return total, {
        "esr": float(waveform.detach()),
        "preemphasis_esr": float(emphasized.detach()),
        "spectral": float(spectral.detach()),
        "normalized_l1": float(normalized_l1.detach()),
        "relative_peak": float(relative_peak.detach()),
        "amplitude_overshoot": float(amplitude_overshoot.detach()),
    }


def normalized_drive_controls(effect: Drive) -> np.ndarray:
    effect.validate()
    return np.asarray(
        (
            effect.gain_db / 30.0,
            effect.tone,
            (effect.level_db + 18.0) / 30.0,
        ),
        dtype=np.float32,
    )


def controls_to_drive(controls: Sequence[float]) -> Drive:
    if len(controls) != 3:
        raise ValueError(f"expected three Drive controls, got {len(controls)}")
    values = tuple(float(value) for value in controls)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"normalized Drive controls are invalid: {values}")
    return Drive(values[0] * 30.0, values[1], values[2] * 30.0 - 18.0)


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty((count, dimensions), dtype=np.float32)
    for dimension in range(dimensions):
        strata = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
        result[:, dimension] = strata[rng.permutation(count)]
    return result


def _segment(path: Path, frames: int, rng: random.Random) -> np.ndarray:
    info = soundfile.info(path)
    native_frames = math.ceil(frames * info.samplerate / FORWARD_RATE) + 16
    start = rng.randrange(max(1, info.frames - native_frames + 1))
    with soundfile.SoundFile(path) as source:
        source.seek(start)
        value = source.read(native_frames, dtype="float32", always_2d=True).mean(axis=1)
    if info.samplerate != FORWARD_RATE:
        common = math.gcd(info.samplerate, FORWARD_RATE)
        value = resample_poly(
            value,
            FORWARD_RATE // common,
            info.samplerate // common,
        ).astype(np.float32)
    if len(value) < frames:
        value = np.pad(value, (0, frames - len(value)))
    return checked_audio(value[:frames], name=f"training segment {path.name}")


class DriveForwardDataset(Dataset):
    """Offline synthetic-control pairs over guitar-disjoint source pools."""

    def __init__(
        self,
        dry_paths: Sequence[Path],
        samples: int,
        frames: int,
        *,
        seed: int,
        domains: Sequence[ForwardDomain],
        input_peak_range: tuple[float, float] = (0.08, 0.24),
        input_peak_ranges: Sequence[tuple[float, float]] | None = None,
    ) -> None:
        if not dry_paths or samples <= 0 or frames < 1_024 or not domains:
            raise ValueError("forward dataset needs sources, samples, domains, and >=1024 frames")
        peak_ranges = (
            (input_peak_range,) if input_peak_ranges is None else tuple(input_peak_ranges)
        )
        if not peak_ranges or any(
            len(values) != 2
            or not all(math.isfinite(value) for value in values)
            or not 0.0 < values[0] <= values[1] <= 1.0
            for values in peak_ranges
        ):
            raise ValueError("input peak ranges must be finite, positive, ordered, and <=1")
        self.dry_paths = tuple(dry_paths)
        self.samples = samples
        self.frames = frames
        self.seed = seed
        self.domains = tuple(domains)
        self.input_peak_ranges = peak_ranges
        self.controls = _latin_hypercube(samples, 3, seed ^ 0x44525645)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rng = random.Random(self.seed + index * 104_729)
        dry = _segment(self.dry_paths[index % len(self.dry_paths)], self.frames, rng)
        peak = max(float(np.max(np.abs(dry))), 1.0e-5)
        # Training-only level augmentation; source files remain read-only and
        # inference/render output never receives automatic normalization.
        target_peak = rng.uniform(*self.input_peak_ranges[index % len(self.input_peak_ranges)])
        dry = (dry * (target_peak / peak)).astype(np.float32)
        controls = self.controls[index]
        effect = controls_to_drive(controls)
        domain = self.domains[index % len(self.domains)]
        spec = ChainSpec((effect,))
        if domain == "pedalboard":
            wet = render_pedalboard_chain(dry, spec, FORWARD_RATE)
        else:
            wet = render_chain(dry, spec, FORWARD_RATE, domain)
        validate_render(dry, wet)
        return {
            "dry": torch.from_numpy(dry.copy()),
            "wet": torch.from_numpy(wet.copy()),
            "controls": torch.from_numpy(controls.copy()),
            "domain": domain,
        }
