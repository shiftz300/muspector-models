"""Small differentiable DSP used only for audio reconstruction supervision."""

from __future__ import annotations

import math

import torch

from .model import Estimate


RATE = 11_025
KIND_DRIVE = 0
KIND_DELAY = 1
KIND_REVERB = 2


def _rfft_filter(value: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    return torch.fft.irfft(torch.fft.rfft(value) * response, n=value.shape[-1])


def drive(value: torch.Tensor, controls: torch.Tensor, renderer: int) -> torch.Tensor:
    gain_db = controls[0] * 30.0
    tone = controls[1]
    level_db = controls[2] * 30.0 - 18.0
    gain = torch.pow(10.0, gain_db / 20.0)
    if renderer == 0:
        shaped = torch.tanh(value * gain) / torch.tanh(gain).clamp_min(1.0e-6)
        cutoff = 900.0 * torch.pow(12_000.0 / 900.0, tone)
    else:
        shaped = torch.atan(value * gain * 1.7) / torch.atan(gain * 1.7).clamp_min(1.0e-6)
        cutoff = 700.0 * torch.pow(14_000.0 / 700.0, tone)
    frequency = torch.fft.rfftfreq(len(value), 1.0 / RATE).to(value.device)
    response = torch.rsqrt(1.0 + torch.pow(frequency / cutoff.clamp_min(30.0), 4.0))
    return _rfft_filter(shaped, response) * torch.pow(10.0, level_db / 20.0)


def delay(value: torch.Tensor, controls: torch.Tensor, renderer: int) -> torch.Tensor:
    time_ms = 40.0 * torch.pow(25.0, controls[3])
    feedback = controls[4] * 0.9
    mix = controls[5] * 0.7
    length = len(value) * 2
    spectrum = torch.fft.rfft(value, n=length)
    omega = 2.0 * math.pi * torch.fft.rfftfreq(length).to(value.device)
    phase = torch.exp(-1j * omega * time_ms * RATE / 1_000.0)
    repeats = phase / (1.0 - feedback * phase)
    if renderer == 1:
        frequency = torch.fft.rfftfreq(length, 1.0 / RATE).to(value.device)
        repeats = repeats / torch.sqrt(1.0 + torch.pow(frequency / 5_500.0, 4.0))
    response = (1.0 - mix) + mix * repeats
    return torch.fft.irfft(spectrum * response, n=length)[: len(value)]


def reverb(value: torch.Tensor, controls: torch.Tensor, renderer: int) -> torch.Tensor:
    decay = 0.2 * torch.pow(40.0, controls[6])
    damping = controls[7]
    mix = controls[8] * 0.7
    length = len(value)
    time = torch.arange(length, device=value.device, dtype=value.dtype) / RATE
    # A deterministic dense tail keeps gradients reproducible.
    seed = 0.731 if renderer == 0 else 1.173
    noise = torch.sin((time * RATE + 1.0) * seed * 17.0) * torch.sin(
        (time * RATE + 3.0) * seed * 41.0
    )
    impulse = noise * torch.pow(10.0, -3.0 * time / decay.clamp_min(0.05))
    cutoff_high, cutoff_low = ((12_000.0, 700.0) if renderer == 0 else (10_000.0, 500.0))
    cutoff = cutoff_high * torch.pow(cutoff_low / cutoff_high, damping)
    padded = length * 2
    frequency = torch.fft.rfftfreq(padded, 1.0 / RATE).to(value.device)
    response = torch.rsqrt(1.0 + torch.pow(frequency / cutoff.clamp_min(30.0), 4.0))
    impulse_spectrum = torch.fft.rfft(impulse, n=padded) * response
    impulse = torch.fft.irfft(impulse_spectrum, n=padded)[:length]
    impulse = impulse / torch.sqrt(torch.sum(impulse.square())).clamp_min(1.0e-6)
    wet = torch.fft.irfft(
        torch.fft.rfft(value, n=padded) * torch.fft.rfft(impulse, n=padded),
        n=padded,
    )[:length]
    return value * (1.0 - mix) + wet * mix


def render(
    dry: torch.Tensor,
    controls: torch.Tensor,
    topology: torch.Tensor,
    renderer: int,
) -> torch.Tensor:
    value = dry
    for encoded in topology:
        kind = int(encoded)
        if kind < 0:
            break
        if kind == KIND_DRIVE:
            value = drive(value, controls, renderer)
        elif kind == KIND_DELAY:
            value = delay(value, controls, renderer)
        elif kind == KIND_REVERB:
            value = reverb(value, controls, renderer)
        else:
            raise ValueError(f"unknown effect kind: {kind}")
    return value


def audio_loss(
    estimate: Estimate,
    batch: dict[str, torch.Tensor],
    maximum_examples: int = 2,
) -> torch.Tensor:
    """Re-render predicted controls using ground-truth topology and compare audio."""

    selected = torch.nonzero(batch["reconstruction_mask"] > 0.5).flatten()[:maximum_examples]
    if not len(selected):
        return estimate.control_logits.sum() * 0.0
    predictions = torch.sigmoid(estimate.control_logits)
    losses = []
    for index_tensor in selected:
        index = int(index_tensor)
        dry = torch.nn.functional.avg_pool1d(
            batch["dry"][index][None, None], 4, 4
        ).flatten()
        target = torch.nn.functional.avg_pool1d(
            batch["wet"][index][None, None], 4, 4
        ).flatten()
        reconstructed = render(
            dry,
            predictions[index],
            batch["topology"][index],
            int(batch["renderer"][index]),
        )
        scale = torch.sqrt(torch.mean(target.square())).clamp_min(1.0e-4)
        waveform = torch.mean(torch.abs(reconstructed - target)) / scale
        spectral = 0.0
        for fft in (512, 2_048):
            window = torch.hann_window(fft, device=target.device)
            expected = torch.stft(target, fft, fft // 4, window=window, return_complex=True)
            actual = torch.stft(
                reconstructed, fft, fft // 4, window=window, return_complex=True
            )
            spectral = spectral + torch.mean(
                torch.abs(torch.log1p(actual.abs()) - torch.log1p(expected.abs()))
            )
        losses.append(waveform + spectral)
    return torch.stack(losses).mean()
