"""Equal-budget independent experts for the restoration foundation comparison."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


MECHANISMS = ("nonlinear", "dynamics", "temporal")


def _smooth(value: torch.Tensor, kernel: int) -> torch.Tensor:
    return F.avg_pool1d(value, kernel, stride=1, padding=kernel // 2)


class Block(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            9,
            padding=4 * dilation,
            dilation=dilation,
            groups=channels,
        )
        self.mix = nn.Conv1d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.norm(value)
        value = F.gelu(self.depthwise(value))
        return residual + self.mix(value)


class Expert(nn.Module):
    """One independent mechanism expert with a mechanism-specific fixed view.

    All three variants have exactly the same trainable parameter budget.  The
    fixed input views encode different inverse priors without forcing nonlinear
    clipping, compressor envelopes and long tails through one output head.
    """

    def __init__(self, mechanism: str, channels: int = 24, blocks: int = 7) -> None:
        super().__init__()
        if mechanism not in MECHANISMS:
            raise ValueError(f"unsupported restoration mechanism: {mechanism}")
        if channels < 4 or blocks < 1:
            raise ValueError("expert geometry is too small")
        self.mechanism = mechanism
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Conv1d(4, channels, 15, padding=7)
        self.body = nn.ModuleList(
            Block(channels, 2 ** (index % 8)) for index in range(blocks)
        )
        self.head = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv1d(channels, 1, 15, padding=7),
        )
        # Begin very close to the safe identity baseline.  Training must earn
        # every correction rather than emitting arbitrary noise at step zero.
        nn.init.normal_(self.head[-1].weight, std=1.0e-4)
        nn.init.zeros_(self.head[-1].bias)

    def view(self, wet: torch.Tensor) -> torch.Tensor:
        if wet.ndim == 2:
            wet = wet[:, None, :]
        if wet.ndim != 3 or wet.shape[1] != 1:
            raise ValueError(f"expected [batch, frames] or [batch, 1, frames], got {wet.shape}")
        if self.mechanism == "nonlinear":
            difference = F.pad(wet[..., 1:] - wet[..., :-1], (1, 0))
            return torch.cat((wet, torch.tanh(wet * 4.0), difference, _smooth(wet, 65)), dim=1)
        envelope = torch.abs(wet)
        if self.mechanism == "dynamics":
            return torch.cat((wet, _smooth(envelope, 65), _smooth(envelope, 513), _smooth(wet, 513)), dim=1)
        return torch.cat((wet, _smooth(wet, 65), _smooth(wet, 513), _smooth(wet, 4095)), dim=1)

    def forward(self, wet: torch.Tensor) -> torch.Tensor:
        mono = wet[:, None, :] if wet.ndim == 2 else wet
        value = self.stem(self.view(wet))
        for block in self.body:
            value = block(value)
        correction = self.head(value)
        restored = mono + correction
        return restored[:, 0, :] if wet.ndim == 2 else restored

    def manifest(self) -> dict:
        input_context_frames = {
            "nonlinear": 65,
            "dynamics": 513,
            "temporal": 4095,
        }[self.mechanism]
        return {
            "mechanism": self.mechanism,
            "channels": self.channels,
            "blocks": self.blocks,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "input_context_frames": input_context_frames,
            "input_view": {
                "nonlinear": ["wet", "softclip", "difference", "65-sample mean"],
                "dynamics": ["wet", "65-sample envelope", "513-sample envelope", "513-sample mean"],
                "temporal": ["wet", "65-sample mean", "513-sample mean", "4095-sample mean"],
            }[self.mechanism],
        }


def restoration_loss(restored: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, dict]:
    waveform = F.l1_loss(restored, clean)
    restored_difference = restored[..., 1:] - restored[..., :-1]
    clean_difference = clean[..., 1:] - clean[..., :-1]
    transient = F.l1_loss(restored_difference, clean_difference)
    envelope = F.l1_loss(_smooth(torch.abs(restored[:, None]), 65), _smooth(torch.abs(clean[:, None]), 65))
    long_shape = F.l1_loss(_smooth(restored[:, None], 513), _smooth(clean[:, None], 513))
    loss = waveform + 0.5 * transient + 0.25 * envelope + 0.15 * long_shape
    return loss, {
        "waveform": float(waveform.detach()),
        "transient": float(transient.detach()),
        "envelope": float(envelope.detach()),
        "long_shape": float(long_shape.detach()),
    }
