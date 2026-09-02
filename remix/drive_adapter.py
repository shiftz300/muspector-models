"""Small causal per-device residual adapter for the frozen Drive renderer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .forward_drive import FORWARD_RATE, DriveForwardRenderer


class DriveDeviceAdapter(torch.nn.Module):
    """Correct a frozen generic Drive using only signal-multiplied controls.

    No constant control feature enters the recurrent path, so zero audio with a
    zero state remains exactly zero for every knob setting.
    """

    def __init__(self, hidden_size: int = 12) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.recurrent = torch.nn.GRU(6, hidden_size, batch_first=True)
        self.output = torch.nn.Linear(hidden_size, 1)
        for name, parameter in self.recurrent.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(parameter)
                parameter.requires_grad_(False)
        torch.nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-4)
        torch.nn.init.zeros_(self.output.bias)
        self.output.bias.requires_grad_(False)

    def forward(
        self,
        dry: torch.Tensor,
        base: torch.Tensor,
        controls: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.jit.is_tracing():
            if dry.ndim != 2 or base.shape != dry.shape:
                raise ValueError("Drive adapter expects matching dry/base [batch,time]")
            if controls.shape != (dry.shape[0], 3):
                raise ValueError("Drive adapter expects normalized [batch,3] controls")
            if not torch.isfinite(dry).all() or not torch.isfinite(base).all():
                raise ValueError("Drive adapter input contains non-finite audio")
            if (
                not torch.isfinite(controls).all()
                or torch.any(controls < 0.0)
                or torch.any(controls > 1.0)
            ):
                raise ValueError("Drive adapter controls must be finite and normalized")
        centered = controls.mul(2.0).sub(1.0)
        difference = base - dry
        features = torch.stack(
            (
                dry,
                base,
                difference,
                dry * centered[:, 0:1],
                dry * centered[:, 1:2],
                base * centered[:, 2:3],
            ),
            dim=-1,
        )
        hidden, next_state = self.recurrent(features, state)
        rendered = base + self.output(hidden).squeeze(-1)
        if not torch.jit.is_tracing() and not torch.isfinite(rendered).all():
            raise ValueError("Drive adapter produced non-finite audio")
        return rendered, next_state


@dataclass(frozen=True)
class DriveAdapterState:
    base_state: tuple[torch.Tensor, torch.Tensor]
    adapter_state: torch.Tensor


class CalibratedDriveRenderer(torch.nn.Module):
    def __init__(self, base: DriveForwardRenderer, adapter: DriveDeviceAdapter) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter

    def forward(
        self,
        dry: torch.Tensor,
        controls: torch.Tensor,
        state: DriveAdapterState | None = None,
    ) -> tuple[torch.Tensor, DriveAdapterState]:
        base_state = None if state is None else state.base_state
        adapter_state = None if state is None else state.adapter_state
        base, next_base = self.base(dry, controls, base_state)
        rendered, next_adapter = self.adapter(dry, base, controls, adapter_state)
        return rendered, DriveAdapterState(next_base, next_adapter)


def load_drive_adapter(path: Path) -> DriveDeviceAdapter:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != 1 or payload.get("sample_rate") != FORWARD_RATE:
        raise ValueError("incompatible Drive device adapter checkpoint")
    adapter = DriveDeviceAdapter(int(payload["hidden_size"]))
    adapter.load_state_dict(payload["state_dict"], strict=True)
    adapter.eval()
    return adapter
