"""Causal zero-input-safe residual adapter for a frozen stable effect."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


class StableEffectResidualAdapter(torch.nn.Module):
    def __init__(self, control_count: int, hidden_size: int = 12) -> None:
        super().__init__()
        if control_count <= 0 or hidden_size <= 0:
            raise ValueError("stable effect adapter dimensions must be positive")
        self.control_count = control_count
        self.hidden_size = hidden_size
        self.recurrent = torch.nn.GRU(
            3 + 2 * control_count,
            hidden_size,
            batch_first=True,
            bias=False,
        )
        self.output = torch.nn.Linear(hidden_size, 1, bias=False)
        torch.nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-4)

    def _condition(self, controls: torch.Tensor, frames: int) -> torch.Tensor:
        if controls.ndim == 2 and controls.shape[1] == self.control_count:
            return controls.unsqueeze(1).expand(-1, frames, -1)
        if controls.ndim == 3 and controls.shape[1:] == (frames, self.control_count):
            return controls
        raise ValueError("stable effect adapter controls have the wrong shape")

    def forward(
        self,
        dry: torch.Tensor,
        base: torch.Tensor,
        controls: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.jit.is_tracing():
            if dry.ndim != 2 or base.shape != dry.shape:
                raise ValueError("stable effect adapter expects matching [batch,time] audio")
            if not torch.isfinite(dry).all() or not torch.isfinite(base).all():
                raise ValueError("stable effect adapter audio must be finite")
            if (
                not torch.isfinite(controls).all()
                or torch.any(controls < 0.0)
                or torch.any(controls > 1.0)
            ):
                raise ValueError("stable effect adapter controls must be normalized")
        condition = self._condition(controls, dry.shape[1]).mul(2.0).sub(1.0)
        audio = torch.stack((dry, base, base - dry), dim=-1)
        features = torch.cat(
            (audio, dry.unsqueeze(-1) * condition, base.unsqueeze(-1) * condition),
            dim=-1,
        )
        hidden, next_state = self.recurrent(features, state)
        rendered = base + self.output(hidden).squeeze(-1)
        if not torch.jit.is_tracing() and not torch.isfinite(rendered).all():
            raise ValueError("stable effect adapter produced non-finite audio")
        return rendered, next_state


@dataclass(frozen=True)
class StableEffectAdapterState:
    base_state: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    adapter_state: torch.Tensor


class AdaptedStableEffectRenderer(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, adapter: StableEffectResidualAdapter) -> None:
        super().__init__()
        if base.control_count != adapter.control_count:
            raise ValueError("stable effect base and adapter control widths differ")
        self.base = base
        self.adapter = adapter
        self.control_count = adapter.control_count

    def forward(
        self,
        dry: torch.Tensor,
        controls: torch.Tensor,
        state: StableEffectAdapterState | None = None,
    ) -> tuple[torch.Tensor, StableEffectAdapterState]:
        base_state = None if state is None else state.base_state
        adapter_state = None if state is None else state.adapter_state
        base, next_base = self.base(dry, controls, base_state)
        rendered, next_adapter = self.adapter(
            dry, base, controls, adapter_state
        )
        return rendered, StableEffectAdapterState(next_base, next_adapter)


def load_stable_effect_adapter(path: Path) -> tuple[StableEffectResidualAdapter, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != 1 or payload.get("sample_rate") != 48_000:
        raise ValueError("incompatible stable effect adapter checkpoint")
    adapter = StableEffectResidualAdapter(
        int(payload["control_count"]), int(payload["hidden_size"])
    )
    adapter.load_state_dict(payload["state_dict"], strict=True)
    return adapter.eval(), payload
