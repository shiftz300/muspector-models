"""Zero-centered stable effect runtime for safe recurrent fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .stable_effect import StableEffectRenderer


@dataclass(frozen=True)
class CenteredStableState:
    signal: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    zero: tuple[tuple[torch.Tensor, torch.Tensor], ...]


class CenteredStableEffectRenderer(torch.nn.Module):
    """Render F(audio, controls) - F(zero, controls) with shared weights."""

    def __init__(self, base: StableEffectRenderer) -> None:
        super().__init__()
        self.base = base
        self.control_count = base.control_count
        self.hidden_size = base.hidden_size
        self.layers = base.layers

    def forward(
        self,
        dry: torch.Tensor,
        controls: torch.Tensor,
        state: CenteredStableState | None = None,
    ) -> tuple[torch.Tensor, CenteredStableState]:
        signal_state = None if state is None else state.signal
        zero_state = None if state is None else state.zero
        signal, next_signal = self.base(dry, controls, signal_state)
        zero, next_zero = self.base(torch.zeros_like(dry), controls, zero_state)
        return signal - zero, CenteredStableState(next_signal, next_zero)

    @torch.no_grad()
    def project_candidate_recurrence(self, maximum: float = 0.995) -> list[float]:
        if not 0.0 < maximum < 1.0:
            raise ValueError("stable recurrence maximum must be between zero and one")
        norms = []
        hidden = self.hidden_size
        for layer in self.base.rnn_layers:
            candidate = layer.weight_hh_l0[2 * hidden : 3 * hidden]
            norm = torch.linalg.matrix_norm(candidate, ord=float("inf"))
            if float(norm) > maximum:
                candidate.mul_(maximum / float(norm))
            norms.append(
                float(
                    torch.linalg.matrix_norm(candidate, ord=float("inf")).detach()
                )
            )
        return norms


def detach_centered_state(state: CenteredStableState) -> CenteredStableState:
    def detached(layers):
        return tuple((hidden.detach(), cell.detach()) for hidden, cell in layers)

    return CenteredStableState(detached(state.signal), detached(state.zero))


def load_centered_stable_effect(path: Path) -> tuple[CenteredStableEffectRenderer, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != 1
        or payload.get("sample_rate") != 48_000
        or payload.get("architecture") != "zero-centered-stable-conditioned-lstm"
    ):
        raise ValueError("incompatible zero-centered stable effect checkpoint")
    base = StableEffectRenderer(
        control_count=int(payload["control_count"]),
        hidden_size=int(payload["hidden_size"]),
        layers=int(payload["layers"]),
        input_coef=float(payload["input_coef"]),
        inverted_controls=tuple(int(value) for value in payload["inverted_controls"]),
    )
    base.load_state_dict(payload["state_dict"], strict=True)
    return CenteredStableEffectRenderer(base).eval(), payload
