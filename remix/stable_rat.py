"""Independent runtime for constrained ASRNN-style RAT checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch


class StableRatRenderer(torch.nn.Module):
    """Deep conditioned LSTM runtime with explicit recurrent state.

    The runtime contains only standard PyTorch operations. Stability and
    zero-input behavior are properties of admitted checkpoint weights and are
    re-audited when a checkpoint is imported.
    """

    def __init__(self, hidden_size: int = 8, layers: int = 4, input_coef: float = 21.4) -> None:
        super().__init__()
        if hidden_size <= 0 or layers <= 0 or input_coef <= 0.0:
            raise ValueError("stable RAT runtime dimensions and input coefficient must be positive")
        self.hidden_size = hidden_size
        self.layers = layers
        self.input_coef = float(input_coef)
        recurrent = [torch.nn.LSTM(4, hidden_size, batch_first=True)]
        recurrent.extend(
            torch.nn.LSTM(hidden_size + 3, hidden_size, batch_first=True)
            for _ in range(1, layers)
        )
        self.rnn_layers = torch.nn.ModuleList(recurrent)
        self.output_layer = torch.nn.Linear(hidden_size, 1, bias=False)

    @staticmethod
    def _official_controls(controls: torch.Tensor, frames: int) -> torch.Tensor:
        if controls.ndim == 2:
            if controls.shape[1] != 3:
                raise ValueError("stable RAT controls must have three values")
            controls = controls.unsqueeze(1).expand(-1, frames, -1)
        elif controls.ndim != 3 or controls.shape[1:] != (frames, 3):
            raise ValueError("stable RAT controls must be [batch,3] or [batch,time,3]")
        # Public checkpoint order is Distortion, raw clockwise high-cut Filter,
        # Volume. Muspector exposes increasing-brightness Tone.
        physical = controls.clone()
        physical[..., 1] = 1.0 - physical[..., 1]
        return physical.mul(2.0).sub(1.0)

    def encode(
        self,
        dry: torch.Tensor,
        controls: torch.Tensor,
        state: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        if not torch.jit.is_tracing():
            if dry.ndim != 2:
                raise ValueError("stable RAT renderer expects dry [batch,time]")
            if controls.shape[0] != dry.shape[0]:
                raise ValueError("stable RAT control batch differs from audio")
            if not torch.isfinite(dry).all() or not torch.isfinite(controls).all():
                raise ValueError("stable RAT inputs must be finite")
            if torch.any(controls < 0.0) or torch.any(controls > 1.0):
                raise ValueError("stable RAT controls must be normalized")
            if state is not None and len(state) != self.layers:
                raise ValueError("stable RAT recurrent state has the wrong layer count")
        condition = self._official_controls(controls, dry.shape[1])
        value = dry.unsqueeze(-1) * self.input_coef
        previous = (None,) * self.layers if state is None else state
        next_states = []
        for layer, layer_state in zip(self.rnn_layers, previous):
            value, next_state = layer(torch.cat((value, condition), dim=2), layer_state)
            next_states.append(next_state)
        return value, tuple(next_states)

    def forward(
        self,
        dry: torch.Tensor,
        controls: torch.Tensor,
        state: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        value, next_states = self.encode(dry, controls, state)
        rendered = self.output_layer(value).squeeze(-1)
        if not torch.jit.is_tracing() and not torch.isfinite(rendered).all():
            raise ValueError("stable RAT renderer produced non-finite audio")
        return rendered, next_states


def load_stable_rat(path: Path) -> StableRatRenderer:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != 1
        or payload.get("sample_rate") != 48_000
        or payload.get("architecture") != "stable-conditioned-lstm"
    ):
        raise ValueError("incompatible stable RAT checkpoint")
    model = StableRatRenderer(
        hidden_size=int(payload["hidden_size"]),
        layers=int(payload["layers"]),
        input_coef=float(payload["input_coef"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()
