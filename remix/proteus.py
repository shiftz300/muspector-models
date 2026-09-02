"""Exact, offline inference for GuitarML Proteus SimpleRNN snapshots.

The runtime accepts in-memory mono arrays only.  It never enumerates or opens
audio devices, never normalizes or limits samples, and never writes audio.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


RATE = 44_100


class ProteusRuntime:
    """Run the LSTM -> Dense + input skip architecture used by Proteus."""

    def __init__(self, document: dict, *, source_hash: str | None = None):
        model = document.get("model_data", {})
        expected = {
            "model": "SimpleRNN",
            "input_size": 1,
            "skip": 1,
            "output_size": 1,
            "unit_type": "LSTM",
            "num_layers": 1,
            "hidden_size": 40,
            "bias_fl": True,
        }
        if model != expected:
            raise ValueError(f"unsupported Proteus model contract: {model}")
        state = document.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError("Proteus state_dict is missing")
        self.rec = torch.nn.LSTM(1, 40, 1)
        self.lin = torch.nn.Linear(40, 1, bias=True)
        tensors = {name: torch.as_tensor(value, dtype=torch.float32) for name, value in state.items()}
        self.rec.load_state_dict({name.removeprefix("rec."): value for name, value in tensors.items() if name.startswith("rec.")})
        self.lin.load_state_dict({name.removeprefix("lin."): value for name, value in tensors.items() if name.startswith("lin.")})
        self.rec.eval()
        self.lin.eval()
        self.source_hash = source_hash

    @classmethod
    def bytes(cls, value: bytes) -> "ProteusRuntime":
        return cls(json.loads(value), source_hash=hashlib.sha256(value).hexdigest())

    @classmethod
    def file(cls, path: Path) -> "ProteusRuntime":
        return cls.bytes(Path(path).read_bytes())

    def render(self, source: np.ndarray, *, chunk: int = 16_384) -> np.ndarray:
        value = np.asarray(source, dtype=np.float32)
        if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
            raise ValueError("Proteus requires finite nonempty mono audio")
        if chunk <= 0:
            raise ValueError("chunk size must be positive")
        before = hashlib.sha256(value.tobytes()).hexdigest()
        hidden = None
        rendered = []
        with torch.inference_mode():
            for start in range(0, len(value), chunk):
                dry = torch.from_numpy(value[start : start + chunk].copy()).reshape(-1, 1, 1)
                state, hidden = self.rec(dry, hidden)
                wet = self.lin(state) + dry
                rendered.append(wet[:, 0, 0].numpy().copy())
        result = np.concatenate(rendered).astype(np.float32, copy=False)
        if result.shape != value.shape or not np.isfinite(result).all():
            raise ValueError("Proteus render violated finite geometry")
        if before != hashlib.sha256(value.tobytes()).hexdigest():
            raise ValueError("Proteus runtime mutated source audio")
        return result

