"""Strict offline inference for NAM A2 full submodels.

Only in-memory mono arrays are accepted.  This module never enumerates audio
devices, writes audio, normalizes, limits, dithers, or changes sample rate.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Callable

import numpy as np
import torch


RATE = 48_000
FIELD = 6_347


def full(document: dict) -> dict:
    """Return the highest-capacity WaveNet in a supported A2 container."""

    if document.get("architecture") != "SlimmableContainer":
        raise ValueError("NAM document is not an A2 SlimmableContainer")
    if float(document.get("sample_rate", 0.0)) != RATE:
        raise ValueError("NAM A2 sample rate differs")
    submodels = document.get("config", {}).get("submodels")
    if not isinstance(submodels, list) or len(submodels) < 2:
        raise ValueError("NAM A2 container has no full/lite submodels")
    candidates = [row.get("model") for row in submodels]
    if any(not isinstance(row, dict) or row.get("architecture") != "WaveNet" for row in candidates):
        raise ValueError("NAM A2 submodel architecture differs")
    selected = max(candidates, key=lambda row: len(row.get("weights", ())))
    if not selected.get("weights"):
        raise ValueError("NAM A2 full submodel has no weights")
    result = dict(selected)
    result["sample_rate"] = float(document["sample_rate"])
    return result


def loader(source: Path, deps: Path) -> Callable[[dict], torch.nn.Module]:
    """Load only the official NAM model implementation, without its GUI/audio entrypoints."""

    source, deps = Path(source).resolve(), Path(deps).resolve()
    if not (source / "nam" / "models" / "_from_nam.py").is_file():
        raise ValueError("official NAM source tree is missing")
    if str(deps) not in sys.path:
        sys.path.append(str(deps))
    existing = sys.modules.get("nam")
    expected = str((source / "nam").resolve())
    if existing is not None and expected not in [str(Path(item).resolve()) for item in getattr(existing, "__path__", ())]:
        raise ValueError("a different NAM implementation is already loaded")
    if existing is None:
        package = types.ModuleType("nam")
        package.__path__ = [expected]
        sys.modules["nam"] = package
    from nam.models import init_from_nam

    return init_from_nam


class A2Runtime:
    """Render the full-capacity member of one NAM A2 container on CPU."""

    def __init__(self, path: Path, init_from_nam: Callable[[dict], torch.nn.Module]):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        self.hash = hashlib.sha256(raw).hexdigest()
        document = json.loads(raw)
        self.metadata = document.get("metadata") or {}
        self.model = init_from_nam(full(document)).eval().cpu()
        if int(self.model.receptive_field) != FIELD:
            raise ValueError("NAM A2 full receptive field differs")

    def render(self, source: np.ndarray) -> np.ndarray:
        value = np.asarray(source, dtype=np.float32)
        if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
            raise ValueError("NAM A2 requires finite nonempty mono audio")
        before = hashlib.sha256(value.tobytes()).hexdigest()
        with torch.inference_mode():
            output = self.model(torch.from_numpy(value.copy()), pad_start=True)
        result = output.detach().cpu().numpy().astype(np.float32, copy=False)
        if result.shape != value.shape or not np.isfinite(result).all():
            raise ValueError("NAM A2 render violated finite geometry")
        if before != hashlib.sha256(value.tobytes()).hexdigest():
            raise ValueError("NAM A2 runtime mutated source audio")
        return result

    def assert_unchanged(self) -> None:
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.hash:
            raise ValueError("NAM A2 model artifact changed")
