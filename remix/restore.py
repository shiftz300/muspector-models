"""Order-free registry and reverse executor for independent inverse stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol

import torch

from .chain_report import load_model
from .net import SpectralNet


KINDS = ("drive", "reverb")
FAMILIES = (
    "drive", "distortion", "fuzz", "compressor", "gate", "eq", "filter", "wah",
    "chorus", "flanger", "phaser", "tremolo", "vibrato", "delay", "reverb",
    "pitch", "octave",
)


def plan(forward: tuple[str, ...]) -> tuple[str, ...]:
    if not forward or len(forward) != len(set(forward)) or not set(forward) <= set(KINDS):
        raise ValueError(f"unsupported or repeated stage order: {forward}")
    return tuple(reversed(forward))


@dataclass(frozen=True)
class Result:
    audio: torch.Tensor
    forward: tuple[str, ...]
    restore: tuple[str, ...]


@dataclass(frozen=True)
class Profile:
    """Explicit clean target; original is the non-converting default."""

    id: str = "original"
    mode: str = "original"
    reference: torch.Tensor | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.id or self.mode not in {"original", "reference", "preset"}:
            raise ValueError("invalid clean profile")
        if self.mode == "reference" and self.reference is None:
            raise ValueError("reference clean profile requires audio")
        if self.reference is not None and (self.reference.ndim != 2 or not torch.isfinite(self.reference).all()):
            raise ValueError("clean profile reference must be finite [batch,time]")


@dataclass(frozen=True)
class Stage:
    """One effect instance; slot identity permits repeated effect families."""

    slot: str
    family: str
    package: str
    device: str | None = None
    controls: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.slot or not self.package or self.family not in FAMILIES:
            raise ValueError("invalid restoration stage")
        finite = all(
            isinstance(name, str) and bool(name) and torch.isfinite(torch.tensor(float(value)))
            for name, value in self.controls.items()
        )
        if not finite:
            raise ValueError("stage controls must be named finite values")


class StageRuntime(Protocol):
    def restore(self, audio: torch.Tensor, stage: Stage, profile: Profile) -> torch.Tensor: ...


@dataclass(frozen=True)
class GraphResult:
    audio: torch.Tensor
    forward: tuple[Stage, ...]
    restore: tuple[Stage, ...]


class Graph:
    """Family-neutral reverse executor for independently packaged stages."""

    def __init__(self, runtimes: Mapping[str, StageRuntime]) -> None:
        if not runtimes or any(not name for name in runtimes):
            raise ValueError("restoration graph requires named runtimes")
        self.runtimes = dict(runtimes)

    @torch.inference_mode()
    def run(self, wet: torch.Tensor, forward: tuple[Stage, ...], profile: Profile | None = None) -> GraphResult:
        if wet.ndim != 2 or not torch.isfinite(wet).all():
            raise ValueError("wet audio must be finite [batch,time]")
        if not forward or len({stage.slot for stage in forward}) != len(forward):
            raise ValueError("forward graph requires unique nonempty stage slots")
        missing = {stage.package for stage in forward} - set(self.runtimes)
        if missing:
            raise ValueError(f"missing restoration runtimes: {sorted(missing)}")
        selected = Profile() if profile is None else profile
        order = tuple(reversed(forward)); value = wet
        for stage in order:
            value = self.runtimes[stage.package].restore(value, stage, selected)
            if value.shape != wet.shape or not torch.isfinite(value).all():
                raise ValueError(f"stage {stage.slot} violated audio geometry or finiteness")
        return GraphResult(value, forward, order)


class Bank:
    """Own stage models by identity; execution order belongs to each request."""

    def __init__(self, models: Mapping[str, SpectralNet]) -> None:
        if set(models) != set(KINDS):
            raise ValueError(f"stage bank must contain exactly {KINDS}")
        self.models = dict(models)

    @classmethod
    def load(cls, root: Path, device: torch.device | str = "cpu") -> "Bank":
        target = torch.device(device)
        return cls({kind: load_model(root / f"{kind}.pt").to(target).eval() for kind in KINDS})

    @torch.inference_mode()
    def run(self, wet: torch.Tensor, forward: tuple[str, ...], strengths: Mapping[str, float] | None = None) -> Result:
        if wet.ndim != 2 or not torch.isfinite(wet).all():
            raise ValueError("wet audio must be finite [batch,time]")
        restore = plan(forward); value = wet
        values = {} if strengths is None else dict(strengths)
        for kind in restore:
            strength = float(values.get(kind, 1.0))
            value = self.models[kind](value, strength=strength)
        return Result(value, forward, restore)
