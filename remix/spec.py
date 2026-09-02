"""Versioned effect-chain contract with physical, user-facing controls."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


SCHEMA_VERSION = 1
KINDS = ("drive", "delay", "reverb")
ORDER_PAIRS = (("drive", "delay"), ("drive", "reverb"), ("delay", "reverb"))


def _bounded(name: str, value: float, minimum: float, maximum: float) -> None:
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be within [{minimum}, {maximum}], got {value}")


@dataclass(frozen=True)
class Drive:
    gain_db: float
    tone: float
    level_db: float
    kind: str = "drive"

    def validate(self) -> None:
        _bounded("gain_db", self.gain_db, 0.0, 30.0)
        _bounded("tone", self.tone, 0.0, 1.0)
        _bounded("level_db", self.level_db, -18.0, 12.0)


@dataclass(frozen=True)
class Delay:
    time_ms: float
    feedback: float
    mix: float
    kind: str = "delay"

    def validate(self) -> None:
        _bounded("time_ms", self.time_ms, 40.0, 1_000.0)
        _bounded("feedback", self.feedback, 0.0, 0.9)
        _bounded("mix", self.mix, 0.0, 0.7)


@dataclass(frozen=True)
class Reverb:
    decay_s: float
    damping: float
    mix: float
    kind: str = "reverb"

    def validate(self) -> None:
        _bounded("decay_s", self.decay_s, 0.2, 8.0)
        _bounded("damping", self.damping, 0.0, 1.0)
        _bounded("mix", self.mix, 0.0, 0.7)


Effect = Drive | Delay | Reverb
CONTROL_NAMES = (
    "drive.gain_db",
    "drive.tone",
    "drive.level_db",
    "delay.time_ms",
    "delay.feedback",
    "delay.mix",
    "reverb.decay_s",
    "reverb.damping",
    "reverb.mix",
)


@dataclass(frozen=True)
class ChainSpec:
    effects: tuple[Effect, ...]
    schema: int = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema != SCHEMA_VERSION:
            raise ValueError(f"unsupported chain schema: {self.schema}")
        kinds = [effect.kind for effect in self.effects]
        if len(kinds) != len(set(kinds)):
            raise ValueError("an effect family may appear only once")
        for effect in self.effects:
            effect.validate()

    def document(self) -> dict:
        self.validate()
        return {
            "schema": self.schema,
            "effects": [asdict(effect) for effect in self.effects],
        }

    @property
    def topology(self) -> tuple[str, ...]:
        return tuple(effect.kind for effect in self.effects)


def order_targets(order: Sequence[str]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return pairwise precedence targets and masks in ``ORDER_PAIRS`` order."""

    position = {kind: index for index, kind in enumerate(order)}
    targets, masks = [], []
    for left, right in ORDER_PAIRS:
        present = left in position and right in position
        masks.append(float(present))
        targets.append(float(present and position[left] < position[right]))
    return tuple(targets), tuple(masks)


def identifiable_order_targets(
    order: Sequence[str],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Mask relations that the canonical renderer cannot uniquely identify.

    Delay and Reverb are LTI in both reference renderers. They commute when
    adjacent, so assigning either order as uniquely correct would train on
    contradictory audio. A Drive between them breaks that equivalence.
    """

    targets, masks = order_targets(order)
    position = {kind: index for index, kind in enumerate(order)}
    masks = list(masks)
    delay_reverb = ORDER_PAIRS.index(("delay", "reverb"))
    if masks[delay_reverb]:
        drive = position.get("drive")
        low, high = sorted((position["delay"], position["reverb"]))
        if drive is None or not low < drive < high:
            masks[delay_reverb] = 0.0
    return targets, tuple(masks)


def control_targets(spec: ChainSpec) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Encode physical controls into stable normalized training targets."""

    values = [0.0] * len(CONTROL_NAMES)
    masks = [0.0] * len(CONTROL_NAMES)
    for effect in spec.effects:
        if isinstance(effect, Drive):
            values[0:3] = [effect.gain_db / 30.0, effect.tone, (effect.level_db + 18.0) / 30.0]
            masks[0:3] = [1.0] * 3
        elif isinstance(effect, Delay):
            values[3:6] = [
                math.log(effect.time_ms / 40.0) / math.log(1_000.0 / 40.0),
                effect.feedback / 0.9,
                effect.mix / 0.7,
            ]
            masks[3:6] = [1.0] * 3
        elif isinstance(effect, Reverb):
            values[6:9] = [
                math.log(effect.decay_s / 0.2) / math.log(8.0 / 0.2),
                effect.damping,
                effect.mix / 0.7,
            ]
            masks[6:9] = [1.0] * 3
    return tuple(values), tuple(masks)


def ranked_topologies(active: Iterable[str], logits: Sequence[float]) -> list[tuple[tuple[str, ...], float]]:
    """Rank every compatible permutation using pairwise log probabilities."""

    active = tuple(dict.fromkeys(active))
    if any(kind not in KINDS for kind in active):
        raise ValueError(f"unknown active topology: {active}")
    if len(logits) != len(ORDER_PAIRS):
        raise ValueError(f"expected {len(ORDER_PAIRS)} order logits")
    result = []
    for order in itertools.permutations(active):
        targets, mask = order_targets(order)
        score = 0.0
        for target, enabled, logit in zip(targets, mask, logits):
            if enabled:
                signed = logit if target else -logit
                score += -max(0.0, -signed) - math.log1p(math.exp(-abs(signed)))
        result.append((order, score))
    return sorted(result, key=lambda item: item[1], reverse=True)
