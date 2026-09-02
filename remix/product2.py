"""Product-safe stateful restoration pairs with realized-source provenance."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .foundation_data import RATE
from .license_gate import require_product_weights
from .product_data import (
    MAX_DELAY_SECONDS,
    MAX_RELEASE_MS,
    RESEARCH_SOURCE_IDS,
    Clean,
    _delay,
    _read,
    discover_clean,
    nonlinear,
)


MECHANISMS = ("nonlinear", "dynamics", "echo")
FROZEN_PRODUCT2_SOURCE_IDS = {
    "dafx25-guitar-effects-chains",
    "egfxset",
    "guitarjam",
}
CONTROL_WIDTH = 5
TARGET_FRAMES_MINIMUM = 4096
HISTORY_FRAMES = {
    "nonlinear": 1024,
    "dynamics": 0,
    "echo": round(RATE * MAX_DELAY_SECONDS) + 1,
}


def _condition_clean(clean: np.ndarray, rng: random.Random) -> tuple[np.ndarray, float]:
    """Expose the renderers to useful but non-clipped product-level input ranges."""
    clean = np.asarray(clean, dtype=np.float64)
    rms = math.sqrt(float(np.mean(clean * clean)) + 1.0e-12)
    target_rms = math.exp(rng.uniform(math.log(0.035), math.log(0.16)))
    gain = target_rms / max(rms, 1.0e-6)
    peak = float(np.max(np.abs(clean)))
    if peak > 0.0:
        gain = min(gain, 0.90 / peak)
    conditioned = np.asarray(clean * gain, dtype=np.float32)
    return conditioned, 20.0 * math.log10(max(gain, 1.0e-8))


def _active_dynamics(clean: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict, np.ndarray]:
    """Repository-owned compressor with a threshold tied to the admitted program."""
    active = np.abs(clean[np.abs(clean) >= 1.0e-5])
    reference = float(np.quantile(active, 0.75)) if len(active) else 0.05
    reference_db = 20.0 * math.log10(max(reference, 1.0e-5))
    threshold_db = float(np.clip(reference_db - rng.uniform(4.0, 14.0), -36.0, -12.0))
    ratio = rng.uniform(2.0, 10.0)
    attack_ms = rng.uniform(0.5, 35.0)
    release_ms = rng.uniform(40.0, MAX_RELEASE_MS)
    makeup_db = rng.uniform(0.0, 8.0)
    attack = math.exp(-1.0 / (RATE * attack_ms / 1000.0))
    release = math.exp(-1.0 / (RATE * release_ms / 1000.0))
    envelope = 0.0
    wet = np.empty_like(clean, dtype=np.float64)
    inverse_log_gain = np.empty_like(clean, dtype=np.float64)
    for index, sample in enumerate(np.asarray(clean, dtype=np.float64)):
        level = abs(sample)
        coefficient = attack if level > envelope else release
        envelope = coefficient * envelope + (1.0 - coefficient) * level
        level_db = 20.0 * math.log10(max(envelope, 1.0e-7))
        reduction_db = max(0.0, level_db - threshold_db) * (1.0 - 1.0 / ratio)
        forward_log_gain = (makeup_db - reduction_db) * math.log(10.0) / 20.0
        wet[index] = sample * math.exp(forward_log_gain)
        inverse_log_gain[index] = -forward_log_gain
    return wet.astype(np.float32), {
        "threshold_db": threshold_db,
        "ratio": ratio,
        "attack_ms": attack_ms,
        "release_ms": release_ms,
        "makeup_db": makeup_db,
    }, inverse_log_gain.astype(np.float32)


def _controls(mechanism: str, values: dict) -> np.ndarray:
    result = np.zeros(CONTROL_WIDTH, dtype=np.float32)
    if mechanism == "nonlinear":
        shape = {"tanh": 0.0, "atan": 0.5, "cubic": 1.0}[values["shape"]]
        result[:] = (
            (values["drive"] - 1.8) / (8.0 - 1.8),
            (values["bias"] + 0.12) / 0.24,
            math.log(values["cutoff_hz"] / 1800.0) / math.log(11000.0 / 1800.0),
            (values["level"] - 0.45) / 0.40,
            shape,
        )
    elif mechanism == "dynamics":
        result[:] = (
            (values["threshold_db"] + 36.0) / 24.0,
            (values["ratio"] - 2.0) / 8.0,
            math.log(values["attack_ms"] / 0.5) / math.log(35.0 / 0.5),
            math.log(values["release_ms"] / 40.0) / math.log(MAX_RELEASE_MS / 40.0),
            values["makeup_db"] / 8.0,
        )
    elif mechanism == "echo":
        result[:3] = (
            (values["time_ms"] / 1000.0 - 0.04) / (MAX_DELAY_SECONDS - 0.04),
            (values["feedback"] - 0.1) / (0.72 - 0.1),
            (values["mix"] - 0.15) / (0.65 - 0.15),
        )
    else:
        raise ValueError(f"unsupported product2 mechanism: {mechanism}")
    if not np.isfinite(result).all() or np.any(result < -1.0e-6) or np.any(result > 1.0 + 1.0e-6):
        raise ValueError(f"invalid normalized controls for {mechanism}: {result}")
    return np.clip(result, 0.0, 1.0)


class ProductPairsV2(torch.utils.data.Dataset):
    """Balanced source sampling plus state warm-up before the supervised tail."""

    def __init__(
        self,
        workspace: Path,
        mechanism: str,
        split: str,
        samples: int,
        target_frames: int,
        seed: int,
    ) -> None:
        if mechanism not in MECHANISMS or split not in {"fit", "calibration", "development", "locked-final"}:
            raise ValueError(f"unsupported product2 selection: {mechanism}/{split}")
        if samples < 1 or target_frames < TARGET_FRAMES_MINIMUM:
            raise ValueError("product2 request is too small")
        self.workspace = workspace.resolve()
        self.mechanism = mechanism
        self.split = split
        self.samples = samples
        self.target_frames = target_frames
        self.history_frames = HISTORY_FRAMES[mechanism]
        self.total_frames = self.history_frames + target_frames
        self.seed = seed
        buckets: dict[str, list[Clean]] = defaultdict(list)
        for item in discover_clean(self.workspace):
            if item.split == split and item.source_id in FROZEN_PRODUCT2_SOURCE_IDS:
                buckets[item.source_id].append(item)
        if not buckets or any(not rows for rows in buckets.values()):
            raise ValueError(f"no product2 clean programs for {split}")
        self.buckets = {name: tuple(rows) for name, rows in sorted(buckets.items())}
        self.source_ids = tuple(self.buckets)
        self._cache: dict[int, dict] = {}
        selected_sources = set(self.realized_source_counts())
        selected_sources.add("muspector-dsp")
        if selected_sources & RESEARCH_SOURCE_IDS:
            raise PermissionError(f"research source entered product2: {selected_sources & RESEARCH_SOURCE_IDS}")
        self.authorization = require_product_weights(
            self.workspace / "remix/data_sources.json", sorted(selected_sources)
        )

    def __len__(self) -> int:
        return self.samples

    def _selection(self, index: int) -> Clean:
        source = self.source_ids[index % len(self.source_ids)]
        rows = self.buckets[source]
        cycle = index // len(self.source_ids)
        offset = (cycle * 104729 + self.seed) % len(rows)
        return rows[offset]

    def realized_source_counts(self) -> dict[str, int]:
        return dict(Counter(self._selection(index).source_id for index in range(self.samples)))

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.samples:
            raise IndexError(index)
        if index in self._cache:
            return self._cache[index]
        rng = random.Random(self.seed + index * 104729)
        selected = self._selection(index)
        clean = _read(selected, self.total_frames, rng.getrandbits(63))
        clean, input_gain_db = _condition_clean(clean, rng)
        if self.mechanism == "nonlinear":
            wet, controls = nonlinear(clean, rng)
            inverse_log_gain = np.zeros_like(clean)
        elif self.mechanism == "dynamics":
            wet, controls, inverse_log_gain = _active_dynamics(clean, rng)
        else:
            wet, controls = _delay(clean, rng)
            inverse_log_gain = np.zeros_like(clean)
        if wet.shape != clean.shape or not np.isfinite(wet).all():
            raise ValueError("product2 renderer changed geometry or emitted non-finite audio")
        if self.mechanism == "echo":
            delay_frames = round(controls["time_ms"] * RATE / 1000.0)
            target_wet = wet[self.history_frames :]
            target_clean = clean[self.history_frames :]
            scaled = (1.0 - controls["mix"]) * target_clean
            if delay_frames > self.history_frames or np.max(np.abs(target_wet - scaled)) < 1.0e-6:
                raise ValueError("echo pair lacks active delayed content in the supervised target")
        result = {
            "wet": torch.from_numpy(wet.copy()),
            "clean": torch.from_numpy(clean.copy()),
            "controls": torch.from_numpy(_controls(self.mechanism, controls)),
            "inverse_log_gain": torch.from_numpy(inverse_log_gain.copy()),
            "control_values": controls,
            "input_gain_db": input_gain_db,
            "target_start": self.history_frames,
            "source_id": selected.source_id,
            "group": selected.group,
        }
        self._cache[index] = result
        return result


def restore_echo(wet: np.ndarray, control_values: dict) -> np.ndarray:
    """Exact causal inverse of the repository-owned echo renderer."""
    wet = np.asarray(wet, dtype=np.float64)
    delay = round(float(control_values["time_ms"]) * RATE / 1000.0)
    feedback = float(control_values["feedback"])
    mix = float(control_values["mix"])
    if wet.ndim != 1 or delay < 1 or not 0.0 <= feedback < 1.0 or not 0.0 < mix < 1.0:
        raise ValueError("invalid echo inverse input")
    clean = np.empty_like(wet)
    echo = np.zeros_like(wet)
    denominator = 1.0 - mix
    for index in range(len(wet)):
        if index >= delay:
            echo[index] = clean[index - delay] + feedback * echo[index - delay]
        clean[index] = (wet[index] - mix * echo[index]) / denominator
    if not np.isfinite(clean).all():
        raise ValueError("echo inverse emitted non-finite audio")
    return clean.astype(np.float32)


def restore_dynamics(
    wet: np.ndarray,
    control_values: dict,
    initial_envelope: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Invert one compressor effect while carrying only its own envelope state."""
    wet = np.asarray(wet, dtype=np.float64)
    if wet.ndim != 1 or not np.isfinite(wet).all() or initial_envelope < 0.0:
        raise ValueError("invalid dynamics inverse input")
    threshold_db = float(control_values["threshold_db"])
    ratio = float(control_values["ratio"])
    attack_ms = float(control_values["attack_ms"])
    release_ms = float(control_values["release_ms"])
    makeup_db = float(control_values["makeup_db"])
    if not -36.0 <= threshold_db <= -12.0 or not 2.0 <= ratio <= 10.0:
        raise ValueError("dynamics controls are outside the admitted domain")
    attack = math.exp(-1.0 / (RATE * attack_ms / 1000.0))
    release = math.exp(-1.0 / (RATE * release_ms / 1000.0))
    compression = 1.0 - 1.0 / ratio
    envelope = float(initial_envelope)
    clean = np.empty_like(wet)
    for index, signed_wet in enumerate(wet):
        target = abs(signed_wet)
        low = 0.0
        high = max(1.0, target * 8.0)
        for _ in range(20):
            level = (low + high) * 0.5
            coefficient = attack if level > envelope else release
            candidate_envelope = coefficient * envelope + (1.0 - coefficient) * level
            level_db = 20.0 * math.log10(max(candidate_envelope, 1.0e-7))
            reduction_db = max(0.0, level_db - threshold_db) * compression
            predicted = level * 10.0 ** ((makeup_db - reduction_db) / 20.0)
            if predicted < target:
                low = level
            else:
                high = level
        level = (low + high) * 0.5
        coefficient = attack if level > envelope else release
        envelope = coefficient * envelope + (1.0 - coefficient) * level
        clean[index] = math.copysign(level, signed_wet)
    if not np.isfinite(clean).all():
        raise ValueError("dynamics inverse emitted non-finite audio")
    return clean.astype(np.float32), envelope
