"""Product-safe Tremolo pairs with an unobserved LFO start phase.

The expert receives only the current Wet signal and this effect's own public
controls.  Forward order, neighbouring effects, Clean audio and the hidden LFO
phase are never inference inputs.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .foundation_data import RATE
from .license_gate import require_product_uses
from .product2 import FROZEN_PRODUCT2_SOURCE_IDS, _condition_clean
from .product_data import RESEARCH_SOURCE_IDS, Clean, _read, discover_clean


CONTROL_WIDTH = 5
CONTEXT_FRAMES = RATE
TARGET_FRAMES_MINIMUM = RATE * 2
RATE_MIN = 1.0
RATE_MAX = 10.0
DEPTH_MIN = 0.35
DEPTH_MAX = 0.82


def _lfo(time: np.ndarray, rate_hz: float, phase: float, waveform: str) -> np.ndarray:
    cycles = rate_hz * time + phase / (2.0 * math.pi)
    if waveform == "sine":
        return 0.5 + 0.5 * np.sin(2.0 * math.pi * cycles)
    if waveform == "triangle":
        fraction = np.mod(cycles, 1.0)
        return 1.0 - 2.0 * np.abs(fraction - 0.5)
    raise ValueError(f"unsupported Tremolo waveform: {waveform}")


def tremolo(clean: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict, np.ndarray]:
    rate_hz = math.exp(rng.uniform(math.log(RATE_MIN), math.log(RATE_MAX)))
    depth = rng.uniform(DEPTH_MIN, DEPTH_MAX)
    waveform = ("sine", "triangle")[rng.randrange(2)]
    hidden_phase = rng.uniform(0.0, 2.0 * math.pi)
    time = np.arange(len(clean), dtype=np.float64) / RATE
    gain = 1.0 - depth * (1.0 - _lfo(time, rate_hz, hidden_phase, waveform))
    wet = clean.astype(np.float64) * gain
    return wet.astype(np.float32), {
        "family": "tremolo",
        "rate_hz": rate_hz,
        "depth": depth,
        "waveform": waveform,
        "hidden_phase_radians": hidden_phase,
        "renderer": "muspector-tremolo-v1",
    }, np.asarray(-np.log(gain), dtype=np.float32)


def normalized_controls(values: dict) -> np.ndarray:
    result = np.zeros(CONTROL_WIDTH, dtype=np.float32)
    result[:3] = (
        math.log(values["rate_hz"] / RATE_MIN) / math.log(RATE_MAX / RATE_MIN),
        (values["depth"] - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN),
        0.0 if values["waveform"] == "sine" else 1.0,
    )
    if not np.isfinite(result).all() or np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError("invalid normalized Tremolo controls")
    return result


class TremoloPairsV3(torch.utils.data.Dataset):
    def __init__(self, workspace: Path, split: str, samples: int, target_frames: int, seed: int) -> None:
        if split not in {"fit", "calibration", "development", "locked-final"}:
            raise ValueError(f"unsupported Tremolo split: {split}")
        if samples < 1 or target_frames < TARGET_FRAMES_MINIMUM:
            raise ValueError("Tremolo pair request is too small")
        self.workspace = workspace.resolve()
        self.split = split
        self.samples = samples
        self.target_frames = target_frames
        self.context_frames = CONTEXT_FRAMES
        self.total_frames = target_frames + 2 * CONTEXT_FRAMES
        self.seed = seed
        buckets: dict[str, list[Clean]] = defaultdict(list)
        for item in discover_clean(self.workspace):
            if item.split == split and item.source_id in FROZEN_PRODUCT2_SOURCE_IDS:
                buckets[item.source_id].append(item)
        if not buckets:
            raise ValueError(f"no product-safe Tremolo Clean for {split}")
        self.buckets = {name: tuple(rows) for name, rows in sorted(buckets.items())}
        self.source_ids = tuple(self.buckets)
        selected_sources = set(self.realized_source_counts()) | {"muspector-dsp"}
        if selected_sources & RESEARCH_SOURCE_IDS:
            raise PermissionError("research source entered Tremolo product pairs")
        requirements = {
            source: "product-clean-source"
            for source in selected_sources
            if source != "muspector-dsp"
        }
        requirements["muspector-dsp"] = ("product-pair-generation", "train-restoration")
        self.authorization = require_product_uses(
            self.workspace / "remix/data_sources.json", requirements
        )
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return self.samples

    def _selection(self, index: int) -> Clean:
        source = self.source_ids[index % len(self.source_ids)]
        rows = self.buckets[source]
        cycle = index // len(self.source_ids)
        return rows[(cycle * 104729 + self.seed) % len(rows)]

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
        wet, values, inverse_log_gain = tremolo(clean, rng)
        result = {
            "wet": torch.from_numpy(wet.copy()),
            "clean": torch.from_numpy(clean.copy()),
            "controls": torch.from_numpy(normalized_controls(values)),
            "inverse_log_gain": torch.from_numpy(inverse_log_gain.copy()),
            "control_values": values,
            "input_gain_db": input_gain_db,
            "target_start": CONTEXT_FRAMES,
            "target_end": CONTEXT_FRAMES + self.target_frames,
            "source_id": selected.source_id,
            "group": selected.group,
        }
        self._cache[index] = result
        return result
