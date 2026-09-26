"""Product-safe Chorus pairs with a hidden LFO start phase."""

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
TARGET_FRAMES_MINIMUM = 2 * RATE
RATE_MIN, RATE_MAX = 1.0, 4.5
BASE_MIN_MS, BASE_MAX_MS = 12.0, 26.0
DEPTH_MIN_MS, DEPTH_MAX_MS = 2.0, 7.0
MIX_MIN, MIX_MAX = 0.22, 0.45


def chorus(clean: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict, np.ndarray]:
    rate_hz = math.exp(rng.uniform(math.log(RATE_MIN), math.log(RATE_MAX)))
    base_ms = rng.uniform(BASE_MIN_MS, BASE_MAX_MS)
    depth_ms = rng.uniform(DEPTH_MIN_MS, min(DEPTH_MAX_MS, base_ms - 5.0))
    mix = rng.uniform(MIX_MIN, MIX_MAX)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    time = np.arange(len(clean), dtype=np.float64) / RATE
    lfo = np.sin(2.0 * math.pi * rate_hz * time + phase)
    delay = (base_ms + depth_ms * lfo) * RATE / 1000.0
    positions = np.arange(len(clean), dtype=np.float64) - delay
    shifted = np.interp(positions, np.arange(len(clean)), clean, left=0.0, right=0.0)
    wet = (1.0 - mix) * clean.astype(np.float64) + mix * shifted
    return wet.astype(np.float32), {
        "family": "chorus", "rate_hz": rate_hz, "base_ms": base_ms,
        "depth_ms": depth_ms, "mix": mix, "hidden_phase_radians": phase,
        "renderer": "muspector-fractional-delay-chorus-v1",
    }, lfo.astype(np.float32)


def normalized_controls(values: dict) -> np.ndarray:
    result = np.zeros(CONTROL_WIDTH, dtype=np.float32)
    result[:4] = (
        math.log(values["rate_hz"] / RATE_MIN) / math.log(RATE_MAX / RATE_MIN),
        (values["base_ms"] - BASE_MIN_MS) / (BASE_MAX_MS - BASE_MIN_MS),
        (values["depth_ms"] - DEPTH_MIN_MS) / (DEPTH_MAX_MS - DEPTH_MIN_MS),
        (values["mix"] - MIX_MIN) / (MIX_MAX - MIX_MIN),
    )
    return np.clip(result, 0.0, 1.0)


def restore_chorus(wet: np.ndarray, values: dict, lfo: np.ndarray) -> np.ndarray:
    wet = np.asarray(wet, dtype=np.float64)
    lfo = np.asarray(lfo, dtype=np.float64)
    if wet.shape != lfo.shape or wet.ndim != 1:
        raise ValueError("Chorus inverse geometry mismatch")
    delay = (float(values["base_ms"]) + float(values["depth_ms"]) * lfo) * RATE / 1000.0
    mix = float(values["mix"])
    clean = np.zeros_like(wet)
    for index in range(len(wet)):
        position = index - delay[index]
        if position >= 0.0:
            lower = int(position)
            upper = min(lower + 1, index - 1)
            fraction = position - lower
            shifted = clean[lower] * (1.0 - fraction) + clean[upper] * fraction
        else:
            shifted = 0.0
        clean[index] = (wet[index] - mix * shifted) / (1.0 - mix)
    return clean.astype(np.float32)


class ChorusPairsV3(torch.utils.data.Dataset):
    def __init__(self, workspace: Path, split: str, samples: int, target_frames: int, seed: int) -> None:
        if split not in {"fit", "calibration", "development", "locked-final"} or target_frames < TARGET_FRAMES_MINIMUM:
            raise ValueError("invalid Chorus pair request")
        self.workspace, self.split, self.samples, self.target_frames, self.seed = workspace.resolve(), split, samples, target_frames, seed
        self.context_frames = CONTEXT_FRAMES
        self.total_frames = target_frames + 2 * CONTEXT_FRAMES
        buckets: dict[str, list[Clean]] = defaultdict(list)
        for item in discover_clean(self.workspace):
            if item.split == split and item.source_id in FROZEN_PRODUCT2_SOURCE_IDS:
                buckets[item.source_id].append(item)
        if not buckets:
            raise ValueError(f"no product-safe Chorus Clean for {split}")
        self.buckets = {name: tuple(rows) for name, rows in sorted(buckets.items())}
        self.source_ids = tuple(self.buckets)
        sources = set(self.realized_source_counts()) | {"muspector-dsp"}
        if sources & RESEARCH_SOURCE_IDS:
            raise PermissionError("research source entered Chorus pairs")
        requirements = {
            source: "product-clean-source" for source in sources if source != "muspector-dsp"
        }
        requirements["muspector-dsp"] = ("product-pair-generation", "train-restoration")
        self.authorization = require_product_uses(
            self.workspace / "remix/data_sources.json", requirements
        )
        self._cache = {}

    def __len__(self): return self.samples
    def _selection(self, index):
        source = self.source_ids[index % len(self.source_ids)]; rows = self.buckets[source]
        return rows[((index // len(self.source_ids)) * 104729 + self.seed) % len(rows)]
    def realized_source_counts(self): return dict(Counter(self._selection(i).source_id for i in range(self.samples)))
    def __getitem__(self, index):
        if index in self._cache: return self._cache[index]
        rng = random.Random(self.seed + index * 104729); selected = self._selection(index)
        clean = _read(selected, self.total_frames, rng.getrandbits(63)); clean, gain_db = _condition_clean(clean, rng)
        wet, values, lfo = chorus(clean, rng)
        row = {"wet":torch.from_numpy(wet.copy()), "clean":torch.from_numpy(clean.copy()),
               "controls":torch.from_numpy(normalized_controls(values)), "lfo":torch.from_numpy(lfo.copy()),
               "control_values":values, "input_gain_db":gain_db, "target_start":CONTEXT_FRAMES,
               "target_end":CONTEXT_FRAMES+self.target_frames, "source_id":selected.source_id, "group":selected.group}
        self._cache[index]=row; return row
