"""Product-safe, split-disjoint pairs for an independent generic EQ expert.

The forward effect is repository-owned offline DSP.  It is intentionally a
generic three-band equalizer, not a claim about any named physical device.
Pairs are generated in memory from admitted Clean programs and are never
written as a training corpus.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .foundation_data import RATE
from .license_gate import require_product_weights
from .product2 import CONTROL_WIDTH, FROZEN_PRODUCT2_SOURCE_IDS, _condition_clean
from .product_data import RESEARCH_SOURCE_IDS, Clean, _read, discover_clean


MECHANISM = "spectral"
GAIN_DB = 8.0
MAXIMUM_CURVE_DB = 12.0
CONTEXT_FRAMES = 2048
TARGET_FRAMES_MINIMUM = 4096


def _curves_numpy(frequency: np.ndarray, values: dict) -> np.ndarray:
    safe = np.maximum(frequency, 20.0)
    logf = np.log2(safe)
    low = 1.0 / (1.0 + np.exp((logf - math.log2(320.0)) * 5.0))
    high = 1.0 / (1.0 + np.exp((math.log2(3600.0) - logf) * 5.0))
    width = 1.20 / math.sqrt(float(values["mid_q"]))
    mid = np.exp(-0.5 * ((logf - math.log2(float(values["mid_hz"]))) / width) ** 2)
    curve = values["low_gain_db"] * low + values["mid_gain_db"] * mid + values["high_gain_db"] * high
    maximum = max(float(np.max(np.abs(curve))), 1.0e-8)
    return curve * min(1.0, MAXIMUM_CURVE_DB / maximum)


def spectral_eq(clean: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """Apply one bounded zero-phase EQ to a complete context window."""
    gains = [rng.uniform(-GAIN_DB, GAIN_DB) for _ in range(3)]
    if sum(abs(value) for value in gains) < 8.0:
        dominant = rng.randrange(3)
        gains[dominant] = math.copysign(rng.uniform(6.0, GAIN_DB), gains[dominant] or 1.0)
    values = {
        "low_gain_db": gains[0],
        "mid_gain_db": gains[1],
        "mid_hz": math.exp(rng.uniform(math.log(250.0), math.log(5200.0))),
        "mid_q": math.exp(rng.uniform(math.log(0.45), math.log(3.5))),
        "high_gain_db": gains[2],
        "renderer": "muspector-zero-phase-three-band-eq-v1",
    }
    frequency = np.fft.rfftfreq(len(clean), 1.0 / RATE)
    response = 10.0 ** (_curves_numpy(frequency, values) / 20.0)
    wet = np.fft.irfft(np.fft.rfft(clean.astype(np.float64)) * response, n=len(clean))
    return wet.astype(np.float32), values


def normalized_controls(values: dict) -> np.ndarray:
    controls = np.asarray(
        (
            (values["low_gain_db"] + GAIN_DB) / (2.0 * GAIN_DB),
            (values["mid_gain_db"] + GAIN_DB) / (2.0 * GAIN_DB),
            math.log(values["mid_hz"] / 250.0) / math.log(5200.0 / 250.0),
            math.log(values["mid_q"] / 0.45) / math.log(3.5 / 0.45),
            (values["high_gain_db"] + GAIN_DB) / (2.0 * GAIN_DB),
        ),
        dtype=np.float32,
    )
    if controls.shape != (CONTROL_WIDTH,) or not np.isfinite(controls).all():
        raise ValueError("invalid spectral controls")
    return np.clip(controls, 0.0, 1.0)


class SpectralPairsV3(torch.utils.data.Dataset):
    """Balanced, split-disjoint Clean programs with ephemeral generic EQ."""

    def __init__(
        self,
        workspace: Path,
        split: str,
        samples: int,
        target_frames: int,
        seed: int,
    ) -> None:
        if split not in {"fit", "calibration", "development", "locked-final"}:
            raise ValueError(f"unsupported spectral split: {split}")
        if samples < 1 or target_frames < TARGET_FRAMES_MINIMUM:
            raise ValueError("spectral pair request is too small")
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
            raise ValueError(f"no product-safe spectral Clean for {split}")
        self.buckets = {name: tuple(rows) for name, rows in sorted(buckets.items())}
        self.source_ids = tuple(self.buckets)
        selected_sources = set(self.realized_source_counts()) | {"muspector-dsp"}
        if selected_sources & RESEARCH_SOURCE_IDS:
            raise PermissionError("research source entered spectral product pairs")
        self.authorization = require_product_weights(
            self.workspace / "remix/data_sources.json", sorted(selected_sources)
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
        wet, values = spectral_eq(clean, rng)
        if wet.shape != clean.shape or not np.isfinite(wet).all():
            raise ValueError("spectral renderer changed geometry or emitted non-finite audio")
        result = {
            "wet": torch.from_numpy(wet.copy()),
            "clean": torch.from_numpy(clean.copy()),
            "controls": torch.from_numpy(normalized_controls(values)),
            "control_values": values,
            "input_gain_db": input_gain_db,
            "target_start": CONTEXT_FRAMES,
            "target_end": CONTEXT_FRAMES + self.target_frames,
            "source_id": selected.source_id,
            "group": selected.group,
        }
        self._cache[index] = result
        return result
