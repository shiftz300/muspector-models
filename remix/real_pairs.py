"""Deterministic read-only real DAFx segments with split-group isolation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .data import _pair, discover_order_records


def single(
    root: Path,
    split: str,
    kind: str,
    frames: int,
    seed: int,
    count: int | None = None,
) -> list[dict]:
    records = [
        row for row in discover_order_records(root)
        if row.split == split and row.order == (kind,)
    ]
    if not records:
        raise RuntimeError(f"no real {kind} pairs for {split}")
    random.Random(seed).shuffle(records)
    if count is not None:
        records = records[:count]
    rows = []
    for index, record in enumerate(records):
        clean, wet = _pair(record)
        clean = resample_poly(clean, 160, 147).astype(np.float32)
        wet = resample_poly(wet, 160, 147).astype(np.float32)
        length = min(len(clean), len(wet))
        start = (seed + index * 104_729) % max(1, length - frames + 1)
        clean = clean[start : start + frames]
        wet = wet[start : start + frames]
        if len(clean) < frames:
            clean = np.pad(clean, (0, frames - len(clean)))
            wet = np.pad(wet, (0, frames - len(wet)))
        peak = max(float(np.max(np.abs(clean))), float(np.max(np.abs(wet))), 1.0e-6)
        rows.append({
            "target": np.asarray(clean * (0.22 / peak), dtype=np.float32),
            "source": np.asarray(wet * (0.22 / peak), dtype=np.float32),
            "group": record.group,
        })
    return rows


def chain(
    root: Path,
    split: str,
    frames: int,
    seed: int,
    count: int | None = None,
) -> list[dict]:
    supported = (("drive", "reverb"), ("reverb", "drive"))
    records = [
        row for row in discover_order_records(root)
        if row.split == split and row.order in supported
    ]
    if not records:
        raise RuntimeError(f"no real two-stage pairs for {split}")
    random.Random(seed).shuffle(records)
    if count is not None:
        records = records[:count]
    rows = []
    for index, record in enumerate(records):
        clean, wet = _pair(record)
        clean = resample_poly(clean, 160, 147).astype(np.float32)
        wet = resample_poly(wet, 160, 147).astype(np.float32)
        length = min(len(clean), len(wet))
        start = (seed + index * 104_729) % max(1, length - frames + 1)
        clean = clean[start : start + frames]
        wet = wet[start : start + frames]
        if len(clean) < frames:
            clean = np.pad(clean, (0, frames - len(clean)))
            wet = np.pad(wet, (0, frames - len(wet)))
        peak = max(float(np.max(np.abs(clean))), float(np.max(np.abs(wet))), 1.0e-6)
        rows.append({
            "target": np.asarray(clean * (0.22 / peak), dtype=np.float32),
            "source": np.asarray(wet * (0.22 / peak), dtype=np.float32),
            "group": record.group,
            "order": record.order,
        })
    return rows
