"""Fail-closed EG-IPT physical Amp+cab+SM57 fit-only pairs."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile
import torch
from scipy.signal import resample_poly

from .license_gate import require_product_uses


SOURCE_ID = "eg-ipt"
SOURCE_RATE = 96_000
RATE = 44_100
LAG_FRAMES = 64
CLEAN_ROOT = Path("data/corpus/eg-ipt")
WET_ROOT = Path("data/corpus/eg-ipt-sm57")
AUDIT = Path("runs/foundation/product3-amp-eg-ipt-source-audit/pair-audit-v1.json")


@dataclass(frozen=True)
class Pair:
    relative: Path
    clean: Path
    wet: Path
    pickup: str
    technique: str
    frames: int


def wet_relative(clean_relative: Path) -> Path:
    parts = list(clean_relative.parts)
    parts[parts.index("DI")] = "dyn"
    parts[-1] = parts[-1].replace("_DI.wav", "_dyn.wav")
    return Path(*parts)


def discover(workspace: Path) -> tuple[Pair, ...]:
    workspace = workspace.resolve()
    audit = json.loads((workspace / AUDIT).read_text())
    if not audit.get("accepted") or audit.get("pairs") != 8_717:
        raise PermissionError("EG-IPT Amp pairs require the accepted v1 source audit")
    clean_root, wet_root = workspace / CLEAN_ROOT, workspace / WET_ROOT
    pairs = []
    for clean in sorted(clean_root.glob("EG-IPT/*/DI/*/*_DI.wav")):
        relative = clean.relative_to(clean_root)
        wet = wet_root / wet_relative(relative)
        if not wet.exists():
            continue
        clean_info, wet_info = soundfile.info(clean), soundfile.info(wet)
        if (
            clean_info.samplerate != SOURCE_RATE or wet_info.samplerate != SOURCE_RATE
            or clean_info.channels != 1 or wet_info.channels != 1
            or clean_info.frames != wet_info.frames
        ):
            raise ValueError(f"EG-IPT pair geometry changed: {relative}")
        pairs.append(Pair(
            relative, clean, wet, relative.parts[1], relative.parts[3], clean_info.frames
        ))
    if len(pairs) != 8_717:
        raise ValueError(f"expected 8717 audited EG-IPT pairs, found {len(pairs)}")
    return tuple(pairs)


def _partition(pair: Pair) -> str:
    digest = hashlib.sha256(pair.relative.as_posix().encode()).digest()
    return "internal_calibration" if int.from_bytes(digest[:4], "big") % 10 == 0 else "fit"


def _read(path: Path, start: int, frames: int) -> np.ndarray:
    with soundfile.SoundFile(path) as stream:
        stream.seek(start)
        value = stream.read(frames, dtype="float32", always_2d=True)
    if value.shape != (frames, 1) or not np.isfinite(value).all():
        raise ValueError(f"invalid EG-IPT source window: {path}: {value.shape}")
    return np.asarray(value[:, 0], dtype=np.float32)


class EgIptAmpPairs(torch.utils.data.Dataset):
    """Virtual balanced crops; both splits remain fit-only, never product validation."""

    def __init__(
        self, workspace: Path, split: str, samples: int, target_frames: int,
        context_frames: int, seed: int,
        *,
        include_relatives: frozenset[str] | None = None,
        exclude_relatives: frozenset[str] = frozenset(),
    ) -> None:
        if split not in {"fit", "internal_calibration", "reserved"}:
            raise ValueError("EG-IPT exposes fit, internal_calibration and explicit reserved pairs only")
        if split == "reserved" and include_relatives is None:
            raise ValueError("reserved EG-IPT split requires an explicit pair allowlist")
        if samples < 2 or target_frames < 4096 or context_frames < 0:
            raise ValueError("invalid EG-IPT Amp dataset request")
        self.workspace = workspace.resolve()
        self.authorization = require_product_uses(
            self.workspace / "remix/data_sources.json", {SOURCE_ID: "train-amp"}
        )
        self.samples = samples
        self.target_frames = target_frames
        self.context_frames = context_frames
        self.total_frames = target_frames + 2 * context_frames
        self.source_frames = math.ceil(self.total_frames * SOURCE_RATE / RATE) + 64
        partition_rows = [
            pair for pair in discover(self.workspace)
            if split == "reserved" or _partition(pair) == split
        ]
        if include_relatives is not None:
            partition_rows = [
                pair for pair in partition_rows
                if pair.relative.as_posix() in include_relatives
            ]
            missing = include_relatives - {
                pair.relative.as_posix() for pair in partition_rows
            }
            if missing:
                raise ValueError(f"reserved EG-IPT pairs missing from split: {len(missing)}")
        if exclude_relatives:
            partition_rows = [
                pair for pair in partition_rows
                if pair.relative.as_posix() not in exclude_relatives
            ]
        rows = [
            pair for pair in partition_rows
            if pair.frames >= self.source_frames + LAG_FRAMES
        ]
        self.short_pairs_excluded = len(partition_rows) - len(rows)
        strata = {}
        for pair in rows:
            strata.setdefault((pair.pickup, pair.technique), []).append(pair)
        pickups = {key[0] for key in strata}
        techniques = {key[1] for key in strata}
        minimum_techniques = 15 if split == "reserved" else 19
        if len(pickups) != 3 or len(techniques) < minimum_techniques:
            raise ValueError(
                f"EG-IPT split lost a pickup or technique: {len(pickups)}, {len(techniques)}"
            )
        self.strata = tuple((key, tuple(value)) for key, value in sorted(strata.items()))
        self.seed = seed
        self.split = split
        self.include_relatives = include_relatives
        self.exclude_relatives = exclude_relatives

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.samples:
            raise IndexError(index)
        stratum, rows = self.strata[index % len(self.strata)]
        rng = random.Random(self.seed + index * 104729)
        pair = rows[rng.randrange(len(rows))]
        last = pair.frames - self.source_frames - LAG_FRAMES
        if last < 0:
            raise ValueError(f"EG-IPT pair is too short: {pair.relative}")
        for _ in range(64):
            start = rng.randint(0, last)
            clean_source = _read(pair.clean, start, self.source_frames)
            wet_source = _read(pair.wet, start + LAG_FRAMES, self.source_frames)
            clean = resample_poly(clean_source, 147, 320)[:self.total_frames].astype(np.float32)
            wet = resample_poly(wet_source, 147, 320)[:self.total_frames].astype(np.float32)
            target = slice(self.context_frames, self.context_frames + self.target_frames)
            signal = max(
                float(np.sqrt(np.mean(clean[target].astype(np.float64) ** 2))),
                float(np.sqrt(np.mean(wet[target].astype(np.float64) ** 2))),
            )
            effect = float(np.sqrt(np.mean((wet[target].astype(np.float64) - clean[target]) ** 2)))
            if signal >= 1.0e-4 and effect >= 0.05 * signal:
                break
        else:
            raise ValueError(f"no active EG-IPT crop found: {pair.relative}")
        return {
            "wet": torch.from_numpy(wet.copy()),
            "clean": torch.from_numpy(clean.copy()),
            "tone_reference": torch.from_numpy(
                np.resize(wet, 3 * RATE).astype(np.float32).copy()
            ),
            "crop_start": self.context_frames,
            "crop_end": self.context_frames + self.target_frames,
            "category": "physical_fixed",
            "source_id": SOURCE_ID,
            "profile_id": "evh5150iii-mesa4x12-v30-sm57-flat",
            "pickup": stratum[0],
            "technique": stratum[1],
            "relative": pair.relative.as_posix(),
            "product_gate_role": "fit-only",
        }


def split_summary(workspace: Path) -> dict:
    pairs = discover(workspace)
    groups = {name: [pair for pair in pairs if _partition(pair) == name] for name in (
        "fit", "internal_calibration"
    )}
    return {
        name: {
            "pairs": len(rows),
            "pickups": sorted({row.pickup for row in rows}),
            "techniques": sorted({row.technique for row in rows}),
            "duration_hours": sum(row.frames for row in rows) / SOURCE_RATE / 3600.0,
            "product_gate_role": "fit-only",
        }
        for name, rows in groups.items()
    }
