#!/usr/bin/env python3
"""Fail-closed admission of the frozen OpenSLR 26 simulated-RIR subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile
from scipy.signal import resample_poly

from .ambience2 import _rir_decay_seconds
from .foundation_data import RATE
from .license_gate import require_product_uses


SOURCE_ID = "openslr26-simulated-rir-external-v1"
DECAY_DOMAIN_SECONDS = (0.10, 1.00)
EXPECTED_ARCHIVE_BYTES = 178_818_025
EXPECTED_ARCHIVE_MD5 = "d44dbfd252bda8885af67bd288c18015"
EXPECTED_CATEGORIES = ("largeroom", "mediumroom", "smallroom")
EXPECTED_ROOMS = tuple(f"Room{index:03d}" for index in range(191, 201))
EXPECTED_POSITIONS = ("00001", "00025", "00050", "00100")
PRODUCT_SPLIT_RANGES = {
    "fit": range(1, 121),
    "calibration": range(121, 151),
    "development": range(151, 181),
    "locked-final": range(0),
}


@dataclass(frozen=True)
class OpenSLR26RIR:
    path: Path
    category: str
    room: str
    decay_p999_seconds: float


def _expected_relative_paths() -> set[Path]:
    return {
        Path("simulated_rirs_16k") / category / room / f"{room}-{position}.wav"
        for category in EXPECTED_CATEGORIES
        for room in EXPECTED_ROOMS
        for position in EXPECTED_POSITIONS
    }


def _onset(value: np.ndarray, path: Path) -> int:
    peak = float(np.max(np.abs(value)))
    active = np.flatnonzero(np.abs(value) >= max(peak * 0.01, 1.0e-8))
    if not len(active):
        raise ValueError(f"silent OpenSLR 26 RIR: {path}")
    return int(active[0])


@lru_cache(maxsize=128)
def load_openslr26_rir(path: Path) -> np.ndarray:
    path = path.resolve()
    value, rate = soundfile.read(path, dtype="float32", always_2d=True)
    if value.shape[1] != 1 or rate != 16_000:
        raise ValueError(f"OpenSLR 26 geometry differs: {path}")
    value = np.asarray(value[:, 0], dtype=np.float64)
    value = value[_onset(value, path) :]
    divisor = math.gcd(rate, RATE)
    value = resample_poly(value, RATE // divisor, rate // divisor)
    value = value[:RATE].copy()
    if len(value) < RATE:
        value = np.pad(value, (0, RATE - len(value)))
    energy = float(np.sum(np.square(value)))
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError(f"invalid OpenSLR 26 RIR energy: {path}")
    value /= np.sqrt(energy)
    value.setflags(write=False)
    return value


def discover_openslr26_rirs(root: Path) -> tuple[list[OpenSLR26RIR], dict]:
    root = root.resolve()
    measurement_root = root / "measurements"
    base = measurement_root / "external-v1/simulated_rirs_16k"
    paths = sorted(path.resolve() for path in base.rglob("*.wav"))
    relative = {
        Path("simulated_rirs_16k") / path.relative_to(base) for path in paths
    }
    expected = _expected_relative_paths()
    if relative != expected:
        raise ValueError(
            "OpenSLR 26 frozen subset differs: "
            f"missing={len(expected - relative)} unexpected={len(relative - expected)}"
        )
    admitted = []
    excluded = []
    for path in paths:
        category, room = path.relative_to(base).parts[:2]
        impulse = load_openslr26_rir(path)
        decay = _rir_decay_seconds(impulse)
        row = OpenSLR26RIR(path, category, f"{category}/{room}", decay)
        if DECAY_DOMAIN_SECONDS[0] <= decay <= DECAY_DOMAIN_SECONDS[1]:
            admitted.append(row)
        else:
            excluded.append({
                "file": str(path.relative_to(measurement_root)),
                "decay_p999_seconds": decay,
                "reason": "outside-product4-decay-domain",
            })
    if len({row.room for row in admitted}) < 6:
        raise ValueError("OpenSLR 26 admission has insufficient independent rooms")
    archive = root / "archives/sim_rir_16k.zip"
    archive_audit = {"present": archive.is_file()}
    if archive.is_file():
        digest = hashlib.md5(archive.read_bytes()).hexdigest()
        if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES or digest != EXPECTED_ARCHIVE_MD5:
            raise ValueError("OpenSLR 26 archive integrity differs")
        archive_audit.update({"bytes": archive.stat().st_size, "md5": digest})
    return admitted, {
        "frozen_subset_files": len(paths),
        "frozen_categories": list(EXPECTED_CATEGORIES),
        "frozen_rooms_per_category": len(EXPECTED_ROOMS),
        "frozen_positions_per_room": list(EXPECTED_POSITIONS),
        "admitted_profiles": len(admitted),
        "admitted_rooms": sorted({row.room for row in admitted}),
        "excluded_profiles": excluded,
        "archive": archive_audit,
        "source_sample_rate": 16_000,
        "runtime_sample_rate": RATE,
        "loaded_profile_frames": RATE,
    }


@lru_cache(maxsize=1)
def product_rir_splits(root: Path) -> tuple[dict[str, list[Path]], dict]:
    """Return room-disjoint Room001-180 splits, excluding external Room191-200."""
    root = root.resolve()
    base = root / "measurements/product-splits/simulated_rirs_16k"
    paths = sorted(path.resolve() for path in base.rglob("*.wav"))
    expected = {
        Path(category) / f"Room{room:03d}" / f"Room{room:03d}-{position}.wav"
        for category in EXPECTED_CATEGORIES
        for room in range(1, 181)
        for position in EXPECTED_POSITIONS
    }
    relative = {path.relative_to(base) for path in paths}
    if relative != expected:
        raise ValueError(
            "OpenSLR 26 product subset differs: "
            f"missing={len(expected - relative)} unexpected={len(relative - expected)}"
        )
    result = {split: [] for split in PRODUCT_SPLIT_RANGES}
    excluded = []
    from .product_data import _rir

    for path in paths:
        room_number = int(path.parent.name.removeprefix("Room"))
        split = next(
            name for name, room_range in PRODUCT_SPLIT_RANGES.items()
            if room_number in room_range
        )
        decay = _rir_decay_seconds(_rir(path))
        if DECAY_DOMAIN_SECONDS[0] <= decay <= DECAY_DOMAIN_SECONDS[1]:
            result[split].append(path)
        else:
            excluded.append({
                "split": split,
                "file": str(path.relative_to(base)),
                "decay_p999_seconds": decay,
            })
    rooms = {
        split: {
            "/".join(path.relative_to(base).parts[:2]) for path in split_paths
        }
        for split, split_paths in result.items()
    }
    active = ("fit", "calibration", "development")
    if any(rooms[left] & rooms[right] for index, left in enumerate(active) for right in active[index + 1:]):
        raise RuntimeError("OpenSLR 26 room leaked across product splits")
    if any(not result[split] for split in active) or result["locked-final"]:
        raise ValueError("OpenSLR 26 product split coverage differs")
    return result, {
        "rir_counts": {name: len(rows) for name, rows in result.items()},
        "room_counts": {name: len(rows) for name, rows in rooms.items()},
        "excluded_outside_decay_domain_count": len(excluded),
        "excluded_outside_decay_domain_by_split": dict(sorted(Counter(
            row["split"] for row in excluded
        ).items())),
        "room_overlap": 0,
        "external_room_range": "Room191-200 excluded",
        "unused_room_range": "Room181-190",
    }


def audit(workspace: Path) -> dict:
    root = (workspace.resolve() / "data/corpus/openslr26-simulated-rir").resolve()
    rows, inventory = discover_openslr26_rirs(root)
    _, product_inventory = product_rir_splits(root)
    authorization = require_product_uses(
        workspace / "remix/data_sources.json",
        {SOURCE_ID: ("train-reverb", "validate-reverb")},
    )
    values = [row.decay_p999_seconds for row in rows]
    return {
        "schema": 1,
        "status": "admitted-room-disjoint-product-and-external-simulation",
        "source_id": SOURCE_ID,
        "authorization": authorization,
        "inventory": inventory,
        "product_room_splits": product_inventory,
        "decay_domain_seconds": list(DECAY_DOMAIN_SECONDS),
        "decay_p999_seconds": {
            "minimum": min(values),
            "median": float(np.median(values)),
            "maximum": max(values),
        },
        "selection_use": "Room001-180 product splits only; Room191-200 external-v1 permanently excluded from gradients and selection",
        "source_audio_read_only": True,
        "locked_final": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product4-reverb-openslr26-data-audit-v3.json"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace OpenSLR 26 audit: {output}")
    report = audit(args.workspace.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output), "status": report["status"],
        "admitted_profiles": report["inventory"]["admitted_profiles"],
        "admitted_rooms": len(report["inventory"]["admitted_rooms"]),
        "decay_p999_seconds": report["decay_p999_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
