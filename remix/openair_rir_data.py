#!/usr/bin/env python3
"""Fail-closed OpenAIR admission for a fresh external-room Reverb gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile
from scipy.signal import resample_poly

from .ambience2 import _rir_decay_seconds
from .foundation_data import RATE
from .license_gate import require_product_uses


SOURCE_ID = "openair-rir-external-v1"
DECAY_DOMAIN_SECONDS = (0.10, 1.00)
ROOM_SPECS = {
    "creswell-crags": {"branch": "b-format", "files": 13, "channels": 4, "rate": 96_000},
    "dixon-studio-theatre-university-york": {
        "branch": "b-format", "files": 5, "channels": 4, "rate": 96_000,
    },
    "genesis-6-studio-live-room-drum-set": {
        "branch": "b-format", "files": 8, "channels": 4, "rate": 96_000,
    },
    "hoffmann-lime-kiln-langcliffeuk": {
        "branch": "b-format", "files": 6, "channels": 4, "rate": 96_000,
    },
    "spokane-womans-club": {"branch": "stereo", "files": 1, "channels": 2, "rate": 44_100},
}
ARCHIVES = {
    "creswell-crags.zip": (8_363_306, "ebd30802d2d2c6d1fc0df107e8bca3cdb2102e0a3d136a26a97b10d41476601a"),
    "dixon-studio-theatre-university-york.zip": (4_215_929, "14c72bc66612932927dd91fbb8c7ec1fa99d3f0b4b55a19be15970d722762a9e"),
    "genesis-6-studio-live-room-drum-set.zip": (11_713_961, "f4d47d33628a27cb4eb343a30581952482aa6f1853392bdb909e6137b8680b64"),
    "hoffmann-lime-kiln-langcliffeuk.zip": (7_419_547, "e2c37dae18f2c503f7972fb9ec02502124183405533484099772c1fcb1187a5c"),
    "spokane-womans-club.zip": (2_629_960, "77d24ffdda95ea53cb5689bf2c195aae6ef09d509a6e4c2f475a79ed409619e6"),
}


@dataclass(frozen=True)
class OpenAIRRIR:
    path: Path
    room: str
    decay_p999_seconds: float


def _source_channel(path: Path, spec: dict) -> np.ndarray:
    info = soundfile.info(path)
    if info.channels != spec["channels"] or info.samplerate != spec["rate"]:
        raise ValueError(f"OpenAIR format differs from frozen inventory: {path}")
    value, rate = soundfile.read(path, dtype="float32", always_2d=True)
    value = np.asarray(value[:, 0], dtype=np.float64)
    if rate != RATE:
        divisor = math.gcd(rate, RATE)
        value = resample_poly(value, RATE // divisor, rate // divisor)
    if not len(value) or not np.isfinite(value).all():
        raise ValueError(f"invalid OpenAIR impulse response: {path}")
    return value


def _onset(value: np.ndarray, path: Path) -> int:
    peak = float(np.max(np.abs(value)))
    active = np.flatnonzero(np.abs(value) >= max(peak * 0.01, 1.0e-8))
    if not len(active):
        raise ValueError(f"silent OpenAIR impulse response: {path}")
    return int(active[0])


@lru_cache(maxsize=64)
def load_openair_rir(path: Path) -> np.ndarray:
    path = path.resolve()
    room = path.parent.parent.name
    if room not in ROOM_SPECS:
        raise ValueError(f"unregistered OpenAIR room: {room}")
    value = _source_channel(path, ROOM_SPECS[room])
    start = _onset(value, path)
    value = value[start : start + RATE].copy()
    if len(value) < RATE:
        value = np.pad(value, (0, RATE - len(value)))
    energy = float(np.sum(np.square(value)))
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError(f"invalid OpenAIR energy: {path}")
    value /= np.sqrt(energy)
    value.setflags(write=False)
    return value


def _archive_audit(root: Path) -> dict:
    archive_root = root / "archives"
    found = sorted(archive_root.glob("*.zip")) if archive_root.is_dir() else []
    if not found:
        return {"present": False, "expected": sorted(ARCHIVES)}
    if {path.name for path in found} != set(ARCHIVES):
        raise ValueError("OpenAIR archive inventory differs")
    rows = {}
    for path in found:
        expected_bytes, expected_sha256 = ARCHIVES[path.name]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != expected_bytes or digest != expected_sha256:
            raise ValueError(f"OpenAIR archive integrity differs: {path}")
        rows[path.name] = {"bytes": expected_bytes, "sha256": digest}
    return {"present": True, "files": rows}


def discover_openair_rirs(root: Path) -> tuple[list[OpenAIRRIR], dict]:
    root = root.resolve()
    measurement_root = root / "measurements"
    admitted = []
    excluded = []
    total = 0
    for room, spec in sorted(ROOM_SPECS.items()):
        paths = sorted((measurement_root / room / spec["branch"]).glob("*.wav"))
        if len(paths) != spec["files"]:
            raise ValueError(
                f"OpenAIR room inventory differs: {room}: {len(paths)} != {spec['files']}"
            )
        total += len(paths)
        for path in paths:
            value = _source_channel(path, spec)
            decay = _rir_decay_seconds(value[_onset(value, path) :])
            row = OpenAIRRIR(path.resolve(), room, decay)
            if DECAY_DOMAIN_SECONDS[0] <= decay <= DECAY_DOMAIN_SECONDS[1]:
                admitted.append(row)
            else:
                excluded.append({
                    "room": room,
                    "file": path.name,
                    "decay_p999_seconds": decay,
                    "reason": "outside-product4-decay-domain",
                })
    rooms = sorted({row.room for row in admitted})
    if total != sum(spec["files"] for spec in ROOM_SPECS.values()):
        raise ValueError("OpenAIR total inventory differs")
    if len(rooms) < 3:
        raise ValueError("OpenAIR admission has insufficient independent rooms")
    return admitted, {
        "measurements": total,
        "admitted_measurements": len(admitted),
        "admitted_rooms": rooms,
        "excluded_measurements": excluded,
        "receiver_channel": 0,
        "receiver_channel_contract": (
            "W channel for B-format files; left physical microphone for the stereo file; "
            "no spatial synthesis or channel combination"
        ),
        "loaded_profile_frames": RATE,
        "archives": _archive_audit(root),
    }


def audit(workspace: Path) -> dict:
    root = (workspace.resolve() / "data/corpus/openair-external-v1").resolve()
    rows, inventory = discover_openair_rirs(root)
    authorization = require_product_uses(
        workspace / "remix/data_sources.json", {SOURCE_ID: "validate-reverb"}
    )
    return {
        "schema": 1,
        "status": "admitted-fresh-external-room-validation",
        "source_id": SOURCE_ID,
        "authorization": authorization,
        "inventory": inventory,
        "decay_domain_seconds": list(DECAY_DOMAIN_SECONDS),
        "decay_p999_seconds": {
            "minimum": min(row.decay_p999_seconds for row in rows),
            "median": float(np.median([row.decay_p999_seconds for row in rows])),
            "maximum": max(row.decay_p999_seconds for row in rows),
        },
        "selection_use": "fresh external validation only; never training, calibration, or parameter selection",
        "source_audio_read_only": True,
        "locked_final": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product4-reverb-openair-data-audit-v1.json"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace OpenAIR audit: {output}")
    report = audit(args.workspace.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "status": report["status"],
        "admitted_measurements": report["inventory"]["admitted_measurements"],
        "admitted_rooms": len(report["inventory"]["admitted_rooms"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
