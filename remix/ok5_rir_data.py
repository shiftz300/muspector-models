"""Fail-closed admission for the OK5 measured-room external validation set."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .ambience2 import _rir_decay_seconds
from .foundation_data import RATE
from .license_gate import require_product_uses


SOURCE_ID = "ok5-rir"
EXPECTED_LICENSE = "Creative Commons Attribution 4.0 International"
EXPECTED_FILES = 25
EXPECTED_MEASUREMENTS = 70
DECAY_DOMAIN_SECONDS = (0.10, 1.00)


@dataclass(frozen=True)
class OK5RIR:
    path: Path
    measurement: int
    room: str
    decay_p999_seconds: float


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _raw_measurements(path: Path) -> tuple[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if _decode(handle.attrs.get("Conventions", "")) != "SOFA":
            raise ValueError(f"OK5 file is not SOFA: {path}")
        if _decode(handle.attrs.get("SOFAConventions", "")) != "SingleRoomSRIR":
            raise ValueError(f"OK5 file has an unexpected SOFA convention: {path}")
        if _decode(handle.attrs.get("License", "")) != EXPECTED_LICENSE:
            raise PermissionError(f"OK5 file license differs: {path}")
        rate = np.asarray(handle["Data.SamplingRate"])
        impulse = np.asarray(handle["Data.IR"], dtype=np.float64)
        title = _decode(handle.attrs.get("Title", ""))
    if rate.shape != (1,) or float(rate[0]) != RATE:
        raise ValueError(f"OK5 file is not {RATE} Hz: {path}")
    if impulse.ndim != 3 or impulse.shape[1] != 6 or impulse.shape[2] < RATE:
        raise ValueError(f"OK5 file does not contain six full-length microphone channels: {path}")
    if not title or title != path.stem or not np.isfinite(impulse).all():
        raise ValueError(f"OK5 file metadata or signal is invalid: {path}")
    return title, impulse


def load_ok5_rir(path: Path, measurement: int) -> np.ndarray:
    """Load microphone channel zero and truncate to the declared Product4 domain."""
    _, values = _raw_measurements(path)
    if not 0 <= measurement < values.shape[0]:
        raise IndexError(measurement)
    value = values[measurement, 0]
    peak = float(np.max(np.abs(value)))
    active = np.flatnonzero(np.abs(value) >= max(peak * 0.01, 1.0e-8))
    if not len(active):
        raise ValueError(f"silent OK5 RIR: {path} measurement={measurement}")
    value = value[int(active[0]) : int(active[0]) + RATE].copy()
    energy = float(np.sum(np.square(value)))
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError(f"invalid OK5 RIR energy: {path} measurement={measurement}")
    value /= np.sqrt(energy)
    value.setflags(write=False)
    return value


def discover_ok5_rirs(root: Path) -> tuple[list[OK5RIR], dict]:
    root = root.resolve()
    paths = sorted((root / "measurements").glob("*.sofa"))
    if len(paths) != EXPECTED_FILES:
        raise ValueError(f"OK5 file inventory differs: {len(paths)} != {EXPECTED_FILES}")
    rows = []
    total = 0
    excluded = []
    for path in paths:
        room, values = _raw_measurements(path)
        total += values.shape[0]
        for measurement in range(values.shape[0]):
            # Admission is computed on the full released response; the loaded
            # profile is then truncated to one second like the Product4 domain.
            raw = values[measurement, 0]
            peak = float(np.max(np.abs(raw)))
            active = np.flatnonzero(np.abs(raw) >= max(peak * 0.01, 1.0e-8))
            if not len(active):
                raise ValueError(f"silent OK5 RIR: {path} measurement={measurement}")
            decay = _rir_decay_seconds(raw[int(active[0]) :])
            if DECAY_DOMAIN_SECONDS[0] <= decay <= DECAY_DOMAIN_SECONDS[1]:
                rows.append(OK5RIR(path.resolve(), measurement, room, decay))
            else:
                excluded.append({
                    "room": room,
                    "measurement": measurement,
                    "decay_p999_seconds": decay,
                    "reason": "outside-product4-decay-domain",
                })
    if total != EXPECTED_MEASUREMENTS:
        raise ValueError(f"OK5 measurement inventory differs: {total} != {EXPECTED_MEASUREMENTS}")
    rooms = sorted({row.room for row in rows})
    if not rows or len(rooms) < 3:
        raise ValueError("OK5 domain admission has insufficient room diversity")
    return rows, {
        "files": len(paths),
        "measurements": total,
        "admitted_measurements": len(rows),
        "admitted_rooms": rooms,
        "excluded_measurements": excluded,
        "receiver_channel": 0,
        "receiver_channel_contract": "one physical GRAS 40AI microphone; no spatial combination",
        "loaded_profile_frames": RATE,
    }


def audit(workspace: Path) -> dict:
    root = (workspace.resolve() / "data/corpus/ok5-rir").resolve()
    rows, inventory = discover_ok5_rirs(root)
    authorization = require_product_uses(
        workspace / "remix/data_sources.json", {SOURCE_ID: "validate-reverb"}
    )
    return {
        "schema": 1,
        "status": "admitted-external-room-validation",
        "source_id": SOURCE_ID,
        "authorization": authorization,
        "inventory": inventory,
        "decay_domain_seconds": list(DECAY_DOMAIN_SECONDS),
        "decay_p999_seconds": {
            "minimum": min(row.decay_p999_seconds for row in rows),
            "median": float(np.median([row.decay_p999_seconds for row in rows])),
            "maximum": max(row.decay_p999_seconds for row in rows),
        },
        "selection_use": "external validation only; never training, calibration, or parameter selection",
        "source_audio_read_only": True,
        "locked_final": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product4-reverb-ok5-data-audit-v1.json"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace OK5 audit: {output}")
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
