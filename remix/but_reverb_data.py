"""Room-disjoint, fail-closed admission for BUT Speech@FIT real RIRs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import soundfile
import numpy as np

from .license_gate import require_product_uses


SOURCE_ID = "but-reverbdb"
EXPECTED_ROOM_SPLITS = {
    "VUT_FIT_Q301": "calibration",
    "VUT_FIT_L207": "fit",
    "VUT_FIT_L212": "fit",
    "Hotel_SkalskyDvur_Room112": "fit",
    "VUT_FIT_L227": "fit",
    "Hotel_SkalskyDvur_ConferenceRoom2": "development",
    "VUT_FIT_E112": "development",
    "VUT_FIT_D105": "locked-final",
    "VUT_FIT_C236": "locked-final",
}
SPLITS = ("fit", "calibration", "development", "locked-final")


def _interleave_rooms(root: Path, paths: list[Path]) -> list[Path]:
    """Keep every early training/evaluation prefix representative of its rooms."""
    by_room: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_room[path.relative_to(root).parts[0]].append(path)
    ordered = []
    rooms = sorted(by_room)
    for index in range(max(len(values) for values in by_room.values())):
        for room in rooms:
            if index < len(by_room[room]):
                ordered.append(by_room[room][index])
    return ordered


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("cannot summarize empty or non-finite BUT ReverbDB values")
    return {
        "minimum": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


def _signal_metrics(path: Path) -> dict:
    audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"invalid BUT ReverbDB RIR audio: {path}")
    mono = np.asarray(audio[:, 0], dtype=np.float64)
    peak = float(np.max(np.abs(mono)))
    active = np.flatnonzero(np.abs(mono) >= max(peak * 0.01, 1.0e-8))
    if not len(active):
        raise ValueError(f"silent BUT ReverbDB RIR: {path}")
    mono = mono[int(active[0]) :]
    energy = np.square(mono)
    cumulative = np.cumsum(energy)
    total = float(cumulative[-1])
    if total <= 0.0:
        raise ValueError(f"zero-energy BUT ReverbDB RIR: {path}")
    decay_index = int(np.searchsorted(cumulative, 0.999 * total))
    direct_frames = max(1, round(0.020 * rate))
    return {
        "decay_p999_seconds": (decay_index + 1) / rate,
        "direct_20ms_energy_fraction": float(np.sum(energy[:direct_frames]) / total),
        "duration_seconds": len(mono) / rate,
        "peak": peak,
    }


def rir_splits(root: Path) -> dict[str, list[Path]]:
    root = root.resolve()
    paths = sorted(
        path.resolve()
        for path in root.rglob("*.wav")
        if "RIR" in path.relative_to(root).parts
    )
    if not paths:
        raise ValueError(f"BUT ReverbDB contains no RIR WAV files: {root}")
    rooms = {path.relative_to(root).parts[0] for path in paths}
    unknown = rooms - set(EXPECTED_ROOM_SPLITS)
    missing = set(EXPECTED_ROOM_SPLITS) - rooms
    if unknown or missing:
        raise ValueError(
            f"BUT ReverbDB room inventory differs: unknown={sorted(unknown)} missing={sorted(missing)}"
        )
    result = {split: [] for split in SPLITS}
    for path in paths:
        room = path.relative_to(root).parts[0]
        result[EXPECTED_ROOM_SPLITS[room]].append(path)
    result = {
        split: _interleave_rooms(root, values)
        for split, values in result.items()
    }
    room_sets = {
        split: {path.relative_to(root).parts[0] for path in values}
        for split, values in result.items()
    }
    if any(
        room_sets[left] & room_sets[right]
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    ):
        raise RuntimeError("BUT ReverbDB room leaked across splits")
    return result


def audit(workspace: Path, *, open_locked_final: bool = False) -> dict:
    workspace = workspace.resolve()
    root = (workspace / "data/corpus/but-reverbdb").resolve()
    splits = rir_splits(root)
    authorization = require_product_uses(
        workspace / "remix/data_sources.json",
        {SOURCE_ID: ("train-reverb", "validate-reverb")},
    )
    geometry = Counter()
    nonfinite_headers = []
    signal_rows = []
    for split, paths in splits.items():
        if split == "locked-final" and not open_locked_final:
            continue
        for path in paths:
            info = soundfile.info(path)
            geometry[(info.samplerate, info.channels, info.subtype)] += 1
            if info.frames <= 0 or info.samplerate <= 0 or info.channels != 1:
                nonfinite_headers.append(str(path.relative_to(root)))
                continue
            values = _signal_metrics(path)
            signal_rows.append({
                "split": split,
                "room": path.relative_to(root).parts[0],
                **values,
            })
    room_counts = {
        split: dict(Counter(path.relative_to(root).parts[0] for path in paths))
        for split, paths in splits.items()
    }
    return {
        "schema": 1,
        "status": "admitted-room-disjoint" if not nonfinite_headers else "rejected",
        "source_id": SOURCE_ID,
        "authorization": authorization,
        "rir_counts": {split: len(paths) for split, paths in splits.items()},
        "room_counts": room_counts,
        "room_overlap": 0,
        "geometry": [
            {"sample_rate": key[0], "channels": key[1], "subtype": key[2], "files": value}
            for key, value in sorted(geometry.items())
        ],
        "invalid_headers": nonfinite_headers,
        "seen_signal_metrics": {
            name: _summary([row[name] for row in signal_rows])
            for name in (
                "decay_p999_seconds",
                "direct_20ms_energy_fraction",
                "duration_seconds",
                "peak",
            )
        },
        "seen_room_signal_metrics": {
            room: {
                name: _summary([row[name] for row in signal_rows if row["room"] == room])
                for name in ("decay_p999_seconds", "direct_20ms_energy_fraction")
            }
            for room in sorted({row["room"] for row in signal_rows})
        },
        "locked_final_audio_opened": open_locked_final,
        "split_policy": "whole rooms only; no microphone or speaker position crosses a split",
        "source_audio_read_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product4-reverb/but-data-audit.json"),
    )
    parser.add_argument("--open-locked-final", action="store_true")
    args = parser.parse_args()
    report = audit(args.workspace, open_locked_final=args.open_locked_final)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace BUT ReverbDB audit: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "admitted-room-disjoint":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
