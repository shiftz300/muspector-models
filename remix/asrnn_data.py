#!/usr/bin/env python3
"""Offline admission and loading for the CC-BY-NC ASRNN physical-device data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile


SAMPLE_RATE = 48_000
LICENSE = "CC-BY-NC-4.0"
RECORD_URL = "https://zenodo.org/records/20406285"


def _rat_controls(path: Path) -> np.ndarray:
    fields = path.stem.split(",")
    if len(fields) != 3:
        raise ValueError(f"RAT filename must encode distortion,filter,volume: {path.name}")
    try:
        distortion, filter_control, volume = (int(value) for value in fields)
    except ValueError as error:
        raise ValueError(f"RAT filename has non-integer controls: {path.name}") from error
    if any(not 0 <= value <= 100 for value in (distortion, filter_control, volume)):
        raise ValueError(f"RAT controls are outside 0-100: {path.name}")
    # RAT Filter is a clockwise high-cut; Muspector Tone increases brightness.
    return np.asarray(
        (distortion / 100.0, 1.0 - filter_control / 100.0, volume / 100.0),
        dtype=np.float32,
    )


def _device_root(root: Path, device: str) -> Path:
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.casefold() == device.casefold()
    ]
    with_splits = [
        path for path in candidates if (path / "train").is_dir() and (path / "eval").is_dir()
    ]
    if len(with_splits) != 1:
        raise ValueError(f"expected one {device} directory with train/eval, found {with_splits}")
    return with_splits[0]


def rat_files(root: Path, split: str) -> list[Path]:
    if split not in {"train", "eval"}:
        raise ValueError(f"unsupported ASRNN split: {split!r}")
    paths = sorted((_device_root(root, "rat") / split).glob("*.wav"))
    if not paths:
        raise ValueError(f"ASRNN RAT {split} split is empty")
    return paths


def read_rat_pair(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    audio, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE or audio.shape[1] != 2:
        raise ValueError(
            f"ASRNN RAT file must be 48 kHz stereo Dry/Wet: {path} "
            f"({sample_rate} Hz, {audio.shape[1]} channels)"
        )
    if not np.isfinite(audio).all():
        raise ValueError(f"ASRNN RAT file contains non-finite audio: {path}")
    return audio[:, 0].copy(), audio[:, 1].copy(), _rat_controls(path)


def audit_asrnn(root: Path, *, scan_audio: bool = True) -> dict:
    splits = {}
    all_controls = []
    dry_digests: dict[str, set[str]] = {"train": set(), "eval": set()}
    total_frames = 0
    subtypes = Counter()
    for split in ("train", "eval"):
        paths = rat_files(root, split)
        split_controls = []
        split_frames = 0
        for path in paths:
            controls = _rat_controls(path)
            split_controls.append(controls)
            info = soundfile.info(path)
            if info.samplerate != SAMPLE_RATE or info.channels != 2:
                raise ValueError(f"ASRNN RAT geometry differs: {path} ({info})")
            if info.subtype not in {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}:
                raise ValueError(f"ASRNN RAT audio is not lossless PCM/float: {path}")
            if scan_audio:
                dry, wet, _ = read_rat_pair(path)
                if len(dry) != len(wet):
                    raise ValueError(f"ASRNN RAT Dry/Wet frame count differs: {path}")
                dry_digests[split].add(hashlib.sha256(dry.tobytes()).hexdigest())
            split_frames += info.frames
            subtypes[info.subtype] += 1
        values = np.stack(split_controls)
        splits[split] = {
            "files": len(paths),
            "duration_hours": split_frames / SAMPLE_RATE / 3600.0,
            "control_minimum": values.min(axis=0).tolist(),
            "control_maximum": values.max(axis=0).tolist(),
            "unique_control_values": [len(np.unique(values[:, index])) for index in range(3)],
        }
        all_controls.append(values)
        total_frames += split_frames
    shared_settings = {
        path.name
        for path in rat_files(root, "train")
    } & {path.name for path in rat_files(root, "eval")}
    exact_dry_overlap = dry_digests["train"] & dry_digests["eval"]
    if exact_dry_overlap:
        raise ValueError(
            "ASRNN RAT contains exact Dry audio duplicates across official train/eval: "
            f"{len(exact_dry_overlap)}"
        )
    values = np.concatenate(all_controls, axis=0)
    return {
        "schema": 1,
        "dataset": "asrnn-physical-effects",
        "record_url": RECORD_URL,
        "license": LICENSE,
        "device": "ProCo RAT",
        "root": str(root),
        "splits": splits,
        "files": sum(value["files"] for value in splits.values()),
        "duration_hours": total_frames / SAMPLE_RATE / 3600.0,
        "control_order": ["distortion", "tone-inverted-from-filter", "volume"],
        "control_minimum": values.min(axis=0).tolist(),
        "control_maximum": values.max(axis=0).tolist(),
        "subtypes": dict(sorted(subtypes.items())),
        # Filenames encode settings, so repeated names across splits are valid and do not
        # identify repeated performances. Exact Dry audio is the local leakage check.
        "official_train_eval_shared_control_settings": len(shared_settings),
        "official_train_eval_exact_dry_duplicates": (
            len(exact_dry_overlap) if scan_audio else None
        ),
        "audio_scanned": scan_audio,
        "automatic_normalization": False,
        "source_files_read_only": True,
        "physical_audio_devices_used": False,
        "scope": (
            "internal non-commercial real-hardware pilot; official data has no independent "
            "calibrate or locked-final split"
        ),
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_asrnn(args.root, scan_audio=not args.metadata_only)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
