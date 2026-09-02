"""Generic offline admission for ASRNN physical-effect Dry/Wet pairs."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile

from .asrnn_data import LICENSE, RECORD_URL, SAMPLE_RATE


@dataclass(frozen=True)
class AsrnnEffectSpec:
    key: str
    directory: str
    display_name: str
    control_names: tuple[str, ...]
    filename_fields: int
    inverted_controls: tuple[int, ...] = ()


EFFECT_SPECS = {
    "rat": AsrnnEffectSpec(
        "rat",
        "rat",
        "ProCo RAT",
        ("distortion", "tone-inverted-from-filter", "volume"),
        3,
        (1,),
    ),
    "dfz": AsrnnEffectSpec(
        "dfz",
        "dfz",
        "Darkglass Duality Fuzz",
        ("blend", "filter"),
        3,
    ),
    "cs3": AsrnnEffectSpec(
        "cs3",
        "cs3",
        "Boss CS-3",
        ("attack",),
        2,
    ),
}


def effect_spec(device: str) -> AsrnnEffectSpec:
    try:
        return EFFECT_SPECS[device.casefold().replace("-", "")]
    except KeyError as error:
        raise ValueError(f"unsupported ASRNN device: {device!r}") from error


def _effect_root(root: Path, spec: AsrnnEffectSpec) -> Path:
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.casefold().replace("-", "") == spec.directory
    ]
    with_splits = [
        path for path in candidates if (path / "train").is_dir() and (path / "eval").is_dir()
    ]
    if len(with_splits) != 1:
        raise ValueError(
            f"expected one {spec.display_name} directory with train/eval, found {with_splits}"
        )
    return with_splits[0]


def effect_files(root: Path, device: str, split: str) -> list[Path]:
    if split not in {"train", "eval"}:
        raise ValueError(f"unsupported ASRNN split: {split!r}")
    spec = effect_spec(device)
    paths = sorted((_effect_root(root, spec) / split).glob("*.wav"))
    if not paths:
        raise ValueError(f"ASRNN {spec.display_name} {split} split is empty")
    return paths


def effect_controls(path: Path, device: str) -> np.ndarray:
    spec = effect_spec(device)
    fields = path.stem.split(",")
    if len(fields) != spec.filename_fields:
        raise ValueError(
            f"{spec.display_name} filename must contain {spec.filename_fields} fields: "
            f"{path.name}"
        )
    try:
        encoded = [int(value) for value in fields]
    except ValueError as error:
        raise ValueError(
            f"{spec.display_name} filename has non-integer fields: {path.name}"
        ) from error
    controls = encoded[: len(spec.control_names)]
    if any(not 0 <= value <= 100 for value in controls):
        raise ValueError(f"{spec.display_name} controls are outside 0-100: {path.name}")
    normalized = np.asarray(controls, dtype=np.float32) / 100.0
    for index in spec.inverted_controls:
        normalized[index] = 1.0 - normalized[index]
    return normalized


def read_effect_pair(
    path: Path, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = effect_spec(device)
    audio, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE or audio.shape[1] != 2:
        raise ValueError(
            f"ASRNN {spec.display_name} file must be 48 kHz stereo Dry/Wet: {path} "
            f"({sample_rate} Hz, {audio.shape[1]} channels)"
        )
    if not np.isfinite(audio).all():
        raise ValueError(f"ASRNN {spec.display_name} file contains non-finite audio: {path}")
    return audio[:, 0].copy(), audio[:, 1].copy(), effect_controls(path, device)


def audit_effect(root: Path, device: str, *, scan_audio: bool = True) -> dict:
    spec = effect_spec(device)
    splits = {}
    all_controls = []
    dry_digests: dict[str, set[str]] = {"train": set(), "eval": set()}
    total_frames = 0
    subtypes = Counter()
    for split in ("train", "eval"):
        paths = effect_files(root, spec.key, split)
        split_controls = []
        split_frames = 0
        for path in paths:
            controls = effect_controls(path, spec.key)
            split_controls.append(controls)
            info = soundfile.info(path)
            if info.samplerate != SAMPLE_RATE or info.channels != 2:
                raise ValueError(f"ASRNN {spec.display_name} geometry differs: {path}")
            if info.subtype not in {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}:
                raise ValueError(f"ASRNN {spec.display_name} audio is not lossless: {path}")
            if scan_audio:
                dry, wet, _ = read_effect_pair(path, spec.key)
                if len(dry) != len(wet):
                    raise ValueError(f"ASRNN Dry/Wet frame count differs: {path}")
                dry_digests[split].add(hashlib.sha256(dry.tobytes()).hexdigest())
            split_frames += info.frames
            subtypes[info.subtype] += 1
        values = np.stack(split_controls)
        splits[split] = {
            "files": len(paths),
            "duration_hours": split_frames / SAMPLE_RATE / 3600.0,
            "control_minimum": values.min(axis=0).tolist(),
            "control_maximum": values.max(axis=0).tolist(),
            "unique_control_values": [
                len(np.unique(values[:, index])) for index in range(values.shape[1])
            ],
        }
        all_controls.append(values)
        total_frames += split_frames
    overlap = dry_digests["train"] & dry_digests["eval"]
    if overlap:
        raise ValueError(
            f"ASRNN {spec.display_name} contains exact Dry duplicates across splits: "
            f"{len(overlap)}"
        )
    values = np.concatenate(all_controls)
    return {
        "schema": 1,
        "dataset": "asrnn-physical-effects",
        "record_url": RECORD_URL,
        "license": LICENSE,
        "device_key": spec.key,
        "device": spec.display_name,
        "root": str(root),
        "splits": splits,
        "files": sum(value["files"] for value in splits.values()),
        "duration_hours": total_frames / SAMPLE_RATE / 3600.0,
        "control_order": list(spec.control_names),
        "control_minimum": values.min(axis=0).tolist(),
        "control_maximum": values.max(axis=0).tolist(),
        "subtypes": dict(sorted(subtypes.items())),
        "official_train_eval_exact_dry_duplicates": len(overlap) if scan_audio else None,
        "audio_scanned": scan_audio,
        "automatic_normalization": False,
        "source_files_read_only": True,
        "physical_audio_devices_used": False,
        "scope": "internal non-commercial real-hardware pilot; no locked-final split",
        "passed": True,
    }
