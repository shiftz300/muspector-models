#!/usr/bin/env python3
"""Validate aligned hardware captures for a future controllable renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile

from .spec import ChainSpec, Delay, Drive, Reverb


SCHEMA = 1
CAPTURE_SAMPLE_RATE = 48_000
SPLITS = {"train", "calibrate", "valid", "locked-final"}
LOSSLESS_SUBTYPES = {
    "PCM_S8",
    "PCM_U8",
    "PCM_16",
    "PCM_24",
    "PCM_32",
    "FLOAT",
    "DOUBLE",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    relative = Path(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"capture paths must be relative and contained: {value}")
    return relative


def _path(root: Path, value: str) -> Path:
    relative = _relative_path(value)
    path = root / relative
    if not path.is_file():
        raise ValueError(f"capture file is missing: {path}")
    return path


def _chain(document: dict) -> ChainSpec:
    effects = []
    for value in document.get("effects", []):
        fields = dict(value)
        kind = fields.pop("kind", None)
        if kind == "drive":
            effects.append(Drive(**fields))
        elif kind == "delay":
            effects.append(Delay(**fields))
        elif kind == "reverb":
            effects.append(Reverb(**fields))
        else:
            raise ValueError(f"unsupported captured effect family: {kind!r}")
    chain = ChainSpec(tuple(effects), schema=int(document.get("schema", 0)))
    chain.validate()
    if not chain.effects:
        raise ValueError("a hardware capture must contain at least one active effect")
    return chain


def _scan_audio(path: Path, subtype: str) -> None:
    if subtype not in LOSSLESS_SUBTYPES:
        raise ValueError(f"capture audio must use lossless PCM/float encoding: {path} ({subtype})")
    for block in soundfile.blocks(path, blocksize=65_536, dtype="float32", always_2d=True):
        if not np.isfinite(block).all():
            raise ValueError(f"capture audio contains non-finite samples: {path}")


def validate_manifest(
    path: Path,
    *,
    verify_hashes: bool = True,
    excluded_splits: frozenset[str] = frozenset(),
) -> dict:
    """Validate a manifest without opening audio from explicitly excluded splits.

    Development audits exclude ``locked-final``. Its metadata remains visible for
    split-leakage checks, but its paths are deliberately never resolved, opened,
    scanned, or hashed until the one-shot final evaluation.
    """

    unknown_exclusions = set(excluded_splits) - SPLITS
    if unknown_exclusions:
        raise ValueError(f"unknown excluded capture splits: {sorted(unknown_exclusions)}")
    document = json.loads(path.read_text())
    if document.get("schema") != SCHEMA:
        raise ValueError(f"unsupported capture manifest schema: {document.get('schema')}")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("capture manifest must contain records")

    root = path.parent
    identifiers = set()
    audio_pairs = set()
    audio_splits: dict[str, set[str]] = defaultdict(set)
    program_splits: dict[str, set[str]] = defaultdict(set)
    session_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    counts, inspected_counts, topologies = Counter(), Counter(), Counter()
    total_seconds = 0.0
    for record in records:
        identifier = str(record.get("id", "")).strip()
        if not identifier or identifier in identifiers:
            raise ValueError(f"capture id is empty or duplicated: {identifier!r}")
        identifiers.add(identifier)
        split = record.get("split")
        if split not in SPLITS:
            raise ValueError(f"capture {identifier} has invalid split: {split!r}")
        for field in ("device_id", "session_id", "player_id"):
            if not str(record.get(field, "")).strip():
                raise ValueError(f"capture {identifier} is missing {field}")
        session = (str(record["device_id"]), str(record["session_id"]))
        session_splits[session].add(split)

        chain = _chain(record.get("chain", {}))
        counts[split] += 1
        topologies["+".join(chain.topology)] += 1
        for label in (
            "clean",
            "wet",
            "raw_program",
            "raw_latency_capture",
            "raw_wet_capture",
        ):
            _relative_path(str(record.get(label, "")))
            digest = str(record.get(f"{label}_sha256", ""))
            if not SHA256.fullmatch(digest):
                raise ValueError(f"capture {identifier} has invalid {label} sha256")
            if label in {"clean", "wet"}:
                audio_splits[digest].add(split)
        program_digest = str(record.get("raw_program_sha256", ""))
        program_splits[program_digest].add(split)
        if split in excluded_splits:
            continue
        inspected_counts[split] += 1

        clean = _path(root, str(record.get("clean", "")))
        wet = _path(root, str(record.get("wet", "")))
        pair = (clean.resolve(), wet.resolve())
        if pair in audio_pairs:
            raise ValueError(f"capture audio pair is duplicated: {identifier}")
        audio_pairs.add(pair)
        clean_info, wet_info = soundfile.info(clean), soundfile.info(wet)
        _scan_audio(clean, clean_info.subtype)
        _scan_audio(wet, wet_info.subtype)
        if (
            clean_info.samplerate != wet_info.samplerate
            or clean_info.channels != wet_info.channels
            or clean_info.frames != wet_info.frames
        ):
            raise ValueError(
                f"capture {identifier} is not sample-aligned: "
                f"{clean_info.samplerate}/{clean_info.channels}/{clean_info.frames} vs "
                f"{wet_info.samplerate}/{wet_info.channels}/{wet_info.frames}"
            )
        if clean_info.samplerate != CAPTURE_SAMPLE_RATE:
            raise ValueError(
                f"capture {identifier} must use {CAPTURE_SAMPLE_RATE} Hz, "
                f"got {clean_info.samplerate}"
            )
        if not record.get("latency_compensated", False):
            raise ValueError(f"capture {identifier} must be latency compensated")
        latency = record.get("measured_latency_samples")
        if not isinstance(latency, int) or latency < 0:
            raise ValueError(f"capture {identifier} has invalid measured latency")
        quality = record.get("capture_quality")
        if not isinstance(quality, dict) or not quality.get("passed", False):
            raise ValueError(f"capture {identifier} has no passing capture-quality report")
        expected_quality = {
            "sample_rate": clean_info.samplerate,
            "frames": clean_info.frames,
            "channels": clean_info.channels,
            "measured_latency_samples": latency,
            "clipped_samples": 0,
            "dropout_blocks": 0,
            "automatic_normalization": False,
        }
        for field, expected in expected_quality.items():
            if quality.get(field) != expected:
                raise ValueError(
                    f"capture {identifier} quality field {field} differs: "
                    f"{quality.get(field)!r}/{expected!r}"
                )
        for raw_field in ("raw_program", "raw_latency_capture", "raw_wet_capture"):
            raw_path = _path(root, str(record.get(raw_field, "")))
            raw_info = soundfile.info(raw_path)
            _scan_audio(raw_path, raw_info.subtype)
            if raw_info.samplerate != CAPTURE_SAMPLE_RATE:
                raise ValueError(
                    f"capture {identifier} {raw_field} must use {CAPTURE_SAMPLE_RATE} Hz"
                )
            if verify_hashes:
                expected = record.get(f"{raw_field}_sha256")
                actual = sha256(raw_path)
                if expected != actual:
                    raise ValueError(
                        f"capture {identifier} {raw_field} hash differs: {expected}/{actual}"
                    )
        for label, source in (("clean", clean), ("wet", wet)):
            expected = record.get(f"{label}_sha256")
            if verify_hashes:
                actual = sha256(source)
                if expected != actual:
                    raise ValueError(
                        f"capture {identifier} {label} hash differs: {expected}/{actual}"
                    )
        total_seconds += clean_info.frames / clean_info.samplerate

    leaked = {
        f"{device}/{session}": sorted(splits)
        for (device, session), splits in session_splits.items()
        if len(splits) > 1
    }
    if leaked:
        raise ValueError(f"device sessions cross data splits: {leaked}")
    audio_leaks = {
        digest: sorted(splits)
        for digest, splits in audio_splits.items()
        if len(splits) > 1
    }
    if audio_leaks:
        raise ValueError(f"identical audio crosses data splits: {audio_leaks}")
    program_leaks = {
        digest: sorted(splits)
        for digest, splits in program_splits.items()
        if len(splits) > 1
    }
    if program_leaks:
        raise ValueError(f"identical replay programs cross data splits: {program_leaks}")
    return {
        "schema": SCHEMA,
        "manifest": str(path),
        "manifest_sha256": sha256(path),
        "records": len(records),
        "records_inspected": sum(inspected_counts.values()),
        "duration_hours": total_seconds / 3600.0,
        "splits": dict(sorted(counts.items())),
        "inspected_splits": dict(sorted(inspected_counts.items())),
        "excluded_splits": sorted(excluded_splits),
        "topologies": dict(sorted(topologies.items())),
        "session_disjoint": True,
        "audio_hash_disjoint": True,
        "replay_program_hash_disjoint": True,
        "hashes_verified": verify_hashes,
        "quality_contract": {
            "source_audio_mutation": "forbidden",
            "sample_alignment": "exact after measured-latency compensation",
            "automatic_normalization": False,
            "lossy_reencoding": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--skip-hashes", action="store_true")
    parser.add_argument(
        "--development",
        action="store_true",
        help="do not open or hash locked-final audio",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate_manifest(
                args.manifest,
                verify_hashes=not args.skip_hashes,
                excluded_splits=frozenset({"locked-final"}) if args.development else frozenset(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
