#!/usr/bin/env python3
"""Validate the retained GuitarSet source archive without extracting audio."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile

from .evaluate_phase7_guitarset_ood import EXPECTED_MD5, MEMBER, _md5


def audit(archive_path: Path) -> dict:
    if archive_path.stat().st_size != 683145360 or _md5(archive_path) != EXPECTED_MD5:
        raise ValueError("GuitarSet archive byte count or official MD5 differs")
    groups = Counter()
    subtypes = Counter()
    digests = set()
    total_frames = 0
    peak_maximum = 0.0
    full_scale_members = []
    full_scale_samples = 0
    with zipfile.ZipFile(archive_path) as archive:
        damaged = archive.testzip()
        if damaged is not None:
            raise ValueError(f"GuitarSet archive CRC failed: {damaged}")
        for name in sorted(archive.namelist()):
            match = MEMBER.match(Path(name).name)
            if not match:
                continue
            raw = archive.read(name)
            info = soundfile.info(io.BytesIO(raw))
            if (
                info.samplerate != 44_100
                or info.channels != 1
                or info.subtype not in {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}
            ):
                raise ValueError(f"unexpected lossless GuitarSet geometry: {name}")
            audio, _ = soundfile.read(io.BytesIO(raw), dtype="float32")
            if not np.isfinite(audio).all():
                raise ValueError(f"non-finite GuitarSet audio: {name}")
            digests.add(hashlib.sha256(audio.tobytes()).hexdigest())
            groups[f"{match.group('player')}-{match.group('style')}"] += 1
            subtypes[info.subtype] += 1
            total_frames += len(audio)
            peak_maximum = max(peak_maximum, float(np.abs(audio).max()))
            clipped = int(np.sum(np.abs(audio) >= 32767.0 / 32768.0))
            if clipped:
                full_scale_members.append(name)
                full_scale_samples += clipped
    if sum(groups.values()) != 360 or len(groups) != 12 or len(digests) != 360:
        raise ValueError("GuitarSet file, player/style, or unique-audio counts differ")
    return {
        "schema": 1,
        "dataset": "GuitarSet v1.1.0 mono pickup mix",
        "record_url": "https://zenodo.org/records/3371780",
        "doi": "10.5281/zenodo.3371780",
        "license": "CC-BY-4.0",
        "license_reference": "https://zenodo.org/api/records/3371780 (metadata.license.id=cc-by-4.0)",
        "attribution": "Qingyang Xi, Rachel M. Bittner, Johan Pauwels, Xuzhou Ye, Juan P. Bello",
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_md5": EXPECTED_MD5,
        "zip_crc_passed": True,
        "files": sum(groups.values()),
        "unique_audio_hashes": len(digests),
        "player_style_counts": dict(sorted(groups.items())),
        "sample_rate": 44_100,
        "channels": 1,
        "subtypes": dict(sorted(subtypes.items())),
        "duration_hours": total_frames / 44_100.0 / 3600.0,
        "source_peak_maximum": peak_maximum,
        "full_scale_hit_members": full_scale_members,
        "full_scale_hit_samples": full_scale_samples,
        "source_amplitude_calibration": "not hardware-referenced; full-scale hits are not proof of sustained clipping",
        "source_archive_read_only": True,
        "audio_extracted_to_disk": False,
        "normalization_applied_by_muspector": False,
        "has_cs3_wet_truth": False,
        "physical_audio_devices_used": False,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.archive)
    if args.output.exists():
        raise ValueError(f"GuitarSet source audit already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
