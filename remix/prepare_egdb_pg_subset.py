"""Plan and verify a byte-range-only EGDB-PG Amp+cab subset.

The 135 GB upstream ZIP stores already-compressed audio entries without ZIP
compression.  This utility reads a separately fetched central directory,
freezes exact byte ranges, and emits a curl config.  It never requests the
locked-final tracks or profiles declared in the subset contract.
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

import soundfile as sf


ARCHIVE_URL = "https://zenodo.org/records/19789500/files/EGDB_FLAC_bundle.zip?download=1"
CENTRAL_HEADER = struct.Struct("<4s6H3I5H2I")


def _zip64_values(extra: bytes, compressed: int, uncompressed: int, offset: int) -> tuple[int, int, int]:
    position = 0
    while position + 4 <= len(extra):
        field_id, field_length = struct.unpack_from("<HH", extra, position)
        field = extra[position + 4 : position + 4 + field_length]
        if field_id == 1:
            field_position = 0
            if uncompressed == 0xFFFFFFFF:
                uncompressed = struct.unpack_from("<Q", field, field_position)[0]
                field_position += 8
            if compressed == 0xFFFFFFFF:
                compressed = struct.unpack_from("<Q", field, field_position)[0]
                field_position += 8
            if offset == 0xFFFFFFFF:
                offset = struct.unpack_from("<Q", field, field_position)[0]
            return compressed, uncompressed, offset
        position += 4 + field_length
    return compressed, uncompressed, offset


def parse_central_directory(path: Path) -> dict[str, dict]:
    payload = path.read_bytes()
    position = 0
    rows = {}
    while position < len(payload):
        values = CENTRAL_HEADER.unpack_from(payload, position)
        if values[0] != b"PK\x01\x02":
            raise ValueError(f"invalid central-directory signature at {position}")
        compressed, uncompressed = values[8], values[9]
        name_length, extra_length, comment_length = values[10:13]
        offset = values[16]
        start = position + CENTRAL_HEADER.size
        name_bytes = payload[start : start + name_length]
        extra = payload[start + name_length : start + name_length + extra_length]
        compressed, uncompressed, offset = _zip64_values(
            extra, compressed, uncompressed, offset
        )
        name = name_bytes.decode("utf-8")
        if name in rows:
            raise ValueError(f"duplicate ZIP entry: {name}")
        rows[name] = {
            "name": name,
            "name_bytes": len(name_bytes),
            "compression": values[4],
            "crc32": values[7],
            "compressed_bytes": compressed,
            "uncompressed_bytes": uncompressed,
            "local_header_offset": offset,
        }
        position = start + name_length + extra_length + comment_length
    return rows


def _materialized_rows(
    contract: dict,
    directory: dict[str, dict],
    output_root: Path,
    include_fresh_validation: bool = False,
    include_fresh_validation_v2: bool = False,
) -> list[dict]:
    selected = []
    clean_tracks = sorted(
        set(contract["tracks"]["fit"])
        | set(contract["tracks"]["calibration"])
        | set(contract["tracks"]["development"])
    )
    if include_fresh_validation:
        clean_tracks = sorted(
            set(clean_tracks)
            | set(contract["tracks"]["fresh_validation_not_downloaded"])
        )
    if include_fresh_validation_v2:
        clean_tracks = sorted(
            set(clean_tracks)
            | set(contract["tracks"]["fresh_validation_v2_not_downloaded"])
        )
    for track in clean_tracks:
        name = f"EGDB-PG_flac_full/audio_DI/{track}.wav"
        selected.append(_range_row(directory[name], output_root / "clean" / f"{track}.wav", "clean", track, None))

    split_groups = [
        ("fit", ("fit_calibration", "fit_calibration_additional")),
        ("calibration", ("fit_calibration", "fit_calibration_additional")),
        ("development", ("development",)),
    ]
    if include_fresh_validation:
        split_groups.append(
            ("fresh_validation", ("fresh_validation_not_downloaded",))
        )
    if include_fresh_validation_v2:
        split_groups.append(
            ("fresh_validation_v2", ("fresh_validation_v2_not_downloaded",))
        )
    for split, profile_groups in split_groups:
        track_group = {
            "fresh_validation": "fresh_validation_not_downloaded",
            "fresh_validation_v2": "fresh_validation_v2_not_downloaded",
        }.get(split, split)
        profiles = {}
        for profile_group in profile_groups:
            for category, values in contract["profiles"].get(profile_group, {}).items():
                profiles.setdefault(category, []).extend(
                    values if isinstance(values, list) else [values]
                )
        for category, category_profiles in profiles.items():
            for profile in category_profiles:
                for track in contract["tracks"][track_group]:
                    name = (
                        "EGDB-PG_flac_full/amplifier_rendered_audio/"
                        f"{profile}/{track}_output.flac"
                    )
                    target = output_root / split / category / profile / f"{track}_output.flac"
                    selected.append(_range_row(directory[name], target, split, track, category))
    return selected


def _range_row(entry: dict, target: Path, split: str, track: int, category: str | None) -> dict:
    if entry["compression"] != 0 or entry["compressed_bytes"] != entry["uncompressed_bytes"]:
        raise ValueError(f"entry is not ZIP-stored: {entry['name']}")
    # Four independently probed entries from each path family have local extra
    # length zero.  CRC and audio-structure verification fail closed if that
    # invariant changes for any selected entry.
    data_start = entry["local_header_offset"] + 30 + entry["name_bytes"]
    return {
        "source_name": entry["name"],
        "target": str(target.resolve()),
        "split": split,
        "track": track,
        "category": category,
        "range_start": data_start,
        "range_end": data_start + entry["compressed_bytes"] - 1,
        "bytes": entry["compressed_bytes"],
        "crc32": f"{entry['crc32']:08x}",
    }


def plan(
    central_directory: Path,
    contract_path: Path,
    output_root: Path,
    include_fresh_validation: bool = False,
    include_fresh_validation_v2: bool = False,
) -> dict:
    contract = json.loads(contract_path.read_text())
    directory = parse_central_directory(central_directory)
    rows = _materialized_rows(
        contract,
        directory,
        output_root,
        include_fresh_validation,
        include_fresh_validation_v2,
    )
    locked_profiles = set(contract["profiles"]["locked_final_not_downloaded"].values())
    locked_tracks = set(contract["tracks"]["locked_final_not_downloaded"])
    if any(
        any(profile in row["source_name"] for profile in locked_profiles)
        or row["track"] in locked_tracks
        for row in rows
    ):
        raise ValueError("locked-final source leaked into materialization plan")
    output_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        Path(row["target"]).parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "source_id": contract["source_id"],
        "source_contract": str(contract_path.resolve()),
        "archive_url": ARCHIVE_URL,
        "archive": contract["archive"],
        "locked_final_downloaded": False,
        "fresh_validation_downloaded": include_fresh_validation,
        "fresh_validation_v2_downloaded": include_fresh_validation_v2,
        "files": rows,
        "file_count": len(rows),
        "download_bytes": sum(row["bytes"] for row in rows),
    }
    pending = []
    for row in rows:
        path = Path(row["target"])
        if path.is_file() and path.stat().st_size == row["bytes"]:
            if f"{zlib.crc32(path.read_bytes()) & 0xFFFFFFFF:08x}" == row["crc32"]:
                continue
        pending.append(row)
    config_path = output_root / "curl-ranges.conf"
    with config_path.open("w") as handle:
        for index, row in enumerate(pending):
            if index:
                handle.write("next\n")
            handle.write("fail\n")
            handle.write("silent\n")
            handle.write("show-error\n")
            handle.write("retry = 6\n")
            handle.write("retry-delay = 3\n")
            handle.write("retry-max-time = 120\n")
            handle.write("retry-all-errors\n")
            handle.write("remove-on-error\n")
            handle.write(f'url = "{ARCHIVE_URL}"\n')
            handle.write(f'range = "{row["range_start"]}-{row["range_end"]}"\n')
            handle.write(f'output = "{row["target"]}"\n')
    manifest["pending_file_count"] = len(pending)
    manifest["pending_download_bytes"] = sum(row["bytes"] for row in pending)
    manifest_path = output_root / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(output_root: Path) -> dict:
    manifest = json.loads((output_root / "download_manifest.json").read_text())
    failures = []
    audio = {}
    for row in manifest["files"]:
        path = Path(row["target"])
        if not path.is_file():
            failures.append(f"missing:{path}")
            continue
        payload = path.read_bytes()
        if len(payload) != row["bytes"]:
            failures.append(f"size:{path}")
            continue
        if f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}" != row["crc32"]:
            failures.append(f"crc32:{path}")
            continue
        info = sf.info(path)
        if info.samplerate != 44100 or info.channels != 1:
            failures.append(f"format:{path}")
            continue
        audio[row["target"]] = info.frames
    clean_frames = {
        row["track"]: audio.get(row["target"])
        for row in manifest["files"]
        if row["split"] == "clean"
    }
    for row in manifest["files"]:
        if row["split"] != "clean" and audio.get(row["target"]) != clean_frames[row["track"]]:
            failures.append(f"frame-pair:{row['target']}")
    profile_ids = {
        Path(row["target"]).parent.name
        for row in manifest["files"] if row["split"] != "clean"
    }
    report = {
        "passed": not failures,
        "file_count": manifest["file_count"],
        "verified_files": manifest["file_count"] - len(failures),
        "download_bytes": manifest["download_bytes"],
        "wet_profile_count": len(profile_ids),
        "locked_final_downloaded": manifest["locked_final_downloaded"],
        "fresh_validation_downloaded": manifest.get(
            "fresh_validation_downloaded", False
        ),
        "fresh_validation_v2_downloaded": manifest.get(
            "fresh_validation_v2_downloaded", False
        ),
        "failures": failures,
    }
    (output_root / "audio_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "verify"))
    parser.add_argument("--central-directory", type=Path)
    parser.add_argument(
        "--contract", type=Path, default=Path(__file__).with_name("egdb_pg_subset_v1.json")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--include-fresh-validation", action="store_true")
    parser.add_argument("--include-fresh-validation-v2", action="store_true")
    args = parser.parse_args()
    if args.mode == "plan":
        if args.central_directory is None:
            parser.error("plan requires --central-directory")
        result = plan(
            args.central_directory,
            args.contract,
            args.output_root,
            args.include_fresh_validation,
            args.include_fresh_validation_v2,
        )
    else:
        result = verify(args.output_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
