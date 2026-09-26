#!/usr/bin/env python3
"""Extract the EG-IPT SM57 Amp+cab channel from audited ZIP byte ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path, PurePosixPath


ARCHIVE_BYTES = 23_757_983_313
ARCHIVE_MD5 = "48a5135adfd090515ff0af7dc5c3c32f"
CENTRAL_START = 23_741_203_027
CENTRAL_BYTES = 16_780_188
EXPECTED_RANGES = (
    (3_951_238_180, 5_283_173_107),
    (11_748_502_523, 13_047_869_317),
    (19_673_248_532, 21_045_727_216),
)


def _zip64_values(extra: bytes) -> list[int]:
    at = 0
    while at + 4 <= len(extra):
        tag, size = struct.unpack_from("<HH", extra, at)
        value = extra[at + 4:at + 4 + size]
        if tag == 1:
            return list(struct.unpack("<" + "Q" * (len(value) // 8), value))
        at += 4 + size
    return []


def _central_rows(path: Path) -> list[dict]:
    data = path.read_bytes()
    if len(data) != CENTRAL_BYTES:
        raise ValueError(f"central directory must be {CENTRAL_BYTES} bytes")
    rows = []
    at = 0
    while at < len(data):
        if data[at:at + 4] != b"PK\x01\x02":
            raise ValueError(f"bad central-directory signature at {at}")
        fields = struct.unpack_from("<4s6H3L5H2L", data, at)
        method, crc32 = fields[4], fields[7]
        compressed, uncompressed = fields[8], fields[9]
        name_len, extra_len, comment_len = fields[10:13]
        local_offset = fields[16]
        name_at = at + 46
        name = data[name_at:name_at + name_len].decode("utf-8")
        extra = data[name_at + name_len:name_at + name_len + extra_len]
        values = iter(_zip64_values(extra))
        if uncompressed == 0xFFFFFFFF:
            uncompressed = next(values)
        if compressed == 0xFFFFFFFF:
            compressed = next(values)
        if local_offset == 0xFFFFFFFF:
            local_offset = next(values)
        rows.append({
            "name": name, "method": method, "crc32": crc32,
            "compressed": compressed, "uncompressed": uncompressed,
            "local_offset": local_offset,
        })
        at = name_at + name_len + extra_len + comment_len
    return rows


def _part_for(offset: int, parts: list[tuple[int, int, Path]]) -> tuple[int, Path]:
    for start, end, path in parts:
        if start <= offset <= end:
            return start, path
    raise ValueError(f"member offset {offset} is outside admitted ranges")


def extract(args: argparse.Namespace) -> dict:
    ranges = list(EXPECTED_RANGES)
    if len(args.part) != len(ranges):
        raise ValueError(f"exactly {len(ranges)} --part files are required")
    parts = []
    for (start, end), path in zip(ranges, args.part, strict=True):
        path = path.resolve()
        expected = end - start + 1
        if path.stat().st_size != expected:
            raise ValueError(f"{path} must contain range {start}-{end} ({expected} bytes)")
        parts.append((start, end, path))

    selected = [
        row for row in _central_rows(args.central_directory.resolve())
        if row["name"].endswith("_dyn.wav")
        and row["name"].startswith("EG-IPT/")
        and "/._" not in row["name"]
    ]
    if len(selected) != 8_717:
        raise ValueError(f"expected 8717 real SM57 members, found {len(selected)}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    handles = {path: path.open("rb") for _, _, path in parts}
    aggregate = hashlib.sha256()
    total_uncompressed = 0
    try:
        for index, row in enumerate(selected, 1):
            base, part = _part_for(row["local_offset"], parts)
            handle = handles[part]
            handle.seek(row["local_offset"] - base)
            header = handle.read(30)
            fields = struct.unpack("<4s5H3L2H", header)
            if fields[0] != b"PK\x03\x04" or fields[3] != row["method"]:
                raise ValueError(f"bad local header for {row['name']}")
            name_len, extra_len = fields[-2:]
            local_name = handle.read(name_len).decode("utf-8")
            handle.seek(extra_len, 1)
            if local_name != row["name"]:
                raise ValueError(f"central/local name mismatch: {row['name']}")
            compressed = handle.read(row["compressed"])
            if len(compressed) != row["compressed"]:
                raise ValueError(f"truncated compressed member: {row['name']}")
            if row["method"] == 0:
                payload = compressed
            elif row["method"] == 8:
                payload = zlib.decompress(compressed, -15)
            else:
                raise ValueError(f"unsupported ZIP method {row['method']}")
            if len(payload) != row["uncompressed"]:
                raise ValueError(f"uncompressed size mismatch: {row['name']}")
            if zlib.crc32(payload) & 0xFFFFFFFF != row["crc32"]:
                raise ValueError(f"CRC mismatch: {row['name']}")
            relative = PurePosixPath(row["name"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe member path: {row['name']}")
            target = output.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            aggregate.update(row["name"].encode())
            aggregate.update(bytes.fromhex(digest))
            total_uncompressed += len(payload)
            if index % 500 == 0:
                print(json.dumps({"extracted": index, "total": len(selected)}), flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    report = {
        "schema": 1,
        "source_id": "eg-ipt",
        "record_url": "https://zenodo.org/records/15205644",
        "license": "CC-BY-4.0",
        "archive": {"bytes": ARCHIVE_BYTES, "md5": ARCHIVE_MD5},
        "selection": "SM57 dyn close microphone at 2.5 cm",
        "members": len(selected),
        "uncompressed_bytes": total_uncompressed,
        "member_manifest_sha256": aggregate.hexdigest(),
        "ranges": [
            {"start": start, "end": end, "bytes": end - start + 1}
            for start, end in ranges
        ],
        "cabinet": "Mesa 4x12 with Celestion V30 UK speakers",
        "amplifier": "EVH 5150 III 50W 6L6, controls flat, reverb off",
        "admission": "extracted-for-pair-alignment-and-quality-audit-only",
    }
    (output / "eg_ipt_sm57_extract.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--central-directory", type=Path, required=True)
    parser.add_argument("--part", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    extract(parser.parse_args())


if __name__ == "__main__":
    main()
