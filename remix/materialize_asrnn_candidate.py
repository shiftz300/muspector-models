#!/usr/bin/env python3
"""Materialize one calibration-screened candidate without changing its weights."""

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from .import_asrnn_effect import import_checkpoint


def materialize(screen_path: Path, archive_path: Path, member: str, output: Path) -> dict:
    screen = json.loads(screen_path.read_text())
    rows = [row for row in screen["results"] if row["member"] == member]
    if len(rows) != 1 or not rows[0]["calibration"]["passes_selection_gate"]:
        raise ValueError("candidate must already pass the frozen calibration screen")
    if output.exists():
        raise ValueError(f"candidate output exists: {output}")
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != screen["archive_sha256"]:
        raise ValueError("source archive differs from the screened archive")
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(prefix="muspector-candidate-") as temporary:
        source = Path(archive.extract(member, temporary))
        converted = Path(temporary) / "converted.pt"
        report = import_checkpoint(source, converted, "cs3")
        if report["output_sha256"] != rows[0]["checkpoint_sha256"]:
            raise ValueError("materialized checkpoint differs from calibration bytes")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(converted, output)
    return {"checkpoint": str(output), "checkpoint_sha256": rows[0]["checkpoint_sha256"], "member": member}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=Path("data/downloads/asrnn-results-20406285.zip"))
    parser.add_argument("--member", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.screen, args.archive, args.member, args.output)))


if __name__ == "__main__":
    main()
