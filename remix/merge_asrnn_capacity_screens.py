#!/usr/bin/env python3
"""Merge the fixed eight-candidate capacity experiment without editing its reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .screen_asrnn_capacity import selection_key


def merge(paths: list[Path], output: Path) -> dict:
    if output.exists():
        raise ValueError(f"capacity pool already exists: {output}")
    screens = [json.loads(path.read_text()) for path in paths]
    if not screens:
        raise ValueError("capacity pool is empty")
    for key in ("archive_sha256", "group_audit_sha256", "calibration_files"):
        if len({screen[key] for screen in screens}) != 1:
            raise ValueError(f"capacity screens disagree on {key}")
    rows = [row for screen in screens for row in screen["results"]]
    expected = {(layers, seed, loss) for layers, seed in ((1, 1), (4, 1), (4, 2), (4, 3)) for loss in ("GFB", "MAE")}
    actual = {
        (row["layers"], int(Path(row["member"]).stem.rsplit("-", 1)[1]), Path(row["member"]).name.split("-")[3])
        for row in rows
    }
    if len(rows) != len(expected) or actual != expected or any(row["hidden_size"] != 64 for row in rows):
        raise ValueError("fixed capacity experiment is incomplete or contains duplicates")
    report = {
        **{key: value for key, value in screens[0].items() if key not in ("results", "selected")},
        "phase": "phase-8-eight-candidate-capacity-pool",
        "screen_complete": True,
        "expected_candidates": len(expected),
        "source_screens": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        "results": rows,
        "selected": min(rows, key=lambda row: selection_key(row["calibration"])),
        "completion_basis": "all eight expected architecture/seed/loss results present; historical reports preserved unchanged",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screens", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = merge(args.screens, args.output)
    print(json.dumps({"candidates": len(report["results"]), "screen_complete": report["screen_complete"]}))


if __name__ == "__main__":
    main()
