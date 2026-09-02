"""Audit the admitted Multimodal Electric Guitar line-output audio."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile


def audit(root: Path) -> dict:
    root = root.resolve()
    paths = sorted(root.rglob("*.wav"))
    if not paths:
        raise ValueError(f"no WAV files below {root}")
    formats: Counter[tuple[int, int, str]] = Counter()
    players: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    total_frames = 0
    nonfinite = 0
    clipped = 0
    file_peaks = []
    file_rms = []
    for path in paths:
        relative = path.relative_to(root)
        parts = relative.parts
        if len(parts) < 4 or parts[0] != "guit_raw_dir":
            raise ValueError(f"unexpected corpus path: {relative}")
        players[parts[1]] += 1
        tasks[parts[2]] += 1
        info = soundfile.info(path)
        formats[(info.samplerate, info.channels, info.subtype)] += 1
        total_frames += info.frames
        peak = 0.0
        square_sum = 0.0
        count = 0
        with soundfile.SoundFile(path) as stream:
            for block in stream.blocks(blocksize=262_144, dtype="float32", always_2d=True):
                finite = np.isfinite(block)
                nonfinite += int(block.size - np.count_nonzero(finite))
                safe = np.where(finite, block, 0.0).astype(np.float64)
                magnitude = np.abs(safe)
                peak = max(peak, float(np.max(magnitude, initial=0.0)))
                clipped += int(np.count_nonzero(magnitude >= (32767.0 / 32768.0)))
                square_sum += float(np.sum(safe * safe))
                count += safe.size
        file_peaks.append(peak)
        file_rms.append(math.sqrt(square_sum / max(1, count)))
    sample_rates = {rate for rate, _, _ in formats}
    hours = sum(
        soundfile.info(path).frames / soundfile.info(path).samplerate for path in paths
    ) / 3600.0
    return {
        "schema": 1,
        "source": "multimodal-electric-guitar-data",
        "root": str(root),
        "files": len(paths),
        "players_in_archive": sorted(players),
        "player_count_in_archive": len(players),
        "published_complete_participants": 31,
        "player_file_counts": dict(sorted(players.items())),
        "task_file_counts": dict(sorted(tasks.items())),
        "formats": [
            {"sample_rate": rate, "channels": channels, "subtype": subtype, "files": count}
            for (rate, channels, subtype), count in sorted(formats.items())
        ],
        "total_frames": total_frames,
        "hours": hours,
        "sample_rates": sorted(sample_rates),
        "nonfinite_samples": nonfinite,
        "clipped_samples": clipped,
        "file_peak": {
            "minimum": float(np.min(file_peaks)),
            "median": float(np.median(file_peaks)),
            "maximum": float(np.max(file_peaks)),
        },
        "file_rms": {
            "minimum": float(np.min(file_rms)),
            "median": float(np.median(file_rms)),
            "maximum": float(np.max(file_rms)),
        },
        "source_audio_modified": False,
        "automatic_normalization": False,
        "lossy_reencoding": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    value = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(value)
    print(value, end="")


if __name__ == "__main__":
    main()
