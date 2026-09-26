"""Ephemeral, research-only Open Riff Box stage-tap materialization."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .license_gate import authorize_product_weights, load_registry


RATE = 44_100
SOURCE_ID = "open-riff-box-stage-renderer"
PINNED_COMMIT = "c980ae874c87e16835d87b265ea58079fc69e7f5"
STAGE_NAMES = (
    "clean-input",
    "v1a",
    "v1b",
    "v2a",
    "v2b",
    "v3a",
    "cathode-follower",
    "tone-stack",
    "power-amp",
    "output-transformer",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def settings(seed: int, count: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        rows.append({
            "gain": rng.uniform(0.30, 1.0),
            "bass": rng.uniform(0.20, 0.80),
            "mid": rng.uniform(0.20, 0.80),
            "treble": rng.uniform(0.20, 0.80),
            "speaker_drive": rng.uniform(0.05, 0.80),
            "ov_level": rng.uniform(0.35, 0.95),
            "master": rng.uniform(0.40, 0.90),
            "gain_mode": index % 2,
            "channel": "od",
            "input_jack": "high" if index % 3 else "low",
        })
    return rows


def render_command(binary: Path, source: Path, output: Path, row: dict, stage: int) -> list[str]:
    if stage not in range(1, 10):
        raise ValueError("Open Riff Box stage tap must be 1..9")
    return [
        str(binary), "--process-file", str(source), str(output),
        "--engine", "platinum",
        "--gain", str(row["gain"]),
        "--bass", str(row["bass"]),
        "--mid", str(row["mid"]),
        "--treble", str(row["treble"]),
        "--speaker-drive", str(row["speaker_drive"]),
        "--ov-level", str(row["ov_level"]),
        "--master", str(row["master"]),
        "--gain-mode", str(row["gain_mode"]),
        "--channel", row["channel"],
        "--input-jack", row["input_jack"],
        "--cabinet", "200",
        "--stage-limit", str(stage),
        "--plat-diag", "noiselevel=0",
    ]


def _active_segment(path: Path, frames: int) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != RATE or len(audio) < frames:
        raise ValueError(f"invalid Clean source geometry: {path}")
    mono = audio.mean(1)
    starts = np.linspace(0, len(mono) - frames, 12, dtype=np.int64)
    start = max(starts, key=lambda value: float(np.square(mono[value:value + frames]).mean()))
    return mono[start:start + frames].astype(np.float32, copy=False)


def prepare(
    workspace: Path,
    binary: Path,
    source_tree: Path,
    output: Path,
    *,
    count: int = 12,
    seconds: float = 2.0,
    seed: int = 20260905,
) -> dict:
    registry = load_registry(workspace / "remix/data_sources.json")
    if authorize_product_weights(registry, (SOURCE_ID,))["authorized"]:
        raise RuntimeError("research renderer must remain blocked from product weights")
    commit = subprocess.check_output(
        ["git", "-C", str(source_tree), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PINNED_COMMIT:
        raise ValueError(f"unexpected Open Riff Box commit: {commit}")
    if not binary.is_file():
        raise FileNotFoundError(binary)

    contract = json.loads((workspace / "remix/egdb_pg_subset_v1.json").read_text())
    tracks = contract["tracks"]["fit"][:count]
    rows = settings(seed, len(tracks))
    frames = int(round(seconds * RATE))
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, (track, controls) in enumerate(zip(tracks, rows, strict=True)):
        sample_dir = output / f"sample-{index:03d}"
        sample_dir.mkdir(exist_ok=True)
        clean = _active_segment(
            workspace / "data/corpus/egdb-pg-subset-v1/clean" / f"{track}.wav", frames
        )
        clean_path = sample_dir / "stage-0.wav"
        sf.write(clean_path, clean, RATE, subtype="FLOAT")
        stage_paths = [clean_path]
        for stage in range(1, 10):
            stage_path = sample_dir / f"stage-{stage}.wav"
            result = subprocess.run(
                render_command(binary, clean_path, stage_path, controls, stage),
                check=True,
                capture_output=True,
                text=True,
            )
            if "Done! Wrote" not in result.stdout or not stage_path.is_file():
                raise RuntimeError(f"renderer did not produce stage {stage} for track {track}")
            rendered, rendered_rate = sf.read(stage_path, dtype="float32", always_2d=True)
            if rendered_rate != RATE or len(rendered) != frames or not np.isfinite(rendered).all():
                raise ValueError(f"invalid rendered stage: {stage_path}")
            mono = rendered.mean(1).astype(np.float32, copy=False)
            sf.write(stage_path, mono, RATE, subtype="FLOAT")
            stage_paths.append(stage_path)
        samples.append({
            "sample": index,
            "track": track,
            "split": "fit" if index < max(1, len(tracks) - 3) else "calibration",
            "controls": controls,
            "stages": [str(path.relative_to(output)) for path in stage_paths],
            "sha256": [sha256(path) for path in stage_paths],
        })

    manifest = {
        "schema": 1,
        "status": "research-only-stage-supervision-not-product-gradient",
        "source_id": SOURCE_ID,
        "source_commit": commit,
        "binary_sha256": sha256(binary),
        "sample_rate": RATE,
        "seconds": seconds,
        "stage_names": list(STAGE_NAMES),
        "cabinet_assets_used": False,
        "thermal_noise_enabled": False,
        "product_weight_eligible": False,
        "samples": samples,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
