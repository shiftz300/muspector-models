"""Materialize a small product-eligible rusty-amp Marshall stage corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .license_gate import require_product_uses
from .prepare_rusty_amp_renderer import PINNED_COMMIT


RATE = 44_100
RENDERER_ID = "rusty-amp-stage-renderer"
CLEAN_ID = "guitarjam"
STAGE_NAMES = (
    "clean-input",
    "front-end",
    "preamp",
    "tone-stack",
    "voice-balance",
    "power-amp",
    "output-transformer",
    "speaker-load",
    "output",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _active_segment(path: Path, frames: int) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != RATE or audio.shape[1] != 1 or len(audio) < frames:
        raise ValueError(f"invalid GuitarJam source geometry: {path}")
    mono = audio[:, 0]
    starts = np.linspace(0, len(mono) - frames, 12, dtype=np.int64)
    start = max(starts, key=lambda value: float(np.square(mono[value:value + frames]).mean()))
    return mono[start:start + frames].astype(np.float32, copy=True)


def _settings(seed: int, count: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    rows = []
    gains = (0.28, 0.48, 0.68, 0.88)
    for index in range(count):
        rows.append({
            "gain": gains[index % len(gains)],
            "bass": rng.uniform(0.15, 0.85),
            "mid": rng.uniform(0.15, 0.85),
            "treble": rng.uniform(0.15, 0.85),
            "presence": rng.uniform(0.15, 0.85),
            "master": rng.uniform(0.38, 0.72),
        })
    return rows


def prepare(
    workspace: Path,
    source_tree: Path,
    binary: Path,
    output: Path,
    *,
    fit_count: int,
    calibration_count: int,
    seconds: float,
    seed: int,
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace stage corpus: {output}")
    authorization = require_product_uses(
        workspace / "remix/data_sources.json",
        {RENDERER_ID: "train-amp", CLEAN_ID: "product-clean-source"},
    )
    commit = subprocess.check_output(
        ["git", "-C", str(source_tree), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PINNED_COMMIT or not binary.is_file():
        raise ValueError("rusty-amp renderer is missing or not pinned")
    root = workspace / "data/corpus/guitarjam/guitar_jam"
    files = (
        [("fit", path) for path in sorted((root / "train").glob("*.wav"))[:fit_count]]
        + [("calibration", path) for path in sorted((root / "val").glob("*.wav"))[:calibration_count]]
    )
    if len(files) != fit_count + calibration_count:
        raise ValueError("insufficient disjoint GuitarJam source files")
    controls = _settings(seed, len(files))
    frames = round(seconds * RATE)
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, ((split, source), row) in enumerate(zip(files, controls, strict=True)):
        sample_dir = output / f"sample-{index:03d}"
        sample_dir.mkdir()
        clean = _active_segment(source, frames)
        sf.write(sample_dir / "stage-0.wav", clean, RATE, subtype="FLOAT")
        command = [
            str(binary), str(sample_dir / "stage-0.wav"), str(sample_dir),
            *(str(row[name]) for name in ("gain", "bass", "mid", "treble", "presence", "master")),
        ]
        subprocess.run(command, check=True)
        stage_paths = [sample_dir / f"stage-{stage}.wav" for stage in range(9)]
        hashes = []
        values = []
        for path in stage_paths:
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
            if rate != RATE or audio.shape != (frames, 1) or not np.isfinite(audio).all():
                raise ValueError(f"invalid rendered stage: {path}")
            hashes.append(_sha256(path))
            values.append(audio[:, 0])
        if len(set(hashes)) != len(hashes):
            raise ValueError(f"duplicate stage output for sample {index}")
        effect_rms = float(np.sqrt(np.mean(np.square(values[-1] - values[0]))))
        clean_rms = float(np.sqrt(np.mean(np.square(values[0]))))
        if effect_rms < 0.10 * max(clean_rms, 1.0e-5):
            raise ValueError(f"rendered Amp effect is too weak for sample {index}")
        samples.append({
            "sample": index,
            "split": split,
            "clean_source": str(source.relative_to(workspace)),
            "clean_source_sha256": _sha256(source),
            "controls": row,
            "stages": [str(path.relative_to(output)) for path in stage_paths],
            "sha256": hashes,
            "effect_rms": effect_rms,
        })
    manifest = {
        "schema": 1,
        "status": "product-eligible-graybox-pretraining",
        "source_ids": [RENDERER_ID, CLEAN_ID],
        "authorization": authorization,
        "source_commit": commit,
        "renderer_binary_sha256": _sha256(binary),
        "sample_rate": RATE,
        "seconds": seconds,
        "stage_names": list(STAGE_NAMES),
        "cabinet_assets_used": False,
        "external_plugins_used": False,
        "product_weight_eligible": True,
        "samples": samples,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-count", type=int, default=24)
    parser.add_argument("--calibration-count", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    report = prepare(
        args.workspace.resolve(), args.source_tree.resolve(), args.binary.resolve(),
        args.output.resolve(), fit_count=args.fit_count,
        calibration_count=args.calibration_count, seconds=args.seconds, seed=args.seed,
    )
    print(json.dumps({
        "status": report["status"], "samples": len(report["samples"]),
        "renderer_binary_sha256": report["renderer_binary_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
