#!/usr/bin/env python3
"""Package two fixed Product4 Reverb v3 development examples as compact A/B audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import soundfile
import torch

from .ambience4 import AmbiencePairsV4MixedFrequencyProfileBank
from .ambience_model4 import AmbienceFrequencyProfileBankExpert
from .train_ambience4_frequency_profile import SEED


FIXED_EXAMPLES = (
    (2, "01-but-measured-shaping-ab.wav"),
    (17, "02-openslr-simulated-shaping-ab.wav"),
)
SILENCE_FRAMES = 36_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/foundation/product4-reverb-frequency-profile-v3-mixed-formal/"
            "ambience/model.pt"
        ),
    )
    parser.add_argument(
        "--runtime-audit",
        type=Path,
        default=Path("runs/foundation/product4-reverb-frequency-runtime-v1/metrics.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demos/reverb-frequency-profile-v3-ab-20260905-v1"),
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    archive = output.with_suffix(".zip")
    if output.exists() or archive.exists():
        raise FileExistsError("refusing to replace an existing Reverb acceptance package")
    runtime = json.loads(args.runtime_audit.read_text())
    if not runtime["accepted"] or runtime["checkpoint"]["sha256"] != _sha256(checkpoint):
        raise PermissionError("Reverb runtime gate is absent or does not bind this checkpoint")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    architecture = payload["architecture"]
    model = AmbienceFrequencyProfileBankExpert(
        channels=architecture["channels"], depth=architecture["depth"],
        n_fft=architecture["n_fft"], hop=architecture["hop"],
    ).eval()
    model.load_state_dict(payload["state_dict"])
    dataset = AmbiencePairsV4MixedFrequencyProfileBank(
        workspace, "development", 24, 65_536, SEED
    )
    output.mkdir(parents=True)
    files = {}
    silence = np.zeros(SILENCE_FRAMES, dtype=np.float32)
    for index, filename in FIXED_EXAMPLES:
        row = dataset[index]
        if row["profile_mode"] != "frequency-shaping-bank":
            raise RuntimeError(f"fixed demo row {index} no longer exercises shaping")
        with torch.inference_mode():
            restored, _ = model(
                row["wet"][None], row["controls"][None], row["late_base"][None]
            )
        wet = row["wet"].numpy()
        candidate = restored[0].numpy()
        if not np.isfinite(candidate).all() or float(np.max(np.abs(candidate))) > 1.05:
            raise RuntimeError(f"fixed demo row {index} is unsafe")
        path = output / filename
        soundfile.write(
            path, np.concatenate((wet, silence, candidate)), 48_000, subtype="FLOAT"
        )
        files[filename] = {
            "fixed_development_index": index,
            "selection": "fixed by source/domain and shaping mode, never by quality score",
            "A": "Wet",
            "B": "Restored v3",
            "source_id": row["source_id"],
            "rir_source_id": row["rir_source_id"],
            "rir": row["rir"],
            "profile_mode": row["profile_mode"],
            "controls": row["control_values"],
            "wet_peak": float(np.max(np.abs(wet))),
            "restored_peak": float(np.max(np.abs(candidate))),
            "wet_to_restored_rms": float(np.sqrt(np.mean(np.square(candidate - wet)))),
            "duration_seconds": (2 * len(wet) + SILENCE_FRAMES) / 48_000.0,
        }
    manifest = {
        "schema": 1,
        "title": "Product4 Reverb frequency-profile v3 compact AB demo",
        "ab_order": ["A: Wet", "0.75-second silence", "B: Restored v3"],
        "sample_rate_hz": 48_000,
        "subtype": "FLOAT",
        "normalization": False,
        "checkpoint_sha256": _sha256(checkpoint),
        "runtime_audit": str(args.runtime_audit),
        "automatic_development_gate_passed": True,
        "locked_final_accessed": False,
        "human_acceptance": "pending",
        "files": files,
    }
    manifest_path = output / "demo.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    package_files = sorted(output.iterdir())
    checksums = "".join(f"{_sha256(path)}  {path.name}\n" for path in package_files)
    (output / "SHA256SUMS.txt").write_text(checksums)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(output.iterdir()):
            bundle.write(path, Path(output.name) / path.name)
    print(json.dumps({
        "output": str(output),
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
        "files": sorted(files),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
