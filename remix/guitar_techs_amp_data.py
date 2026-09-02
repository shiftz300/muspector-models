"""Corrected P1/P2 Guitar-TECHS DI to Amp-cab-mic restoration pairs."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile
import torch
from scipy.signal import correlate, correlation_lags

from .license_gate import require_product_weights


RATE = 48_000
SOURCE_ID = "guitar-techs"
ROOT_RELATIVE = Path("data/corpus/guitar-techs/extracted")
HISTORY_FRAMES = 8192
TIME_SPLITS = {
    "fit": (0.0, 400.0),
    "calibration": (400.0, 475.0),
    "development": (475.0, 550.0),
}


@dataclass(frozen=True)
class Profile:
    id: str
    lag_frames: int
    direct: str
    micamp: str
    hardware: str


PROFILES = (
    Profile(
        "P1-orange-cr60-sm57",
        35,
        "P1_singlenotes/audio/directinput/directinput_allsinglenotes.wav",
        "P1_singlenotes/audio/micamp/micamp_allsinglenotes.wav",
        "Ibanez PF300 / Orange CR-60 / SM57",
    ),
    Profile(
        "P2-yamaha-yb15-at2020",
        37,
        "P2_singlenotes/audio/directinput/directinput_allsinglenotes.wav",
        "P2_singlenotes/audio/micamp/micamp_allsinglenotes.wav",
        "EVH Wolfgang / Yamaha YB-15 / AT2020",
    ),
)


def _read(path: Path, start: int, frames: int) -> np.ndarray:
    with soundfile.SoundFile(path) as stream:
        stream.seek(start)
        value = stream.read(frames, dtype="float32", always_2d=True)
    if value.shape[0] != frames or value.shape[1] not in (1, 2):
        raise ValueError(f"invalid Guitar-TECHS window: {path}: {value.shape}")
    if value.shape[1] == 2 and not np.array_equal(value[:, 0], value[:, 1]):
        raise ValueError(f"Guitar-TECHS stereo channels are not duplicated: {path}")
    result = np.asarray(value[:, 0], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite Guitar-TECHS audio: {path}")
    return result


class AmpCabPairs(torch.utils.data.Dataset):
    def __init__(self, workspace: Path, split: str, samples: int, target_frames: int, seed: int) -> None:
        if split not in TIME_SPLITS or samples < 2 or target_frames < 4096:
            raise ValueError("invalid Amp-cab dataset request")
        self.workspace = workspace.resolve()
        self.root = (self.workspace / ROOT_RELATIVE).resolve()
        self.split = split
        self.samples = samples
        self.target_frames = target_frames
        self.total_frames = HISTORY_FRAMES + target_frames
        self.seed = seed
        self.authorization = require_product_weights(
            self.workspace / "remix/data_sources.json", (SOURCE_ID,)
        )
        if any((self.root / profile.direct).parts[-4].startswith("P3") for profile in PROFILES):
            raise ValueError("P3 must never enter Guitar-TECHS Amp-cab pairs")

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.samples:
            raise IndexError(index)
        profile_index = index % len(PROFILES)
        profile = PROFILES[profile_index]
        split_start, split_end = TIME_SPLITS[self.split]
        first = round(split_start * RATE)
        last = round(split_end * RATE) - self.total_frames - profile.lag_frames
        rng = random.Random(self.seed + index * 104729)
        start = rng.randint(first, last)
        clean = _read(self.root / profile.direct, start, self.total_frames)
        wet = _read(self.root / profile.micamp, start + profile.lag_frames, self.total_frames)
        target = slice(HISTORY_FRAMES, None)
        if float(np.sqrt(np.mean(np.square(wet[target] - clean[target])))) < 0.01:
            raise ValueError(f"Amp-cab profile has no meaningful effect: {profile.id}")
        onehot = torch.zeros(len(PROFILES), dtype=torch.float32)
        onehot[profile_index] = 1.0
        return {
            "wet": torch.from_numpy(wet.copy()),
            "clean": torch.from_numpy(clean.copy()),
            "profile": onehot,
            "profile_id": profile.id,
            "target_start": HISTORY_FRAMES,
            "start_frame": start,
        }


def _fit_alignment(root: Path, profile: Profile) -> dict:
    starts = (30.0, 100.0, 180.0, 260.0, 340.0)
    rows = []
    frames = 4 * RATE
    for seconds in starts:
        start = round(seconds * RATE)
        clean = _read(root / profile.direct, start, frames).astype(np.float64)
        wet = _read(root / profile.micamp, start, frames).astype(np.float64)
        clean = np.diff(clean, prepend=clean[0]); clean -= clean.mean()
        wet = np.diff(wet, prepend=wet[0]); wet -= wet.mean()
        values = correlate(wet, clean, mode="full", method="fft")
        lags = correlation_lags(len(wet), len(clean), mode="full")
        selected = np.abs(lags) <= 4800
        values = values[selected]; lags = lags[selected]
        peak = int(np.argmax(np.abs(values)))
        rows.append({
            "start_seconds": seconds,
            "lag_frames": int(lags[peak]),
            "absolute_correlation": float(abs(values[peak]) / max(
                np.linalg.norm(clean) * np.linalg.norm(wet), 1.0e-12
            )),
        })
    median = int(np.median([row["lag_frames"] for row in rows]))
    maximum_residual = max(abs(row["lag_frames"] - profile.lag_frames) for row in rows)
    return {
        "passed": abs(median - profile.lag_frames) <= 32
        and maximum_residual <= 512
        and min(row["absolute_correlation"] for row in rows) >= 0.20,
        "profile": profile.id,
        "configured_lag_frames": profile.lag_frames,
        "fit_median_lag_frames": median,
        "maximum_residual_group_delay_frames": maximum_residual,
        "minimum_absolute_correlation": min(row["absolute_correlation"] for row in rows),
        "rows": rows,
    }


def audit(workspace: Path) -> dict:
    workspace = workspace.resolve()
    root = (workspace / ROOT_RELATIVE).resolve()
    authorization = require_product_weights(workspace / "remix/data_sources.json", (SOURCE_ID,))
    if list(root.glob("P3*")):
        raise PermissionError("P3 final-contaminated audio must remain removed")
    alignment = [_fit_alignment(root, profile) for profile in PROFILES]
    geometry = []
    for profile in PROFILES:
        direct = soundfile.info(root / profile.direct)
        micamp = soundfile.info(root / profile.micamp)
        if direct.samplerate != RATE or micamp.samplerate != RATE:
            raise ValueError("Guitar-TECHS Amp-cab rate changed")
        geometry.append({
            "profile": profile.id,
            "hardware": profile.hardware,
            "direct_frames": direct.frames,
            "micamp_frames": micamp.frames,
            "direct_channels": direct.channels,
            "micamp_channels": micamp.channels,
            "lag_frames": profile.lag_frames,
        })
    passed = all(row["passed"] for row in alignment)
    return {
        "schema": 1,
        "status": "audited-product-eligible-not-trained" if passed else "rejected-alignment",
        "source_id": SOURCE_ID,
        "authorization": authorization,
        "profiles": geometry,
        "alignment": alignment,
        "split_contract": TIME_SPLITS,
        "p3": {
            "status": "permanently-blocked-final-contaminated-and-local-audio-removed",
            "models_trained_on_p3": False,
        },
        "scope": "named fixed Amp+cab+mic profiles; not speaker-output Amp and no knob interpolation",
        "graph_order_input": False,
        "neighbor_effect_input": False,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "profiles": [
            {"profile": row["profile"], "lag": row["configured_lag_frames"], "fit_median": row["fit_median_lag_frames"], "passed": row["passed"]}
            for row in report["alignment"]
        ],
        "p3": report["p3"],
    }, indent=2, sort_keys=True))
    if report["status"].startswith("rejected"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
