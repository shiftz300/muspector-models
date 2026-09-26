"""Fail-closed EGDB-PG Amp+cab pairs with disjoint profile evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .license_gate import require_product_uses


RATE = 44_100
SOURCE_ID = "egdb-pg-v2"
ROOT_RELATIVE = Path("data/corpus/egdb-pg-subset-v1")
CONTRACT_RELATIVE = Path("remix/egdb_pg_subset_v1.json")
AUDIT_RELATIVE = ROOT_RELATIVE / "audio_audit.json"
CATEGORIES = ("low_gain", "crunch", "high_gain")


def _profiles(contract: dict, split: str) -> list[tuple[str, str]]:
    if split in {"fit", "calibration"}:
        groups = ("fit_calibration", "fit_calibration_additional")
    elif split == "development":
        groups = ("development",)
    elif split == "fresh_validation":
        groups = ("fresh_validation_not_downloaded",)
    else:
        groups = ("fresh_validation_v2_not_downloaded",)
    result = []
    for group in groups:
        for category in CATEGORIES:
            values = contract["profiles"].get(group, {}).get(category, [])
            for profile in values if isinstance(values, list) else [values]:
                result.append((category, profile))
    return result


def _read(path: Path, start: int, frames: int) -> np.ndarray:
    with sf.SoundFile(path) as stream:
        stream.seek(start)
        value = stream.read(frames, dtype="float32", always_2d=True)
    if value.shape != (frames, 1) or not np.isfinite(value).all():
        raise ValueError(f"invalid EGDB-PG window: {path}: {value.shape}")
    return np.asarray(value[:, 0], dtype=np.float32)


class EgdbPgAmpPairs(torch.utils.data.Dataset):
    """Balanced windows; category/profile are reporting metadata, never inputs."""

    def __init__(
        self, workspace: Path, split: str, samples: int, target_frames: int,
        context_frames: int, seed: int, reference_frames: int = 0,
    ) -> None:
        if split not in {
            "fit", "calibration", "development",
            "fresh_validation", "fresh_validation_v2",
        }:
            raise ValueError(f"invalid EGDB-PG split: {split}")
        if samples < 1 or target_frames < 4096 or context_frames < 1024:
            raise ValueError("EGDB-PG request is too small")
        self.workspace = workspace.resolve()
        self.root = (self.workspace / ROOT_RELATIVE).resolve()
        self.contract = json.loads((self.workspace / CONTRACT_RELATIVE).read_text())
        self.audit = json.loads((self.workspace / AUDIT_RELATIVE).read_text())
        if not self.audit.get("passed") or self.audit.get("locked_final_downloaded") is not False:
            raise PermissionError("EGDB-PG audio audit or locked-final boundary failed")
        if split == "fresh_validation" and not self.audit.get(
            "fresh_validation_downloaded", False
        ):
            raise PermissionError("EGDB-PG fresh validation remains sealed")
        if split == "fresh_validation_v2" and not self.audit.get(
            "fresh_validation_v2_downloaded", False
        ):
            raise PermissionError("EGDB-PG fresh validation v2 remains sealed")
        self.authorization = require_product_uses(
            self.workspace / "remix/data_sources.json", {SOURCE_ID: "train-amp-cab"}
        )
        self.split = split
        self.samples = samples
        self.target_frames = target_frames
        self.context_frames = context_frames
        self.reference_frames = reference_frames
        self.total_frames = target_frames + 2 * context_frames
        self.seed = seed
        self.profile_rows = _profiles(self.contract, split)
        track_group = {
            "fresh_validation": "fresh_validation_not_downloaded",
            "fresh_validation_v2": "fresh_validation_v2_not_downloaded",
        }.get(split, split)
        self.tracks = tuple(self.contract["tracks"][track_group])
        if split in {"development", "fresh_validation", "fresh_validation_v2"}:
            training = {profile for _, profile in _profiles(self.contract, "fit")}
            evaluation = {profile for _, profile in self.profile_rows}
            if training & evaluation:
                raise PermissionError(
                    f"EGDB-PG {split} profile leaked into training"
                )
        locked_profiles = set(self.contract["profiles"]["locked_final_not_downloaded"].values())
        locked_tracks = set(self.contract["tracks"]["locked_final_not_downloaded"])
        if locked_profiles & {profile for _, profile in self.profile_rows} or locked_tracks & set(self.tracks):
            raise PermissionError("EGDB-PG locked-final material entered a decoded split")

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.samples:
            raise IndexError(index)
        profile_index = index % len(self.profile_rows)
        cycle = index // len(self.profile_rows)
        category, profile = self.profile_rows[profile_index]
        track = self.tracks[cycle % len(self.tracks)]
        clean_path = self.root / "clean" / f"{track}.wav"
        wet_path = self.root / self.split / category / profile / f"{track}_output.flac"
        clean_info, wet_info = sf.info(clean_path), sf.info(wet_path)
        last = min(clean_info.frames, wet_info.frames) - self.total_frames
        if last < 0:
            raise ValueError(f"EGDB-PG track shorter than requested context: {track}")
        rng = random.Random(self.seed + index * 104_729)
        target = slice(self.context_frames, -self.context_frames)
        for _ in range(48):
            start = rng.randint(0, last)
            clean = _read(clean_path, start, self.total_frames)
            wet = _read(wet_path, start, self.total_frames)
            signal = max(float(np.sqrt(np.mean(clean[target] ** 2))), float(np.sqrt(np.mean(wet[target] ** 2))))
            effect = float(np.sqrt(np.mean((wet[target] - clean[target]) ** 2)))
            if signal >= 5.0e-4 and effect >= 0.05 * signal:
                break
        else:
            raise ValueError(f"no active EGDB-PG window found: {self.split}/{profile}/{track}")
        # Joint post-render gain augmentation preserves the inverse relation and
        # prevents the Wet-only conditioner from treating loudness as profile ID.
        gain = 10.0 ** rng.uniform(-3.0, 3.0) / 20.0
        row = {
            "wet": torch.from_numpy((wet * gain).copy()),
            "clean": torch.from_numpy((clean * gain).copy()),
            "category": category,
            "profile_id": profile,
            "track": track,
            "start_frame": start,
            "crop_start": self.context_frames,
            "crop_end": self.context_frames + self.target_frames,
        }
        if self.reference_frames:
            reference_last = wet_info.frames - self.reference_frames
            if reference_last < 0:
                raise ValueError(f"EGDB-PG track shorter than tone reference: {track}")
            reference_start = rng.randint(0, reference_last)
            reference_gain = 10.0 ** rng.uniform(-3.0, 3.0) / 20.0
            reference = _read(wet_path, reference_start, self.reference_frames)
            row["tone_reference"] = torch.from_numpy((reference * reference_gain).copy())
        return row
