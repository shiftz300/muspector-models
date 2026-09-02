"""Product-eligible aligned Marshall JVM410H Amp pairs and split contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import soundfile
import numpy as np
import torch
from scipy.signal import correlate, correlation_lags

from .license_gate import require_product_weights


RATE = 44_100
SOURCE_ID = "marshall-jvm410h"
ROOT_RELATIVE = Path("data/corpus/marshall-jvm410h")
SETTING = re.compile(
    r"^B(?P<bass>\d+(?:\.\d+)?)_M(?P<mid>\d+(?:\.\d+)?)_"
    r"T(?P<treble>\d+(?:\.\d+)?)_G(?P<gain>\d+(?:\.\d+)?)$"
)
UNSEEN_CONTROLS = {
    (0.0, 0.0, 10.0, 5.0),
    (0.0, 10.0, 0.0, 5.0),
    (0.0, 10.0, 10.0, 5.0),
    (10.0, 0.0, 0.0, 5.0),
    (10.0, 0.0, 10.0, 5.0),
    (10.0, 10.0, 0.0, 5.0),
    (6.5, 8.5, 3.5, 5.0),
    (1.0, 3.0, 7.0, 5.0),
    (3.0, 7.0, 1.0, 5.0),
}
DEVELOPMENT_ONLY_CONTROLS = {
    (5.0, 5.0, 5.0, 1.0),
    (5.0, 5.0, 5.0, 8.0),
}
QUARANTINED_SETTING = "B5_M5_T5_G6"
EXPECTED_FIT_SETTINGS = 25
EXPECTED_DEVELOPMENT_SETTINGS = 27
TIME_SPLITS = {
    "fit": (0.0, 240.0),
    "calibration": (240.0, 300.0),
    "development": (300.0, 360.0),
}
HISTORY_FRAMES = 8192


@dataclass(frozen=True)
class AmpPair:
    setting: str
    controls: tuple[float, float, float, float]
    clean: Path
    wet: Path
    locked_final: bool
    development_only: bool

    @property
    def normalized_controls(self) -> tuple[float, float, float, float]:
        return tuple(value / 10.0 for value in self.controls)


def parse_controls(setting: str) -> tuple[float, float, float, float]:
    match = SETTING.fullmatch(setting)
    if match is None:
        raise ValueError(f"invalid JVM410H setting directory: {setting}")
    controls = tuple(float(match.group(name)) for name in ("bass", "mid", "treble", "gain"))
    if any(not 0.0 <= value <= 10.0 for value in controls):
        raise ValueError(f"JVM410H control outside 0..10: {setting}")
    return controls  # type: ignore[return-value]


def discover(workspace: Path) -> list[AmpPair]:
    root = (workspace.resolve() / ROOT_RELATIVE).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing JVM410H corpus: {root}")
    pairs = []
    for wet in sorted(root.rglob("*-speakerout.wav")):
        if "__MACOSX" in wet.parts:
            continue
        setting = wet.parent.name
        controls = parse_controls(setting)
        prefix = wet.name.removesuffix("-speakerout.wav")
        if setting == QUARANTINED_SETTING:
            if prefix != "B5_M5_T5_G4":
                raise ValueError(f"unexpected contents in quarantined JVM410H setting: {wet}")
            continue
        if prefix != setting:
            raise ValueError(f"JVM410H filename and setting disagree: {wet}")
        clean = wet.with_name(f"{prefix}-input.wav")
        if not clean.is_file():
            raise FileNotFoundError(f"missing aligned JVM410H input for {wet}")
        if clean.resolve().parent != wet.resolve().parent or root not in wet.resolve().parents:
            raise ValueError(f"JVM410H pair escaped corpus root: {wet}")
        pairs.append(AmpPair(
            setting,
            controls,
            clean,
            wet,
            controls in UNSEEN_CONTROLS,
            controls in DEVELOPMENT_ONLY_CONTROLS,
        ))
    if not pairs:
        raise ValueError(f"no JVM410H speaker-output pairs under {root}")
    settings = [pair.setting for pair in pairs]
    if len(settings) != len(set(settings)):
        raise ValueError("duplicate JVM410H setting directory")
    return pairs


def _read_window(path: Path, start: int, frames: int) -> np.ndarray:
    with soundfile.SoundFile(path) as stream:
        stream.seek(start)
        audio = stream.read(frames, dtype="float32", always_2d=True)
    if audio.shape != (frames, 1):
        raise ValueError(f"short JVM410H window: {path}: {audio.shape}")
    value = np.asarray(audio[:, 0], dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite JVM410H window: {path}")
    return value


class AmpPairs(torch.utils.data.Dataset):
    """Balanced seen-setting pairs; locked-final controls are structurally excluded."""

    def __init__(
        self,
        workspace: Path,
        split: str,
        samples: int,
        target_frames: int,
        seed: int,
    ) -> None:
        if split not in TIME_SPLITS:
            raise ValueError(f"unsupported Amp split: {split}")
        if samples < 1 or target_frames < 4096:
            raise ValueError("Amp dataset request is too small")
        self.workspace = workspace.resolve()
        self.split = split
        self.samples = samples
        self.target_frames = target_frames
        self.history_frames = HISTORY_FRAMES
        self.total_frames = HISTORY_FRAMES + target_frames
        self.seed = seed
        available = tuple(pair for pair in discover(self.workspace) if not pair.locked_final)
        self.pairs = tuple(
            pair for pair in available
            if split == "development" or not pair.development_only
        )
        expected = (
            EXPECTED_DEVELOPMENT_SETTINGS
            if split == "development"
            else EXPECTED_FIT_SETTINGS
        )
        if len(self.pairs) != expected:
            raise ValueError(
                f"Amp {split} requires exactly {expected} settings, got {len(self.pairs)}"
            )
        self.authorization = require_product_weights(
            self.workspace / "remix/data_sources.json", (SOURCE_ID,)
        )

    def __len__(self) -> int:
        return self.samples

    def realized_setting_counts(self) -> dict[str, int]:
        return {
            pair.setting: sum(index % len(self.pairs) == offset for index in range(self.samples))
            for offset, pair in enumerate(self.pairs)
        }

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.samples:
            raise IndexError(index)
        pair = self.pairs[index % len(self.pairs)]
        split_start, split_end = TIME_SPLITS[self.split]
        first = round(split_start * RATE)
        last = round(split_end * RATE) - self.total_frames
        if last < first:
            raise ValueError("Amp split is shorter than the requested model context")
        rng = random.Random(self.seed + index * 104729)
        start = rng.randint(first, last)
        clean = _read_window(pair.clean, start, self.total_frames)
        wet = _read_window(pair.wet, start, self.total_frames)
        target = slice(self.history_frames, None)
        distance = float(np.sqrt(np.mean(np.square(wet[target] - clean[target]), dtype=np.float64)))
        clean_rms = float(np.sqrt(np.mean(np.square(clean[target]), dtype=np.float64)))
        if distance < 0.02 * max(clean_rms, 1.0e-6):
            raise ValueError(f"JVM410H pair has no meaningful Amp effect: {pair.setting}")
        return {
            "wet": torch.from_numpy(wet.copy()),
            "clean": torch.from_numpy(clean.copy()),
            "controls": torch.tensor(pair.normalized_controls, dtype=torch.float32),
            "control_values": {
                "bass": pair.controls[0],
                "mid": pair.controls[1],
                "treble": pair.controls[2],
                "gain": pair.controls[3],
            },
            "target_start": self.history_frames,
            "setting": pair.setting,
            "source_id": SOURCE_ID,
            "start_frame": start,
        }


def _header(path: Path) -> dict:
    info = soundfile.info(path)
    if info.format != "WAV" or info.channels != 1 or info.samplerate != RATE:
        raise ValueError(f"unsupported JVM410H audio geometry: {path}: {info}")
    if info.subtype not in {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}:
        raise ValueError(f"lossy or unsupported JVM410H subtype: {path}: {info.subtype}")
    return {
        "frames": int(info.frames),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "format": info.format,
        "subtype": info.subtype,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_data_path(workspace: Path, path: Path) -> str:
    root = (workspace / ROOT_RELATIVE).resolve()
    return str(ROOT_RELATIVE / path.resolve().relative_to(root))


def _alignment_report(pairs: list[AmpPair]) -> dict:
    rows = []
    window_frames = 2 * RATE
    maximum_lag = 2048
    audit_starts = (10.0, 70.0, 130.0, 190.0, 250.0, 310.0, 376.0)
    for pair in pairs:
        for start_seconds in audit_starts:
            start = round(start_seconds * RATE)
            clean = _read_window(pair.clean, start, window_frames)
            wet = _read_window(pair.wet, start, window_frames)
            clean_feature = np.diff(clean, prepend=clean[0]).astype(np.float64)
            wet_feature = np.diff(wet, prepend=wet[0]).astype(np.float64)
            clean_feature -= clean_feature.mean()
            wet_feature -= wet_feature.mean()
            correlations = correlate(wet_feature, clean_feature, mode="full", method="fft")
            lags = correlation_lags(len(wet_feature), len(clean_feature), mode="full")
            selected = np.abs(lags) <= maximum_lag
            local = correlations[selected]
            local_lags = lags[selected]
            peak_index = int(np.argmax(np.abs(local)))
            lag = int(local_lags[peak_index])
            score = float(abs(local[peak_index]) / max(
                np.linalg.norm(clean_feature) * np.linalg.norm(wet_feature), 1.0e-12
            ))
            rows.append({
                "setting": pair.setting,
                "start_seconds": start_seconds,
                "lag_frames": lag,
                "absolute_correlation": score,
                "clean_peak": float(np.max(np.abs(clean))),
                "wet_peak": float(np.max(np.abs(wet))),
            })
    lags = np.asarray([row["lag_frames"] for row in rows], dtype=np.int64)
    median_lag = int(np.median(lags))
    maximum_deviation = int(np.max(np.abs(lags - median_lag)))
    maximum_absolute_lag = int(np.max(np.abs(lags)))
    minimum_correlation = float(min(row["absolute_correlation"] for row in rows))
    fit_calibration_spans = []
    for pair in pairs:
        selected = [
            row["lag_frames"]
            for row in rows
            if row["setting"] == pair.setting and row["start_seconds"] <= 250.0
        ]
        fit_calibration_spans.append(max(selected) - min(selected))
    maximum_fit_calibration_lag_span = int(max(fit_calibration_spans))
    reference = next(pair for pair in pairs if pair.setting == "B5_M5_T5_G5")
    reference_features = {}
    for start_seconds in audit_starts:
        start = round(start_seconds * RATE)
        value = _read_window(reference.clean, start, window_frames)
        feature = np.diff(value, prepend=value[0]).astype(np.float64)
        feature -= feature.mean()
        reference_features[start_seconds] = feature
    source_rows = []
    for pair in pairs:
        for start_seconds in audit_starts:
            start = round(start_seconds * RATE)
            value = _read_window(pair.clean, start, window_frames)
            feature = np.diff(value, prepend=value[0]).astype(np.float64)
            feature -= feature.mean()
            reference_feature = reference_features[start_seconds]
            correlations = correlate(feature, reference_feature, mode="full", method="fft")
            lags = correlation_lags(len(feature), len(reference_feature), mode="full")
            selected = np.abs(lags) <= 256
            local = correlations[selected]
            local_lags = lags[selected]
            peak_index = int(np.argmax(np.abs(local)))
            source_rows.append({
                "setting": pair.setting,
                "start_seconds": start_seconds,
                "lag_frames": int(local_lags[peak_index]),
                "absolute_correlation": float(abs(local[peak_index]) / max(
                    np.linalg.norm(feature) * np.linalg.norm(reference_feature), 1.0e-12
                )),
            })
    source_maximum_absolute_lag = max(abs(row["lag_frames"]) for row in source_rows)
    source_minimum_correlation = min(row["absolute_correlation"] for row in source_rows)
    passed = (
        maximum_absolute_lag <= 128
        and minimum_correlation >= 0.09
        and source_maximum_absolute_lag <= 1
        and source_minimum_correlation >= 0.99
    )
    return {
        "passed": passed,
        "windows": len(rows),
        "window_frames": window_frames,
        "maximum_search_lag_frames": maximum_lag,
        "audit_start_seconds": list(audit_starts),
        "median_lag_frames": median_lag,
        "maximum_lag_deviation_frames": maximum_deviation,
        "maximum_absolute_lag_frames": maximum_absolute_lag,
        "maximum_fit_calibration_lag_span_frames": maximum_fit_calibration_lag_span,
        "minimum_absolute_correlation": minimum_correlation,
        "zero_latency_required": False,
        "observed_group_delay_removed": False,
        "source_program_alignment": {
            "reference_setting": reference.setting,
            "maximum_absolute_lag_frames": source_maximum_absolute_lag,
            "minimum_absolute_correlation": source_minimum_correlation,
            "rows": source_rows,
        },
        "rows": rows,
    }


def _scan_seen_audio(pairs: list[AmpPair]) -> dict:
    rows = []
    for pair in pairs:
        for role, path in (("clean", pair.clean), ("wet", pair.wet)):
            peak = 0.0
            clipped = 0
            samples = 0
            energy = 0.0
            with soundfile.SoundFile(path) as stream:
                for block in stream.blocks(blocksize=1024 * 1024, dtype="float32", always_2d=True):
                    value = np.asarray(block[:, 0], dtype=np.float32)
                    if not np.isfinite(value).all():
                        raise ValueError(f"non-finite JVM410H source audio: {path}")
                    peak = max(peak, float(np.max(np.abs(value), initial=0.0)))
                    clipped += int(np.count_nonzero(np.abs(value) >= 0.999999))
                    samples += len(value)
                    energy += float(np.sum(np.square(value, dtype=np.float64)))
            rows.append({
                "setting": pair.setting,
                "role": role,
                "samples": samples,
                "peak": peak,
                "rms": float(np.sqrt(energy / max(samples, 1))),
                "full_scale_fraction": clipped / max(samples, 1),
            })
    maximum_clipping = max(row["full_scale_fraction"] for row in rows)
    minimum_rms = min(row["rms"] for row in rows)
    return {
        "passed": maximum_clipping <= 0.001 and minimum_rms >= 1.0e-5,
        "files": len(rows),
        "maximum_full_scale_fraction": maximum_clipping,
        "minimum_rms": minimum_rms,
        "rows": rows,
    }


def audit(workspace: Path, *, hash_audio: bool = False, decode_seen: bool = False) -> dict:
    workspace = workspace.resolve()
    authorization = require_product_weights(
        workspace / "remix/data_sources.json", (SOURCE_ID,)
    )
    pairs = discover(workspace)
    seen = [pair for pair in pairs if not pair.locked_final]
    locked = [pair for pair in pairs if pair.locked_final]
    if len(seen) != EXPECTED_DEVELOPMENT_SETTINGS or len(locked) != 9:
        raise ValueError(
            "JVM410H corpus must contain "
            f"{EXPECTED_DEVELOPMENT_SETTINGS} development-visible and 9 locked settings, "
            f"got {len(seen)}/{len(locked)}"
        )
    rows = []
    clean_hashes = set()
    for pair in pairs:
        clean = _header(pair.clean)
        wet = _header(pair.wet)
        if clean["frames"] != wet["frames"]:
            raise ValueError(f"JVM410H pair length mismatch: {pair.setting}")
        minimum_seconds = 380.0 if pair.locked_final else 360.0
        if clean["frames"] < round(minimum_seconds * RATE):
            raise ValueError(f"JVM410H pair is too short for frozen splits: {pair.setting}")
        row = {
            "setting": pair.setting,
            "controls": {
                "bass": pair.controls[0],
                "mid": pair.controls[1],
                "treble": pair.controls[2],
                "gain": pair.controls[3],
            },
            "locked_final": pair.locked_final,
            "development_only": pair.development_only,
            "frames": clean["frames"],
            "seconds": clean["frames"] / RATE,
            "subtype": clean["subtype"],
            "paths": {
                "clean": _logical_data_path(workspace, pair.clean),
                "wet": _logical_data_path(workspace, pair.wet),
            },
        }
        if hash_audio and not pair.locked_final:
            row["sha256"] = {"clean": _sha256(pair.clean), "wet": _sha256(pair.wet)}
            clean_hashes.add(row["sha256"]["clean"])
        rows.append(row)
    alignment = _alignment_report(seen) if decode_seen else None
    signal_quality = _scan_seen_audio(seen) if decode_seen else None
    status = (
        "audited-product-eligible-not-trained"
        if alignment is not None and alignment["passed"] and signal_quality is not None and signal_quality["passed"]
        else "metadata-audited-alignment-pending"
        if alignment is None
        else "rejected-alignment"
    )
    return {
        "schema": 1,
        "status": status,
        "source_id": SOURCE_ID,
        "authorization": authorization,
        "pairs": {
            "fit_calibration": sum(not pair.development_only for pair in seen),
            "development_visible": len(seen),
            "development_only": sum(pair.development_only for pair in seen),
            "locked_final": len(locked),
            "quarantined": 1,
            "total_in_archive": len(pairs) + 1,
        },
        "geometry": {
            "sample_rate": RATE,
            "channels": 1,
            "aligned_by_author_contract": True,
            "latency_recheck_required_before_training": True,
        },
        "split_contract": {
            "fit_calibration_settings": {
                "count": EXPECTED_FIT_SETTINGS,
                "time_splits": {
                    "fit": TIME_SPLITS["fit"],
                    "calibration": TIME_SPLITS["calibration"],
                },
            },
            "development_settings": {
                "count": EXPECTED_DEVELOPMENT_SETTINGS,
                "time_split": TIME_SPLITS["development"],
                "gain_interpolation_holdouts": [1.0, 8.0],
            },
            "unseen_settings": "locked-final; audio samples remain unopened until promotion gate",
            "quarantined_setting": {
                "setting": QUARANTINED_SETTING,
                "reason": "directory says Gain=6 while all three member filenames say Gain=4",
            },
            "same_source_program_across_settings": True,
            "source_program_diversity": 1,
        },
        "control_contract": {
            "controls": ["bass", "mid", "treble", "gain"],
            "gain_domain_fit_calibration": [0.0, 2.0, 4.0, 5.0, 10.0],
            "gain_domain_development_only": [1.0, 8.0],
            "limitation": "Marshall JVM410H OD1 only; Gain=6 source is quarantined for contradictory naming",
        },
        "hash_audio": hash_audio,
        "locked_final_audio_hashed": False,
        "unique_clean_file_hashes": len(clean_hashes) if hash_audio else None,
        "rows": rows,
        "seen_alignment": alignment,
        "seen_signal_quality": signal_quality,
        "provenance": {
            "product_source_only": True,
            "research_data_used": False,
            "locked_final_audio_decoded": False,
            "physical_audio_devices_used": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product3-amp-marshall/data-audit.json"),
    )
    parser.add_argument("--hash-audio", action="store_true")
    parser.add_argument("--decode-seen", action="store_true")
    args = parser.parse_args()
    report = audit(args.workspace, hash_audio=args.hash_audio, decode_seen=args.decode_seen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    alignment = report.get("seen_alignment") or {}
    signal = report.get("seen_signal_quality") or {}
    print(json.dumps({
        "status": report["status"],
        "pairs": report["pairs"],
        "alignment": {
            name: alignment.get(name)
            for name in (
                "passed",
                "windows",
                "median_lag_frames",
                "maximum_absolute_lag_frames",
                "maximum_fit_calibration_lag_span_frames",
                "minimum_absolute_correlation",
            )
        },
        "source_program_alignment": {
            name: alignment.get("source_program_alignment", {}).get(name)
            for name in (
                "maximum_absolute_lag_frames",
                "minimum_absolute_correlation",
            )
        },
        "signal_quality": {
            name: signal.get(name)
            for name in ("passed", "maximum_full_scale_fraction", "minimum_rms")
        },
    }, indent=2, sort_keys=True))
    if report["status"].startswith("rejected"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
