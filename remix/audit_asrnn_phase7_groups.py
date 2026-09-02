#!/usr/bin/env python3
"""Audit performance-group coverage for Phase-7 CS-3 development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .asrnn_effects import effect_files, read_effect_pair


def take_id(path: Path) -> int:
    return int(path.stem.split(",")[-1])


def attack_value(path: Path) -> int:
    return int(path.stem.split(",")[0])


def dry_features(dry: np.ndarray) -> np.ndarray:
    absolute = np.abs(dry)
    rms = float(np.sqrt(np.mean(np.square(dry))))
    peak = float(absolute.max())
    emphasized = np.abs(np.diff(dry, prepend=dry[0]) + 0.05 * dry)
    return np.asarray(
        (
            peak,
            rms,
            peak / max(rms, 1.0e-8),
            float(np.quantile(absolute, 0.95)),
            float(np.quantile(emphasized, 0.99)),
            float(
                np.mean(
                    emphasized
                    >= max(4.0 * float(np.sqrt(np.mean(np.square(emphasized)))), 1.0e-8)
                )
            ),
        ),
        dtype=np.float64,
    )


def performances(corpus: Path, split: str) -> dict[str, dict]:
    result = {}
    for path in effect_files(corpus, "cs3", split):
        dry, _, _ = read_effect_pair(path, "cs3")
        envelope_frames = len(dry) // 480 * 480
        result[path.name] = {
            "take_id": take_id(path),
            "attack": attack_value(path),
            "frames": len(dry),
            "dry_sha256": hashlib.sha256(dry.tobytes()).hexdigest(),
            "features": dry_features(dry),
            "envelope": np.sqrt(
                np.mean(dry[:envelope_frames].reshape(-1, 480) ** 2, axis=1)
            ),
            "wave_probe": dry[::16].astype(np.float64),
        }
    return result


def near_performance_pairs(rows: dict[str, dict]) -> list[dict]:
    identities = sorted(rows)
    envelopes = np.stack([rows[name]["envelope"] for name in identities])
    pairs = {}
    frames = envelopes.shape[1]
    for lag in range(-3, 4):
        left = envelopes[:, max(lag, 0) : frames + min(lag, 0)].astype(np.float64)
        right = envelopes[:, max(-lag, 0) : frames - max(lag, 0)].astype(np.float64)
        left = (left - left.mean(1, keepdims=True)) / left.std(1, keepdims=True).clip(1.0e-9)
        right = (right - right.mean(1, keepdims=True)) / right.std(1, keepdims=True).clip(1.0e-9)
        correlations = left @ right.T / left.shape[1]
        for first, second in np.argwhere(np.triu(correlations >= 0.995, k=1)):
            a = rows[identities[int(first)]]["wave_probe"]
            b = rows[identities[int(second)]]["wave_probe"]
            offset = lag * 30
            a = a[max(offset, 0) : len(a) + min(offset, 0)]
            b = b[max(-offset, 0) : len(b) - max(offset, 0)]
            waveform_correlation = float(np.corrcoef(a, b)[0, 1])
            if waveform_correlation < 0.95:
                continue
            key = (identities[int(first)], identities[int(second)])
            row = {
                "first": key[0],
                "second": key[1],
                "lag_seconds": lag * 0.01,
                "envelope_correlation": float(correlations[first, second]),
                "waveform_correlation": waveform_correlation,
            }
            if key not in pairs or row["waveform_correlation"] > pairs[key]["waveform_correlation"]:
                pairs[key] = row
    return [pairs[key] for key in sorted(pairs)]


def _maximin(identities: list[str], values: np.ndarray, count: int) -> list[str]:
    if not 0 < count < len(identities):
        raise ValueError("calibration count must leave at least one fit file")
    scale = values.std(axis=0)
    normalized = (values - values.mean(axis=0)) / np.where(scale > 1.0e-9, scale, 1.0)
    distances = np.linalg.norm(normalized - normalized.mean(axis=0), axis=1)
    selected = [int(np.argmax(distances))]
    while len(selected) < count:
        nearest = np.min(
            np.linalg.norm(
                normalized[:, None, :] - normalized[np.asarray(selected)][None, :, :],
                axis=2,
            ),
            axis=1,
        )
        nearest[np.asarray(selected)] = -1.0
        selected.append(int(np.argmax(nearest)))
    return sorted(identities[index] for index in selected)


def coverage_calibration_files(
    rows: dict[str, dict], count_per_control: int = 8
) -> list[str]:
    selected = []
    attacks = sorted({row["attack"] for row in rows.values()})
    for attack in attacks:
        identities = sorted(
            identity for identity, row in rows.items() if row["attack"] == attack
        )
        values = np.stack([rows[identity]["features"] for identity in identities])
        selected.extend(_maximin(identities, values, count_per_control))
    return sorted(selected)


def audit(corpus: Path) -> dict:
    train = performances(corpus, "train")
    challenge = performances(corpus, "eval")
    near_pairs = near_performance_pairs(
        {**{f"train/{name}": row for name, row in train.items()},
         **{f"eval/{name}": row for name, row in challenge.items()}}
    )
    cross_near_pairs = [
        pair for pair in near_pairs
        if pair["first"].split("/")[0] != pair["second"].split("/")[0]
    ]
    if cross_near_pairs:
        raise ValueError("Phase-7 train and development challenge share near-identical performances")
    train_hashes = {value["dry_sha256"] for value in train.values()}
    challenge_hashes = {value["dry_sha256"] for value in challenge.values()}
    overlap = train_hashes & challenge_hashes
    if overlap:
        raise ValueError("Phase-7 train and development challenge share Dry audio")
    feature_names = (
        "absolute_peak",
        "rms",
        "crest_factor",
        "absolute_q95",
        "preemphasis_q99",
        "strong_transient_density",
    )
    train_values = np.stack([value["features"] for value in train.values()])
    challenge_values = np.stack([value["features"] for value in challenge.values()])
    scale = train_values.std(axis=0)
    normalized_train = (train_values - train_values.mean(axis=0)) / np.where(
        scale > 1.0e-9, scale, 1.0
    )
    normalized_challenge = (challenge_values - train_values.mean(axis=0)) / np.where(
        scale > 1.0e-9, scale, 1.0
    )
    nearest = np.min(
        np.linalg.norm(
            normalized_challenge[:, None, :] - normalized_train[None, :, :], axis=2
        ),
        axis=1,
    )
    calibration_set = set(coverage_calibration_files(train))
    train_near_pairs = [
        pair for pair in near_pairs
        if pair["first"].startswith("train/") and pair["second"].startswith("train/")
    ]
    changed = True
    while changed:
        changed = False
        for pair in train_near_pairs:
            names = {pair["first"].split("/", 1)[1], pair["second"].split("/", 1)[1]}
            if names & calibration_set and not names <= calibration_set:
                calibration_set.update(names)
                changed = True
    calibration = sorted(calibration_set)
    calibration_hashes = {train[name]["dry_sha256"] for name in calibration}
    fit_hashes = {
        value["dry_sha256"] for name, value in train.items() if name not in calibration
    }
    if calibration_hashes & fit_hashes:
        raise ValueError("Phase-7 fit and calibration share exact Dry audio")
    return {
        "schema": 1,
        "phase": "phase-7-performance-group-audit",
        "device": "cs3",
        "feature_names": list(feature_names),
        "train_performances": len(train),
        "train_unique_dry_hashes": len(train_hashes),
        "train_clip_frames": sorted({value["frames"] for value in train.values()}),
        "development_challenge_performances": len(challenge),
        "development_challenge_unique_dry_hashes": len(challenge_hashes),
        "cross_split_exact_dry_duplicates": len(overlap),
        "cross_split_near_performance_pairs": len(cross_near_pairs),
        "near_performance_rule": "10-ms RMS envelopes with +/-30-ms lag >=0.995, verified waveform correlation >=0.95",
        "train_near_performance_pairs": train_near_pairs,
        "fit_calibration_near_performance_pairs": 0,
        "fit_calibration_exact_dry_duplicates": len(calibration_hashes & fit_hashes),
        "calibration_selection": "dry-feature maximin per Attack, expanded by near-performance groups; Wet labels unused",
        "calibration_files": calibration,
        "fit_files": sorted(set(train) - set(calibration)),
        "challenge_to_train_dry_feature_distance": {
            "median": float(np.median(nearest)),
            "p95": float(np.quantile(nearest, 0.95)),
            "maximum": float(nearest.max()),
        },
        "train_feature_ranges": {
            name: {
                "minimum": float(train_values[:, index].min()),
                "maximum": float(train_values[:, index].max()),
            }
            for index, name in enumerate(feature_names)
        },
        "development_challenge_feature_ranges": {
            name: {
                "minimum": float(challenge_values[:, index].min()),
                "maximum": float(challenge_values[:, index].max()),
            }
            for index, name in enumerate(feature_names)
        },
        "source_files_read_only": True,
        "automatic_normalization": False,
        "physical_audio_devices_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.corpus.resolve())
    if args.output.exists():
        raise ValueError(f"Phase-7 group audit already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
