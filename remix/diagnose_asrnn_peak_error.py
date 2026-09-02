#!/usr/bin/env python3
"""Diagnose CS-3 peak errors without modifying audio or model weights."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .asrnn_effects import effect_files, read_effect_pair
from .audit_asrnn_phase7_groups import dry_features
from .centered_stable_effect import load_centered_stable_effect
from .stable_effect import load_stable_effect


@torch.inference_mode()
def diagnose(args) -> dict:
    load = load_centered_stable_effect if args.centered else load_stable_effect
    model, payload = load(args.checkpoint)
    if payload["device"] != "cs3":
        raise ValueError("peak diagnosis requires a CS-3 checkpoint")
    paths = effect_files(args.corpus.resolve(), "cs3", args.split)
    rows = []
    for offset in range(0, len(paths), 32):
        selected = paths[offset : offset + 32]
        pairs = [read_effect_pair(path, "cs3") for path in selected]
        dry = torch.from_numpy(np.stack([pair[0] for pair in pairs]))
        wet = torch.from_numpy(np.stack([pair[1] for pair in pairs]))
        controls = torch.from_numpy(np.stack([pair[2] for pair in pairs]))
        state = None
        chunks = []
        for start in range(0, dry.shape[1], 4_096):
            output, state = model(dry[:, start : start + 4_096], controls, state)
            chunks.append(output)
        prediction = torch.cat(chunks, dim=1)[:, 1_024:]
        target = wet[:, 1_024:]
        error = (prediction - target).square().mean(1)
        energy = target.square().mean(1)
        esr = (error / energy.clamp_min(1.0e-8)).tolist()
        target_peak = target.abs().amax(1).tolist()
        predicted_peak = prediction.abs().amax(1).tolist()
        target_peak_frames = (target.abs().argmax(1) + 1024).tolist()
        predicted_peak_frames = (prediction.abs().argmax(1) + 1024).tolist()
        for path, pair, value, expected_peak, actual_peak, target_frame, predicted_frame in zip(
            selected, pairs, esr, target_peak, predicted_peak, target_peak_frames, predicted_peak_frames
        ):
            features = dry_features(pair[0])
            rows.append(
                {
                    "file": path.name,
                    "attack": int(path.stem.split(",")[0]),
                    "dry_peak": float(features[0]),
                    "dry_rms": float(features[1]),
                    "dry_crest": float(features[2]),
                    "dry_preemphasis_q99": float(features[4]),
                    "dry_transient_density": float(features[5]),
                    "target_peak": expected_peak,
                    "predicted_peak": actual_peak,
                    "signed_peak_error": actual_peak - expected_peak,
                    "absolute_peak_error": abs(actual_peak - expected_peak),
                    "esr": value,
                    "target_peak_seconds": target_frame / 48000.0,
                    "predicted_peak_seconds": predicted_frame / 48000.0,
                }
            )
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["attack"]].append(row)
    errors = np.asarray([row["absolute_peak_error"] for row in rows])
    correlations = {}
    for feature in (
        "dry_peak",
        "dry_rms",
        "dry_crest",
        "dry_preemphasis_q99",
        "dry_transient_density",
        "target_peak",
    ):
        values = np.asarray([row[feature] for row in rows])
        correlations[feature] = float(np.corrcoef(values, errors)[0, 1])
    return {
        "schema": 1,
        "phase": "phase-7-peak-diagnosis",
        "split": "development-challenge" if args.split == "eval" else "train",
        "checkpoint": str(args.checkpoint),
        "examples": len(rows),
        "absolute_peak_error_p95": float(np.quantile(errors, 0.95)),
        "absolute_peak_error_over_002_count": int(np.sum(errors > 0.02)),
        "peak_error_correlations": correlations,
        "by_attack": {
            str(attack): {
                "files": len(values),
                "absolute_peak_error_p95": float(
                    np.quantile([row["absolute_peak_error"] for row in values], 0.95)
                ),
                "mean_esr": float(np.mean([row["esr"] for row in values])),
            }
            for attack, values in sorted(grouped.items())
        },
        "worst_files": sorted(rows, key=lambda row: -row["absolute_peak_error"])[:24],
        "source_files_read_only": True,
        "physical_audio_devices_used": False,
        "automatic_normalization": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "eval"), default="eval")
    parser.add_argument("--centered", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args)
    if args.output.exists():
        raise ValueError(f"peak diagnostic already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "examples": report["examples"],
                "absolute_peak_error_p95": report["absolute_peak_error_p95"],
                "peak_error_correlations": report["peak_error_correlations"],
                "worst_files": report["worst_files"][:3],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
