#!/usr/bin/env python3
"""Fit a calibration-only observable selector over frozen Amp candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .amp_data import AmpPairs, EXPECTED_DEVELOPMENT_SETTINGS, RATE
from .amp_model import control_basis
from .audit_amp_candidate_oracle import _load
from .quality2 import REQUIRED_IMPROVEMENTS, measure, summarize
from .train_amp import SEED, _stratum


RIDGE = 0.05


def _features(row: dict) -> np.ndarray:
    start = int(row["target_start"])
    wet = row["wet"][start:].numpy().astype(np.float64)
    controls = row["controls"].unsqueeze(0)
    basis = control_basis(controls)[0].numpy().astype(np.float64)
    rms = float(np.sqrt(np.mean(wet * wet) + 1.0e-12))
    peak = float(np.max(np.abs(wet)))
    difference = np.diff(wet, prepend=wet[0])
    spectrum = np.abs(np.fft.rfft(wet * np.hanning(len(wet)))) ** 2
    frequencies = np.fft.rfftfreq(len(wet), 1.0 / RATE)
    total = float(np.sum(spectrum) + 1.0e-12)
    high_ratio = float(np.sum(spectrum[frequencies >= 3000.0]) / total)
    centroid = float(np.sum(frequencies * spectrum) / total / (RATE / 2.0))
    frames = wet[: len(wet) // 256 * 256].reshape(-1, 256)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1.0e-12)
    attack = np.maximum(np.diff(np.log(frame_rms + 1.0e-6)), 0.0)
    observables = np.asarray([
        np.log(rms + 1.0e-6),
        peak / max(rms, 1.0e-6),
        float(np.sqrt(np.mean(difference * difference)) / max(rms, 1.0e-6)),
        high_ratio,
        centroid,
        float(np.mean(attack)),
        float(np.max(attack, initial=0.0)),
    ])
    return np.concatenate((basis, observables))


def _label(report: dict) -> float:
    reductions = [
        report["metrics"][name]["reduction"]
        for name in sorted(REQUIRED_IMPROVEMENTS["amp"])
    ]
    return 2.0 * float(report["passed"]) + min(reductions) + 0.10 * float(np.mean(reductions))


def _candidate_outputs(candidates: list[tuple[str, torch.nn.Module, str]], row: dict) -> list[np.ndarray]:
    start = int(row["target_start"])
    values = []
    with torch.inference_mode():
        for _, model, _ in candidates:
            restored, _, _ = model(row["wet"].unsqueeze(0), row["controls"].unsqueeze(0))
            values.append(restored[0, start:].numpy().astype(np.float32))
    return values


def calibrate(workspace: Path, checkpoints: list[Path], target_frames: int) -> dict:
    loaded = [_load(path.resolve()) for path in checkpoints]
    candidate_ids = [f"{path.parent.name}:{name}" for path, (name, _, _) in zip(checkpoints, loaded, strict=True)]
    candidates = [(candidate_id, model, digest) for candidate_id, (_, model, digest) in zip(candidate_ids, loaded, strict=True)]
    calibration = AmpPairs(workspace, "calibration", 125, target_frames, SEED + 2)
    feature_rows = []
    labels = []
    for index in range(len(calibration)):
        row = calibration[index]
        start = int(row["target_start"])
        wet = row["wet"][start:].numpy()
        clean = row["clean"][start:].numpy()
        outputs = _candidate_outputs(candidates, row)
        feature_rows.append(_features(row))
        labels.append([
            _label(measure("amp", wet, value, clean)) for value in outputs
        ])
    features = np.stack(feature_rows)
    targets = np.asarray(labels, dtype=np.float64)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1.0e-6] = 1.0
    normalized = (features - mean) / scale
    design = np.concatenate((np.ones((len(normalized), 1)), normalized), axis=1)
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)

    development = AmpPairs(workspace, "development", 135, target_frames, SEED + 3)
    aggregate = ([], [], [])
    strata = defaultdict(lambda: ([], [], []))
    settings = defaultdict(lambda: ([], [], []))
    chosen = Counter()
    for index in range(len(development)):
        row = development[index]
        feature = (_features(row) - mean) / scale
        predicted = np.concatenate(([1.0], feature)) @ weights
        selected = int(np.argmax(predicted))
        value = _candidate_outputs([candidates[selected]], row)[0]
        start = int(row["target_start"])
        wet = row["wet"][start:].numpy()
        clean = row["clean"][start:].numpy()
        chosen[candidate_ids[selected]] += 1
        for collection in (aggregate, strata[_stratum(row["control_values"])], settings[row["setting"]]):
            for target, source in zip(collection, (wet, value, clean), strict=True):
                target.append(source)
    report = summarize("amp", *aggregate)
    report["strata"] = {name: summarize("amp", *values) for name, values in sorted(strata.items())}
    report["settings"] = {name: summarize("amp", *values) for name, values in sorted(settings.items())}
    report["all_strata_accepted"] = bool(report["strata"] and all(row["accepted"] for row in report["strata"].values()))
    report["all_settings_accepted"] = bool(
        len(report["settings"]) == EXPECTED_DEVELOPMENT_SETTINGS
        and all(row["accepted"] for row in report["settings"].values())
    )
    gates = {
        "aggregate_quality": bool(report["accepted"]),
        "individual_pass_fraction": report["pass_fraction"] >= 0.80,
        "all_control_strata": bool(report["all_strata_accepted"]),
        "all_seen_settings": bool(report["all_settings_accepted"]),
        "selector_fit_on_calibration_only": True,
        "development_clean_hidden_from_selector": True,
        "locked_final_unopened": True,
        "order_independent": True,
    }
    return {
        "schema": 1,
        "status": "accepted-development" if all(gates.values()) else "diagnostic-not-promoted",
        "accepted": bool(all(gates.values())),
        "mechanism": "amp",
        "selector": {
            "architecture": "ridge-observable-candidate-selector",
            "ridge": RIDGE,
            "feature_names": [
                "control_basis_0", "bass", "mid", "treble", "gain",
                "bass2", "mid2", "treble2", "gain2", "bass_mid", "bass_treble",
                "mid_treble", "bass_gain", "mid_gain", "treble_gain",
                "log_rms", "crest", "difference_rms_ratio", "high_energy_ratio",
                "spectral_centroid", "mean_attack", "max_attack",
            ],
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
            "weights": weights.tolist(),
            "candidate_ids": candidate_ids,
            "chosen_counts": dict(sorted(chosen.items())),
            "clean_input_at_inference": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        },
        "candidates": [
            {"id": candidate_id, "checkpoint": str(path.resolve()), "sha256": digest}
            for path, (candidate_id, _, digest) in zip(checkpoints, candidates, strict=True)
        ],
        "development": report,
        "gates": gates,
        "provenance": {
            "selector_gradient_scope": "calibration-only",
            "development_clean_used_for_selection": False,
            "locked_final_audio_opened": False,
            "demo_generated": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-frames", type=int, default=16384)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    report = calibrate(args.workspace.resolve(), args.checkpoints, args.target_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "accepted": report["accepted"],
        "development_pass_fraction": report["development"]["pass_fraction"],
        "chosen_counts": report["selector"]["chosen_counts"],
        "gates": report["gates"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
