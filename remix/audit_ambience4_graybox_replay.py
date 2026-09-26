#!/usr/bin/env python3
"""Audit whether exact re-convolution can safely identify Reverb inverses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, stft

from .ambience5_profile_direct import AmbiencePairsV4MixedProfileDirect
from .audit_ambience4_profile_direct_gate import _rows
from .foundation_data import RATE
from .train_ambience4_profile_direct import SEED


FEATURES = (
    "replay_rms_ratio",
    "replay_peak_ratio",
    "replay_log_spectral_error",
    "transfer_minimum_magnitude",
    "transfer_condition_number",
    "candidate_correction_rms_ratio",
    "candidate_peak_ratio",
)


def _replay_features(item: dict) -> dict[str, float]:
    start = int(item["target_start"])
    wet = item["wet"].numpy().astype(np.float64)
    candidate = item["analytic_base"].numpy().astype(np.float64)
    transfer = item["transfer"].numpy().astype(np.float64)
    replay = fftconvolve(candidate, transfer, mode="full")[: len(wet)]
    wet_target = wet[start:]
    candidate_target = candidate[start:]
    replay_target = replay[start:]
    wet_rms = math.sqrt(float(np.mean(np.square(wet_target))) + 1.0e-12)
    replay_error = replay_target - wet_target
    fft_size = 1 << (len(transfer) - 1).bit_length()
    response = np.abs(np.fft.rfft(transfer, fft_size))
    response_floor = max(float(response.max()) * 1.0e-8, 1.0e-12)
    _, _, wet_spectrum = stft(
        wet_target, fs=RATE, nperseg=1024, noverlap=768,
        boundary=None, padded=False,
    )
    _, _, replay_spectrum = stft(
        replay_target, fs=RATE, nperseg=1024, noverlap=768,
        boundary=None, padded=False,
    )
    magnitude_floor = max(float(np.abs(wet_spectrum).max()) * 1.0e-5, 1.0e-10)
    spectral_error = np.log(np.abs(replay_spectrum) + magnitude_floor) - np.log(
        np.abs(wet_spectrum) + magnitude_floor
    )
    return {
        "replay_rms_ratio": math.sqrt(float(np.mean(np.square(replay_error))) + 1.0e-18) / wet_rms,
        "replay_peak_ratio": float(np.max(np.abs(replay_error))) / max(float(np.max(np.abs(wet_target))), 1.0e-8),
        "replay_log_spectral_error": math.sqrt(float(np.mean(np.square(spectral_error)))),
        "transfer_minimum_magnitude": float(response.min()),
        "transfer_condition_number": float(response.max() / max(float(response.min()), response_floor)),
        "candidate_correction_rms_ratio": math.sqrt(float(np.mean(np.square(candidate_target - wet_target))) + 1.0e-18) / wet_rms,
        "candidate_peak_ratio": float(np.max(np.abs(candidate_target))) / max(float(np.max(np.abs(wet_target))), 1.0e-8),
    }


def _tag(dataset) -> list[dict]:
    labels = _rows(dataset)
    tagged = []
    for index, label in enumerate(labels):
        item = dataset[index]
        tagged.append({
            "index": index,
            "source_id": label["source_id"],
            "rir_source_id": label["rir_source_id"],
            "room_group": label["room_group"],
            "decay_stratum": label["decay_stratum"],
            "analytic_mode": label["analytic_mode"],
            "effective": bool(label["candidate_effective"]),
            **_replay_features(item),
        })
    return tagged


def _threshold(fit: list[dict], feature: str, direction: str) -> float | None:
    ordered = sorted(
        fit, key=lambda row: row[feature], reverse=direction == "greater"
    )
    selected = []
    for row in ordered:
        if not row["effective"]:
            break
        selected.append(row)
    if not selected:
        return None
    boundary = selected[-1][feature]
    if len(selected) == len(ordered):
        return boundary
    rejected = ordered[len(selected)][feature]
    return 0.5 * (boundary + rejected)


def _evaluate(rows: list[dict], feature: str, direction: str, threshold: float | None) -> dict:
    if threshold is None:
        selected = []
    elif direction == "less":
        selected = [row for row in rows if row[feature] <= threshold]
    else:
        selected = [row for row in rows if row[feature] >= threshold]
    positives = [row for row in rows if row["effective"]]
    true = sum(row["effective"] for row in selected)
    false = len(selected) - true
    return {
        "selected": len(selected),
        "true_positive": true,
        "false_positive": false,
        "precision": true / max(len(selected), 1),
        "effective_coverage": true / max(len(positives), 1),
    }


def _distribution(rows: list[dict], feature: str, effective: bool) -> dict:
    values = np.asarray(
        [row[feature] for row in rows if row["effective"] is effective], dtype=np.float64
    )
    return {
        "count": len(values),
        "minimum": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product4-reverb-graybox-replay-audit-v1.json"),
    )
    parser.add_argument("--fit-samples", type=int, default=192)
    parser.add_argument("--calibration-samples", type=int, default=192)
    parser.add_argument("--target-frames", type=int, default=65_536)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace replay audit: {output}")
    workspace = args.workspace.resolve()
    fit_dataset = AmbiencePairsV4MixedProfileDirect(
        workspace, "fit", args.fit_samples, args.target_frames, SEED + 1
    )
    calibration_dataset = AmbiencePairsV4MixedProfileDirect(
        workspace, "calibration", args.calibration_samples, args.target_frames, SEED + 2
    )
    fit = _tag(fit_dataset)
    calibration = _tag(calibration_dataset)
    calibration_sources = {row["source_id"] for row in calibration}
    fit = [row for row in fit if row["source_id"] in calibration_sources]
    screens = {}
    for feature in FEATURES:
        candidates = []
        for direction in ("less", "greater"):
            threshold = _threshold(fit, feature, direction)
            candidates.append({
                "direction": direction,
                "threshold": threshold,
                "fit": _evaluate(fit, feature, direction, threshold),
                "calibration": _evaluate(calibration, feature, direction, threshold),
            })
        screens[feature] = max(
            candidates,
            key=lambda row: (
                row["calibration"]["precision"] == 1.0,
                row["calibration"]["effective_coverage"],
                row["fit"]["effective_coverage"],
            ),
        ) | {
            "distribution": {
                "fit_effective": _distribution(fit, feature, True),
                "fit_ineffective": _distribution(fit, feature, False),
                "calibration_effective": _distribution(calibration, feature, True),
                "calibration_ineffective": _distribution(calibration, feature, False),
            }
        }
    best_name, best = max(
        screens.items(),
        key=lambda item: (
            item[1]["calibration"]["precision"] == 1.0,
            item[1]["calibration"]["effective_coverage"],
        ),
    )
    accepted = bool(
        best["calibration"]["false_positive"] == 0
        and best["calibration"]["effective_coverage"] >= 0.50
    )
    report = {
        "schema": 1,
        "status": "replay-gate-capacity-passed" if accepted else "replay-gate-capacity-rejected",
        "accepted": accepted,
        "purpose": "fit-derived scalar graybox replay gate capacity audit",
        "best_feature": best_name,
        "best_screen": best,
        "screens": screens,
        "fit_rows": len(fit),
        "calibration_rows": len(calibration),
        "development_opened": False,
        "listening_rows_used": False,
        "fresh_rochester_opened": False,
        "locked_final_accessed": False,
        "order_contract": {
            "chain_order_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "best_feature": best_name,
        "best_screen": best,
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
