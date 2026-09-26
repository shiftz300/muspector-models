#!/usr/bin/env python3
"""Calibration-frozen observable gate for the Product4 analytic gray box."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .ambience3 import wet_candidate_tail_ratio
from .ambience4 import release_event_frames
from .ambience5_profile_direct import AmbiencePairsV4MixedProfileDirect
from .foundation_data import RATE
from .reverb_quality import envelope
from .reverb_quality2 import measure, summarize
from .train_ambience4_profile_direct import SEED


OBSERVABLE_FEATURE_NAMES = (
    "log_tail_ratio", "correction_rms_ratio", "candidate_rms_ratio",
    "wet_peak", "candidate_peak", "waveform_correlation",
    "wet_release_fraction", "candidate_release_fraction",
    "wet_envelope_dynamic_range", "candidate_envelope_dynamic_range",
    "spectral_log_ratio_mean", "spectral_log_ratio_std",
    "spectral_log_ratio_q10", "spectral_log_ratio_q90",
    "spectral_log_ratio_low", "spectral_log_ratio_high",
    "control_mix", "control_room_gain", "control_decay",
    "profile_log_magnitude_mean", "profile_log_magnitude_std",
    "profile_decay_mean", "profile_decay_std", "exact_mode",
)


def observable_features(item: dict, score: float) -> np.ndarray:
    """Runtime-available scalar evidence; deliberately excludes Clean/source/order."""
    start = int(item["target_start"])
    wet = item["wet"][start:].numpy().astype(np.float64)
    candidate = item["analytic_base"][start:].numpy().astype(np.float64)
    wet_rms = math.sqrt(float(np.mean(np.square(wet))) + 1.0e-12)
    candidate_rms = math.sqrt(float(np.mean(np.square(candidate))) + 1.0e-12)
    correction_rms = math.sqrt(float(np.mean(np.square(candidate - wet))) + 1.0e-12)
    correlation = float(np.dot(wet, candidate) / (
        math.sqrt(float(np.dot(wet, wet) * np.dot(candidate, candidate))) + 1.0e-12
    ))
    wet_env = envelope(wet.astype(np.float32)).astype(np.float64)
    candidate_env = envelope(candidate.astype(np.float32)).astype(np.float64)

    def dynamic_range(value: np.ndarray) -> float:
        return float(np.log((np.quantile(value, 0.90) + 1.0e-8) / (
            np.quantile(value, 0.10) + 1.0e-8
        )))

    wet_spectrum = np.abs(np.fft.rfft(wet))
    candidate_spectrum = np.abs(np.fft.rfft(candidate))
    floor = max(1.0e-8, 1.0e-4 * float(np.max(wet_spectrum)))
    spectral_ratio = np.log((candidate_spectrum + floor) / (wet_spectrum + floor))
    frequencies = np.fft.rfftfreq(len(wet), 1.0 / RATE)
    low = frequencies <= 2_000.0
    high = frequencies >= 4_000.0
    profile = item["profile_features"].numpy().astype(np.float64)
    controls = item["controls"].numpy().astype(np.float64)
    values = np.asarray((
        np.log(min(max(score, 1.0e-6), 1.0e4)),
        correction_rms / wet_rms,
        candidate_rms / wet_rms,
        float(np.max(np.abs(wet))),
        float(np.max(np.abs(candidate))),
        correlation,
        release_event_frames(wet) / max(len(wet_env), 1),
        release_event_frames(candidate) / max(len(candidate_env), 1),
        dynamic_range(wet_env),
        dynamic_range(candidate_env),
        float(np.mean(spectral_ratio)),
        float(np.std(spectral_ratio)),
        float(np.quantile(spectral_ratio, 0.10)),
        float(np.quantile(spectral_ratio, 0.90)),
        float(np.mean(spectral_ratio[low])),
        float(np.mean(spectral_ratio[high])),
        *controls.tolist(),
        float(np.mean(profile[0])),
        float(np.std(profile[0])),
        float(np.mean(profile[3])),
        float(np.std(profile[3])),
        float(item["analytic_mode"] == "exact"),
    ), dtype=np.float32)
    if values.shape != (len(OBSERVABLE_FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("observable gate feature vector changed or became non-finite")
    return values


def _rows(dataset) -> list[dict]:
    rows = []
    for index in range(len(dataset)):
        item = dataset[index]
        start = int(item["target_start"])
        wet_full = item["wet"].numpy()
        candidate_full = item["analytic_base"].numpy()
        wet = wet_full[start:]
        candidate = candidate_full[start:]
        clean = item["clean"][start:].numpy()
        result = measure(wet, candidate, clean)
        score = wet_candidate_tail_ratio(wet_full, candidate_full, start)
        features = observable_features(item, score)
        rows.append({
            "index": index,
            "score": float(score),
            "candidate_result": result,
            "candidate_effective": result["decision"] == "effective-restoration",
            "observable_features": features,
            "source_id": item["source_id"],
            "rir_source_id": item["rir_source_id"],
            "room_group": dataset.room_group(item),
            "decay_stratum": dataset.decay_stratum(
                float(item["control_values"]["decay_p999_seconds"])
            ),
            "analytic_mode": item["analytic_mode"],
            "wet": wet,
            "candidate": candidate,
            "clean": clean,
        })
    return rows


def _threshold(rows: list[dict]) -> float | None:
    """Largest score whose selected calibration prefix contains no failure."""
    finite = sorted(set(row["score"] for row in rows if math.isfinite(row["score"])))
    safe = None
    for threshold in finite:
        selected = [row for row in rows if row["score"] <= threshold]
        if selected and all(row["candidate_effective"] for row in selected):
            safe = threshold
        elif selected:
            break
    return safe


def _deploy(rows: list[dict], thresholds: dict[str, float | None]) -> dict:
    results = []
    groups = defaultdict(list)
    public_rows = []
    for row in rows:
        threshold_key = f"{row['analytic_mode']}:{row['decay_stratum']}"
        threshold = thresholds.get(threshold_key)
        selected = threshold is not None and row["score"] <= threshold
        restored = row["candidate"] if selected else row["wet"]
        result = measure(row["wet"], restored, row["clean"])
        results.append(result)
        groups[f"source:{row['source_id']}"].append(result)
        groups[f"room:{row['room_group']}"].append(result)
        groups[f"decay:{row['decay_stratum']}"].append(result)
        public_rows.append({
            key: row[key]
            for key in (
                "index", "score", "candidate_effective", "source_id", "rir_source_id",
                "room_group", "decay_stratum", "analytic_mode",
            )
        } | {"selected": selected, "decision": result["decision"]})
    aggregate = summarize(results)
    group_reports = {name: summarize(values) for name, values in sorted(groups.items())}
    accepted = bool(aggregate["accepted"] and all(row["accepted"] for row in group_reports.values()))
    return {
        "accepted": accepted,
        "aggregate": aggregate,
        "groups": group_reports,
        "rows": public_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/foundation/product4-reverb-profile-direct-v4-observable-gate-v1.json"),
    )
    parser.add_argument("--fit-samples", type=int, default=192)
    parser.add_argument("--calibration-samples", type=int, default=192)
    parser.add_argument("--target-frames", type=int, default=65_536)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace profile-direct gate audit: {output}")
    if args.fit_samples < 48 or args.calibration_samples < 48:
        raise ValueError("profile-direct gate audit needs at least 48 rows per partition")
    workspace = args.workspace.resolve()
    fit = AmbiencePairsV4MixedProfileDirect(
        workspace, "fit", args.fit_samples, args.target_frames, SEED + 1
    )
    calibration = AmbiencePairsV4MixedProfileDirect(
        workspace, "calibration", args.calibration_samples, args.target_frames, SEED + 2
    )
    fit_rows = _rows(fit)
    calibration_rows = _rows(calibration)
    modes = sorted(set(
        f"{row['analytic_mode']}:{row['decay_stratum']}"
        for row in fit_rows + calibration_rows
    ))
    thresholds = {
        mode: _threshold([
            row for row in calibration_rows
            if f"{row['analytic_mode']}:{row['decay_stratum']}" == mode
        ])
        for mode in modes
    }
    fit_report = _deploy(fit_rows, thresholds)
    calibration_report = _deploy(calibration_rows, thresholds)
    accepted = calibration_report["accepted"]
    report = {
        "schema": 1,
        "status": "accepted-calibration-observable-gate" if accepted else "rejected-calibration-observable-gate",
        "accepted": accepted,
        "deployable_gate": True,
        "selection_input": "Wet and current analytic profile inverse only",
        "score": "observable candidate-to-Wet post-activity envelope ratio",
        "threshold_selection_partition": "calibration",
        "evaluation_partition": "calibration",
        "thresholds_by_analytic_mode_and_decay": thresholds,
        "fit": fit_report,
        "calibration": calibration_report,
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
        "thresholds": thresholds,
        "fit": fit_report["aggregate"],
        "calibration": calibration_report["aggregate"],
        "failed_groups": [
            name for name, row in calibration_report["groups"].items()
            if not row["accepted"]
        ],
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
