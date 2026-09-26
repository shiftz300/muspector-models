#!/usr/bin/env python3
"""Audit EG-IPT DI to SM57 physical Amp+cab+mic pairs before admission."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile
from scipy.signal import correlate, correlation_lags


RATE = 96_000
EXPECTED_PAIRS = 8_717
EXPECTED_MISSING = {
    "EG-IPT/HB-neck/DI/bottleneck/HB-neck_bottleneck-slow_179-02_DI.wav",
    "EG-IPT/HB-neck/DI/bottleneck/HB-neck_bottleneck-slow_179-04_DI.wav",
    "EG-IPT/HB-neck/DI/bottleneck/HB-neck_bottleneck-slow_179-06_DI.wav",
}


def _wet_relative(clean_relative: Path) -> Path:
    parts = list(clean_relative.parts)
    parts[parts.index("DI")] = "dyn"
    parts[-1] = parts[-1].replace("_DI.wav", "_dyn.wav")
    return Path(*parts)


def _read(path: Path) -> np.ndarray:
    value, rate = soundfile.read(path, dtype="float32", always_2d=True)
    if rate != RATE or value.shape[1] != 1:
        raise ValueError(f"invalid EG-IPT audio geometry: {path}: {rate}, {value.shape}")
    result = np.asarray(value[:, 0], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite EG-IPT audio: {path}")
    return result


def _alignment(clean: np.ndarray, wet: np.ndarray) -> dict:
    frames = min(len(clean), len(wet), 3 * RATE)
    clean = clean[:frames].astype(np.float64)
    wet = wet[:frames].astype(np.float64)
    clean_feature = np.diff(clean, prepend=clean[0])
    wet_feature = np.diff(wet, prepend=wet[0])
    clean_feature -= clean_feature.mean()
    wet_feature -= wet_feature.mean()
    values = correlate(wet_feature, clean_feature, mode="full", method="fft")
    lags = correlation_lags(len(wet_feature), len(clean_feature), mode="full")
    selected = np.abs(lags) <= 1_024
    values, lags = values[selected], lags[selected]
    peak = int(np.argmax(np.abs(values)))
    denominator = max(
        np.linalg.norm(clean_feature) * np.linalg.norm(wet_feature), 1.0e-12
    )
    return {
        "lag_frames": int(lags[peak]),
        "absolute_derivative_correlation": float(abs(values[peak]) / denominator),
    }


def audit(clean_root: Path, wet_root: Path) -> dict:
    clean_root, wet_root = clean_root.resolve(), wet_root.resolve()
    clean_files = sorted(clean_root.glob("EG-IPT/*/DI/*/*_DI.wav"))
    if len(clean_files) != 8_720:
        raise ValueError(f"expected 8720 retained DI files, found {len(clean_files)}")
    pairs = []
    missing = []
    for clean in clean_files:
        relative = clean.relative_to(clean_root)
        wet = wet_root / _wet_relative(relative)
        if wet.exists():
            pairs.append((relative, clean, wet))
        else:
            missing.append(relative.as_posix())
    if set(missing) != EXPECTED_MISSING or len(pairs) != EXPECTED_PAIRS:
        raise ValueError(f"unexpected EG-IPT missing pairs: {missing}")

    geometry_failures = []
    clipped = {"clean": 0, "wet": 0}
    meaningful = 0
    total_frames = 0
    manifest = hashlib.sha256()
    strata = defaultdict(list)
    for relative, clean_path, wet_path in pairs:
        clean_info = soundfile.info(clean_path)
        wet_info = soundfile.info(wet_path)
        if (
            clean_info.samplerate != RATE or wet_info.samplerate != RATE
            or clean_info.channels != 1 or wet_info.channels != 1
            or clean_info.frames != wet_info.frames
        ):
            geometry_failures.append(relative.as_posix())
            continue
        clean = _read(clean_path)
        wet = _read(wet_path)
        clipped["clean"] += int(np.count_nonzero(np.abs(clean) >= 0.999))
        clipped["wet"] += int(np.count_nonzero(np.abs(wet) >= 0.999))
        clean_rms = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2)))
        wet_rms = float(np.sqrt(np.mean(wet.astype(np.float64) ** 2)))
        difference_rms = float(np.sqrt(np.mean((wet.astype(np.float64) - clean) ** 2)))
        meaningful += int(
            max(clean_rms, wet_rms) >= 1.0e-4
            and difference_rms >= 0.05 * max(clean_rms, wet_rms)
        )
        total_frames += len(clean)
        manifest.update(relative.as_posix().encode())
        manifest.update(str(clean_info.frames).encode())
        pickup, technique = relative.parts[1], relative.parts[3]
        strata[(pickup, technique)].append((relative, clean_path, wet_path))

    alignment_rows = []
    for (pickup, technique), rows in sorted(strata.items()):
        for index in sorted({0, len(rows) // 2, len(rows) - 1}):
            relative, clean_path, wet_path = rows[index]
            result = _alignment(_read(clean_path), _read(wet_path))
            alignment_rows.append({
                "file": relative.as_posix(), "pickup": pickup,
                "technique": technique, **result,
            })
    lags = np.asarray([row["lag_frames"] for row in alignment_rows], dtype=np.int64)
    correlations = [row["absolute_derivative_correlation"] for row in alignment_rows]
    broadband_rows = [
        row for row in alignment_rows if row["technique"] in {"scratch", "snap-pizz"}
    ]
    broadband_lags = np.asarray(
        [row["lag_frames"] for row in broadband_rows], dtype=np.int64
    )
    median_lag = int(np.median(broadband_lags))
    broadband_deviations = np.abs(broadband_lags - median_lag)
    broadband_correlations = [
        row["absolute_derivative_correlation"] for row in broadband_rows
    ]
    gates = {
        "expected_pairs": len(pairs) == EXPECTED_PAIRS,
        "known_missing_only": set(missing) == EXPECTED_MISSING,
        "geometry": not geometry_failures,
        "meaningful_effect_fraction": meaningful / len(pairs) >= 0.99,
        "no_clean_clipping": clipped["clean"] == 0,
        "wet_clipping_fraction_below_0_1_percent": clipped["wet"] / total_frames < 0.001,
        "broadband_alignment_p95_within_1ms": (
            float(np.quantile(broadband_deviations, 0.95)) <= 96
        ),
        "alignment_median_correlation": float(np.median(correlations)) >= 0.20,
        "broadband_alignment_median_correlation": (
            float(np.median(broadband_correlations)) >= 0.35
        ),
    }
    accepted = all(gates.values())
    return {
        "schema": 1,
        "status": "audited-product-pair-eligible" if accepted else "rejected-pair-audit",
        "accepted": accepted,
        "source_id": "eg-ipt",
        "scope": "fixed EVH 5150 III 50W 6L6 plus Mesa 4x12 V30 plus close SM57",
        "license": "CC-BY-4.0",
        "pairs": len(pairs),
        "missing_pairs": missing,
        "duration_hours": total_frames / RATE / 3600.0,
        "pair_manifest_sha256": manifest.hexdigest(),
        "geometry_failures": geometry_failures,
        "meaningful_effect_fraction": meaningful / len(pairs),
        "clipped_samples": clipped,
        "alignment": {
            "sampled_pairs": len(alignment_rows),
            "broadband_transient_pairs": len(broadband_rows),
            "median_lag_frames_at_96khz": median_lag,
            "broadband_p95_deviation_frames": float(
                np.quantile(broadband_deviations, 0.95)
            ),
            "broadband_maximum_deviation_frames": int(broadband_deviations.max()),
            "median_absolute_derivative_correlation": float(np.median(correlations)),
            "minimum_absolute_derivative_correlation": float(min(correlations)),
            "all_technique_lag_p05_p95_frames": [
                float(np.quantile(lags, 0.05)), float(np.quantile(lags, 0.95))
            ],
            "method": (
                "global acquisition lag is estimated only from broadband scratch and snap-pizz "
                "transients; pitched-technique cross-correlation lag is reported but not gated "
                "because Amp/cab phase and group delay are part of the effect to invert"
            ),
            "rows": alignment_rows,
        },
        "gates": gates,
        "limitations": [
            "one player, one guitar, one amplifier/cabinet setting and one close microphone",
            "mostly isolated monophonic techniques; all pairs are fit-only if admitted",
            "cannot by itself provide source-disjoint development or broad Amp generalization",
        ],
        "graph_order_input": False,
        "neighbor_effect_input": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--wet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.clean_root, args.wet_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "accepted": report["accepted"],
        "pairs": report["pairs"], "duration_hours": report["duration_hours"],
        "alignment": {key: value for key, value in report["alignment"].items() if key != "rows"},
        "gates": report["gates"],
    }, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
