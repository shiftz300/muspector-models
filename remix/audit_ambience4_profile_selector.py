#!/usr/bin/env python3
"""Calibrate a Wet-observable safety selector for Product4 profile inversion."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .ambience2 import AmbienceAbstention
from .ambience3 import profile_inverse_base, wet_candidate_tail_ratio
from .ambience4 import AmbiencePairsV4, TARGET_FRAMES_MINIMUM
from .product_data import _rir
from .train_ambience2 import _summaries


THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
MINIMUM_COVERAGE = 0.50


def _materialize(workspace: Path, split: str, samples: int, seed: int) -> tuple[list[dict], str]:
    dataset = AmbiencePairsV4(
        workspace, split, samples, TARGET_FRAMES_MINIMUM, seed, include_late_base=False
    )
    rows = []
    for index in range(len(dataset)):
        row = dataset[index]
        try:
            restored, profile = profile_inverse_base(
                row["wet"].numpy(),
                _rir(dataset.rir_root / row["rir"]),
                row["control_values"],
            )
        except AmbienceAbstention as error:
            rows.append({"error": str(error)})
            continue
        start = int(row["target_start"])
        rows.append({
            "wet": row["wet"][start:].numpy(),
            "restored": restored[start:].astype(np.float32),
            "clean": row["clean"][start:].numpy(),
            "source": str(row["source_id"]),
            "room": dataset.room_group(row),
            "decay": dataset.decay_stratum(
                float(row["control_values"]["decay_p999_seconds"])
            ),
            "mode": str(profile["mode"]),
            "tail_ratio": (
                0.0
                if profile["mode"] == "exact"
                else wet_candidate_tail_ratio(row["wet"].numpy(), restored, start)
            ),
        })
    return rows, dataset.quality_contract


def _summary(rows: list[dict], contract: str) -> dict:
    if not rows:
        return {"accepted": False, "reason": "no accepted examples"}
    values = (
        [row["wet"] for row in rows],
        [row["restored"] for row in rows],
        [row["clean"] for row in rows],
    )
    return _summaries(values, contract)


def evaluate(rows: list[dict], contract: str, threshold: float) -> dict:
    valid = [row for row in rows if "error" not in row]
    selected = [
        row for row in valid
        if row["mode"] == "exact" or row["tail_ratio"] <= threshold
    ]
    grouped = {}
    for axis in ("source", "room", "decay"):
        attempted = sorted({row[axis] for row in valid})
        reports = {}
        for name in attempted:
            attempts = [row for row in valid if row[axis] == name]
            accepted = [row for row in selected if row[axis] == name]
            report = _summary(accepted, contract)
            report["attempted_examples"] = len(attempts)
            report["accepted_examples"] = len(accepted)
            report["coverage"] = len(accepted) / len(attempts)
            reports[name] = report
        grouped[axis] = reports
    aggregate = _summary(selected, contract)
    fallback = _summary([row for row in selected if row["mode"] != "exact"], contract)
    coverage = len(selected) / len(rows)
    gates = {
        "aggregate": bool(aggregate.get("accepted")),
        "fallback": bool(fallback.get("accepted")),
        "coverage": coverage >= MINIMUM_COVERAGE,
        "each_source": bool(grouped["source"]) and all(
            report.get("accepted", False) for report in grouped["source"].values()
        ),
        "each_room": bool(grouped["room"]) and all(
            report.get("accepted", False) for report in grouped["room"].values()
        ),
        "each_decay_stratum": bool(grouped["decay"]) and all(
            report.get("accepted", False) for report in grouped["decay"].values()
        ),
        "no_inverse_error": len(valid) == len(rows),
    }
    return {
        "accepted": all(gates.values()),
        "threshold": threshold,
        "coverage": coverage,
        "accepted_examples": len(selected),
        "accepted_modes": {
            mode: sum(row["mode"] == mode for row in selected)
            for mode in sorted({row["mode"] for row in valid})
        },
        "gates": gates,
        "aggregate": aggregate,
        "fallback": fallback,
        "sources": grouped["source"],
        "rooms": grouped["room"],
        "decay_strata": grouped["decay"],
        "inverse_errors": [row["error"] for row in rows if "error" in row],
    }


def _rank(report: dict) -> tuple:
    return (
        int(report["accepted"]),
        sum(report["gates"].values()),
        report["coverage"],
        -report["threshold"],
    )


def audit(workspace: Path, calibration_samples: int, development_samples: int) -> dict:
    calibration_rows, contract = _materialize(
        workspace, "calibration", calibration_samples, 20260906
    )
    candidate_reports = {
        str(threshold): evaluate(calibration_rows, contract, threshold)
        for threshold in THRESHOLDS
    }
    selected_threshold = max(
        THRESHOLDS, key=lambda value: _rank(candidate_reports[str(value)])
    )
    development_rows, development_contract = _materialize(
        workspace, "development", development_samples, 20260907
    )
    if development_contract != contract:
        raise ValueError("calibration/development quality contracts differ")
    development = evaluate(development_rows, contract, selected_threshold)
    accepted = bool(candidate_reports[str(selected_threshold)]["accepted"] and development["accepted"])
    return {
        "schema": 1,
        "status": "accepted-selective-known-profile-development" if accepted else "rejected-known-profile-selector",
        "accepted": accepted,
        "mechanism": "ambience",
        "quality_contract": contract,
        "selector": {
            "feature": "wet-to-profile-candidate-post-activity-tail-envelope-ratio",
            "threshold_candidates": list(THRESHOLDS),
            "selected_threshold": selected_threshold,
            "selection_split": "calibration",
            "clean_input_at_runtime": False,
            "unknown_profile_decision": "abstain",
            "chain_order_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        },
        "calibration": {
            "samples": calibration_samples,
            "candidates": candidate_reports,
            "selected": candidate_reports[str(selected_threshold)],
        },
        "development": {"samples": development_samples, **development},
        "provenance": {
            "rir_source": "but-reverbdb",
            "locked_final_opened": False,
            "generated_audio_written": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--calibration-samples", type=int, default=96)
    parser.add_argument("--development-samples", type=int, default=96)
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product4-reverb-known-profile-selector-v1/metrics.json"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace profile selector audit: {output}")
    report = audit(
        args.workspace.resolve(), args.calibration_samples, args.development_samples
    )
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "metrics": str(output),
        "status": report["status"],
        "threshold": report["selector"]["selected_threshold"],
        "calibration": {
            "accepted": report["calibration"]["selected"]["accepted"],
            "coverage": report["calibration"]["selected"]["coverage"],
        },
        "development": {
            "accepted": report["development"]["accepted"],
            "coverage": report["development"]["coverage"],
            "gates": report["development"]["gates"],
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
