#!/usr/bin/env python3
"""Calibrate and audit finite known-profile response shortening for Product4."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from .ambience2 import AmbienceAbstention, restore_known_ambience_profile
from .ambience4 import AmbiencePairsV4, TARGET_FRAMES_MINIMUM
from .ambience4_shaping import profile_shortening_inverse
from .product_data import _rir
from .train_ambience2 import _summaries


CANDIDATES = (
    {
        "id": "early10-rt160-low030-high000",
        "early_ms": 10.0,
        "target_rt60_ms": 160.0,
        "maximum_gain": 4.0,
        "low_band_strength": 0.30,
        "high_band_strength": 0.00,
    },
    {
        "id": "early25-rt40-low030-high000",
        "early_ms": 25.0,
        "target_rt60_ms": 40.0,
        "maximum_gain": 4.0,
        "low_band_strength": 0.30,
        "high_band_strength": 0.00,
    },
    {
        "id": "early10-rt160-low025-high003",
        "early_ms": 10.0,
        "target_rt60_ms": 160.0,
        "maximum_gain": 4.0,
        "low_band_strength": 0.25,
        "high_band_strength": 0.03,
    },
)
MINIMUM_COVERAGE = 0.50


def _summary(rows: list[dict], contract: str) -> dict:
    if not rows:
        return {"accepted": False, "reason": "no admitted examples"}
    return _summaries((
        [row["wet"] for row in rows],
        [row["restored"] for row in rows],
        [row["clean"] for row in rows],
    ), contract)


def _evaluate(rows: list[dict], contract: str) -> dict:
    valid = [row for row in rows if "error" not in row]
    grouped = {}
    for axis in ("source", "room", "decay"):
        reports = {}
        for name in sorted({row[axis] for row in rows}):
            attempts = [row for row in rows if row[axis] == name]
            admitted = [row for row in valid if row[axis] == name]
            report = _summary(admitted, contract)
            report["attempted_examples"] = len(attempts)
            report["admitted_examples"] = len(admitted)
            report["coverage"] = len(admitted) / len(attempts)
            reports[name] = report
        grouped[axis] = reports
    aggregate = _summary(valid, contract)
    shaping = _summary([row for row in valid if row["mode"] == "shaping"], contract)
    coverage = len(valid) / len(rows)
    gates = {
        "aggregate": bool(aggregate.get("accepted")),
        "shaping_fallback": bool(shaping.get("accepted")),
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
        "coverage": coverage,
        "admitted_examples": len(valid),
        "modes": dict(Counter(row["mode"] for row in valid)),
        "gates": gates,
        "aggregate": aggregate,
        "shaping_fallback": shaping,
        "sources": grouped["source"],
        "rooms": grouped["room"],
        "decay_strata": grouped["decay"],
        "errors": [row["error"] for row in rows if "error" in row],
    }


def _materialize(
    workspace: Path, split: str, samples: int, seed: int, candidate: dict
) -> tuple[list[dict], str, float]:
    dataset = AmbiencePairsV4(
        workspace, split, samples, TARGET_FRAMES_MINIMUM, seed,
        include_late_base=False,
    )
    rows = []
    elapsed = 0.0
    parameters = {key: value for key, value in candidate.items() if key != "id"}
    for index in range(len(dataset)):
        row = dataset[index]
        source = str(row["source_id"])
        room = dataset.room_group(row)
        decay = dataset.decay_stratum(float(row["control_values"]["decay_p999_seconds"]))
        impulse = _rir(dataset.rir_root / row["rir"])
        started = time.perf_counter()
        try:
            try:
                restored, _ = restore_known_ambience_profile(
                    row["wet"].numpy(), impulse, row["control_values"]
                )
                mode = "exact"
            except AmbienceAbstention:
                restored, _ = profile_shortening_inverse(
                    row["wet"].numpy(), impulse, row["control_values"], **parameters
                )
                mode = "shaping"
        except AmbienceAbstention as error:
            rows.append({
                "source": source,
                "room": room,
                "decay": decay,
                "error": str(error),
            })
            continue
        elapsed += time.perf_counter() - started
        start = int(row["target_start"])
        rows.append({
            "wet": row["wet"][start:].numpy(),
            "restored": restored[start:],
            "clean": row["clean"][start:].numpy(),
            "source": source,
            "room": room,
            "decay": decay,
            "mode": mode,
        })
    return rows, dataset.quality_contract, elapsed


def _rank(report: dict) -> tuple:
    tail = report["shaping_fallback"].get("tail", {})
    return (
        int(report["accepted"]),
        sum(report["gates"].values()),
        report["coverage"],
        -float(tail.get("added_reverb_fraction", 1.0)),
        float(tail.get("pass_fraction", 0.0)),
        float(tail.get("median_tail_excess_reduction", -1.0)),
    )


def audit(workspace: Path, calibration_samples: int, development_samples: int) -> dict:
    calibration = {}
    calibration_seconds = 0.0
    contract = None
    for candidate in CANDIDATES:
        rows, candidate_contract, elapsed = _materialize(
            workspace, "calibration", calibration_samples, 20260906, candidate
        )
        if contract is not None and candidate_contract != contract:
            raise ValueError("calibration quality contracts differ")
        contract = candidate_contract
        calibration_seconds += elapsed
        calibration[candidate["id"]] = _evaluate(rows, candidate_contract)
    selected = max(CANDIDATES, key=lambda row: _rank(calibration[row["id"]]))
    development_rows, development_contract, development_seconds = _materialize(
        workspace, "development", development_samples, 20260907, selected
    )
    if development_contract != contract:
        raise ValueError("calibration/development quality contracts differ")
    development = _evaluate(development_rows, development_contract)
    selected_calibration = calibration[selected["id"]]
    accepted = bool(selected_calibration["accepted"] and development["accepted"])
    return {
        "schema": 1,
        "status": (
            "accepted-profile-shaping-upper-bound-development"
            if accepted else "rejected-profile-shaping-upper-bound"
        ),
        "accepted": accepted,
        "promotable": False,
        "promotion_blockers": [
            "finite-kernel streaming/partitioned runtime is not sealed",
            "listening acceptance and locked-final remain unopened",
        ],
        "mechanism": "ambience",
        "quality_contract": contract,
        "selection": {
            "split": "calibration",
            "candidate_ids": [row["id"] for row in CANDIDATES],
            "selected": selected,
            "development_used_for_selection": False,
        },
        "profile_contract": {
            "profile_required": True,
            "profile_inputs": ["rir", "mix", "room_gain_db"],
            "clean_input": False,
            "unknown_profile_decision": "abstain",
            "chain_order_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        },
        "calibration": {
            "samples": calibration_samples,
            "candidates": calibration,
            "selected": selected_calibration,
        },
        "development": {"samples": development_samples, **development},
        "runtime": {
            "calibration_total_seconds": calibration_seconds,
            "development_total_seconds": development_seconds,
            "development_mean_seconds_per_example": development_seconds / development_samples,
            "ordinary_cpu": True,
            "audio_callback": False,
        },
        "method": {
            "objective": "preserve direct/early response and shorten late decay with bounded partial equalization",
            "implementation_origin": "local clean-room implementation; no third-party code copied",
            "references": [
                {
                    "title": "Multi-Channel Room Impulse Response Shaping - A Study",
                    "url": "https://doi.org/10.1109/ICASSP.2006.1661222",
                },
                {
                    "title": "Application of Channel Shortening to Acoustic Channel Equalization in the Presence of Noise and Estimation Error",
                    "url": "https://doi.org/10.1109/ASPAA.2011.6082286",
                },
            ],
            "third_party_runtime_code": False,
        },
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
        default=Path("runs/foundation/product4-reverb-profile-shaping-upper-bound-v1/metrics.json"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace shaping audit: {output}")
    report = audit(args.workspace.resolve(), args.calibration_samples, args.development_samples)
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "metrics": str(output),
        "status": report["status"],
        "selected": report["selection"]["selected"]["id"],
        "calibration_accepted": report["calibration"]["selected"]["accepted"],
        "development_accepted": report["development"]["accepted"],
        "development_gates": report["development"]["gates"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
