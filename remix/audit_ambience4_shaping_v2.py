#!/usr/bin/env python3
"""Freeze the narrower-band Product4 response-shortening calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit_ambience4_shaping import _evaluate, _materialize


CANDIDATES = tuple(
    {
        "id": f"early25-rt40-low{int(strength * 100):03d}-band2k3k",
        "early_ms": 25.0,
        "target_rt60_ms": 40.0,
        "maximum_gain": 4.0,
        "low_band_strength": strength,
        "high_band_strength": 0.0,
        "low_band_end_hz": 2_000.0,
        "high_band_start_hz": 3_000.0,
    }
    for strength in (0.20, 0.22, 0.24, 0.26)
)


def _safe_candidate(report: dict) -> bool:
    tail = report["shaping_fallback"].get("tail", {})
    return bool(
        report["accepted"]
        and tail.get("pass_fraction", 0.0) >= 0.75
        and tail.get("median_tail_excess_reduction", 0.0) >= 0.30
        and tail.get("added_reverb_fraction", 1.0) == 0.0
    )


def audit(workspace: Path, samples: int) -> dict:
    candidates = {}
    elapsed = 0.0
    contract = None
    for candidate in CANDIDATES:
        rows, candidate_contract, seconds = _materialize(
            workspace, "calibration", samples, 20260906, candidate
        )
        if contract is not None and contract != candidate_contract:
            raise ValueError("calibration quality contracts differ")
        contract = candidate_contract
        elapsed += seconds
        candidates[candidate["id"]] = _evaluate(rows, candidate_contract)
    safe = [row for row in CANDIDATES if _safe_candidate(candidates[row["id"]])]
    selected = min(safe, key=lambda row: row["low_band_strength"]) if safe else None
    accepted = selected is not None
    return {
        "schema": 1,
        "status": (
            "accepted-profile-shaping-calibration-candidate"
            if accepted else "rejected-profile-shaping-calibration"
        ),
        "accepted": accepted,
        "promotable": False,
        "promotion_blockers": [
            "canonical development rooms were consumed by the rejected wider-band v1",
            "new room-disjoint external validation is required",
            "partitioned streaming runtime, listening, and locked-final are not sealed",
        ],
        "mechanism": "ambience",
        "quality_contract": contract,
        "selection": {
            "split": "calibration",
            "rule": "lowest strength with all subgroup gates, fallback pass >= 0.75, median tail reduction >= 0.30, and zero added reverb",
            "selected": selected,
            "development_used": False,
        },
        "calibration": {
            "samples": samples,
            "candidates": candidates,
            "selected": None if selected is None else candidates[selected["id"]],
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
        "runtime": {
            "calibration_total_seconds": elapsed,
            "ordinary_cpu": True,
            "audio_callback": False,
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
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/foundation/product4-reverb-profile-shaping-calibration-v2/metrics.json"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace shaping calibration: {output}")
    report = audit(args.workspace.resolve(), args.samples)
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "metrics": str(output),
        "status": report["status"],
        "selected": None if report["selection"]["selected"] is None else report["selection"]["selected"]["id"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
