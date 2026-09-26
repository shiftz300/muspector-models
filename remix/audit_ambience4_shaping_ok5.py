#!/usr/bin/env python3
"""Blind external-room audit of the frozen Product4 shaping candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .ambience2 import HISTORY_FRAMES, AmbienceAbstention, _render, restore_known_ambience_profile
from .ambience4 import (
    PRODUCT4_CLEAN_SOURCE_IDS,
    TARGET_FRAMES_MINIMUM,
    AmbiencePairsV4,
    release_event_frames,
)
from .ambience4_shaping import profile_shortening_inverse
from .ok5_rir_data import SOURCE_ID, discover_ok5_rirs, load_ok5_rir
from .product2 import _condition_clean
from .product_data import Clean, _read, discover_clean


SEED = 20260908
QUALITY_CONTRACT = "tail-removal-with-global-nonregression"
FROZEN_CANDIDATE = {
    "id": "early25-rt40-low024-band2k3k",
    "early_ms": 25.0,
    "target_rt60_ms": 40.0,
    "maximum_gain": 4.0,
    "low_band_strength": 0.24,
    "high_band_strength": 0.0,
    "low_band_end_hz": 2_000.0,
    "high_band_start_hz": 3_000.0,
}


def _calibration_seal(workspace: Path) -> dict:
    path = (
        workspace
        / "runs/foundation/product4-reverb-profile-shaping-calibration-v2/metrics.json"
    ).resolve()
    payload = path.read_bytes()
    report = json.loads(payload)
    if report["status"] != "accepted-profile-shaping-calibration-candidate":
        raise ValueError("frozen shaping calibration is not accepted")
    if report["selection"]["selected"] != FROZEN_CANDIDATE:
        raise ValueError("frozen shaping parameters differ from calibration")
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def _clean_buckets(workspace: Path) -> dict[str, tuple[Clean, ...]]:
    buckets: dict[str, list[Clean]] = defaultdict(list)
    for item in discover_clean(workspace):
        if item.split == "development" and item.source_id in PRODUCT4_CLEAN_SOURCE_IDS:
            buckets[item.source_id].append(item)
    result = {name: tuple(rows) for name, rows in sorted(buckets.items()) if rows}
    if len(result) < 2:
        raise ValueError("OK5 audit requires at least two development Clean sources")
    return result


def _event_clean(
    rows: tuple[Clean, ...], source: str, total_frames: int, seed: int
) -> tuple[Clean, np.ndarray]:
    start = seed % len(rows)
    best = -1
    searched = min(len(rows), 64)
    for offset in range(searched):
        selected = rows[(start + offset) % len(rows)]
        for crop in range(8):
            value = _read(
                selected,
                total_frames,
                seed + offset * 32_452_843 + crop * 49_979_687,
            )
            score = release_event_frames(value[HISTORY_FRAMES:])
            best = max(best, score)
            if score >= 4:
                return selected, value
    raise ValueError(
        f"OK5 audit found no measurable release event for {source}; "
        f"searched_files={searched} best_frames={best}"
    )


def _materialize(workspace: Path, crops_per_source: int) -> tuple[list[dict], float, dict]:
    rir_root = (workspace / "data/corpus/ok5-rir").resolve()
    rirs, inventory = discover_ok5_rirs(rir_root)
    clean_buckets = _clean_buckets(workspace)
    total_frames = HISTORY_FRAMES + TARGET_FRAMES_MINIMUM
    parameters = {key: value for key, value in FROZEN_CANDIDATE.items() if key != "id"}
    rows = []
    elapsed = 0.0
    for rir_index, record in enumerate(rirs):
        impulse = load_ok5_rir(record.path, record.measurement)
        decay = AmbiencePairsV4.decay_stratum(record.decay_p999_seconds)
        room_measurement = f"{record.room}:m{record.measurement}"
        for source_index, (source, clean_rows) in enumerate(clean_buckets.items()):
            for crop in range(crops_per_source):
                seed = (
                    SEED
                    + rir_index * 15_485_863
                    + source_index * 32_452_843
                    + crop * 49_979_687
                )
                rng = random.Random(seed)
                selected, clean = _event_clean(
                    clean_rows, source, total_frames, rng.getrandbits(63)
                )
                clean, _ = _condition_clean(clean, rng)
                wet, controls = _render(clean, impulse, rng)
                started = time.perf_counter()
                try:
                    try:
                        restored, runtime = restore_known_ambience_profile(
                            wet, impulse, controls
                        )
                        mode = "exact"
                    except AmbienceAbstention:
                        restored, runtime = profile_shortening_inverse(
                            wet, impulse, controls, **parameters
                        )
                        mode = "shaping"
                except AmbienceAbstention as error:
                    rows.append({
                        "source": source,
                        "room": record.room,
                        "room_measurement": room_measurement,
                        "decay": decay,
                        "clean_group": selected.group,
                        "error": str(error),
                    })
                    continue
                elapsed += time.perf_counter() - started
                rows.append({
                    "wet": wet[HISTORY_FRAMES:],
                    "restored": restored[HISTORY_FRAMES:],
                    "clean": clean[HISTORY_FRAMES:],
                    "source": source,
                    "room": record.room,
                    "room_measurement": room_measurement,
                    "decay": decay,
                    "clean_group": selected.group,
                    "mode": mode,
                    "runtime_contract": runtime,
                })
    inventory = {
        **inventory,
        "development_clean_sources": list(clean_buckets),
        "crops_per_source_per_measurement": crops_per_source,
        "attempted_examples": len(rows),
    }
    return rows, elapsed, inventory


def audit(workspace: Path, crops_per_source: int) -> dict:
    if crops_per_source < 2:
        raise ValueError("external audit requires at least two crops per source and RIR")
    from .audit_ambience4_shaping import _evaluate

    calibration = _calibration_seal(workspace)
    rows, elapsed, inventory = _materialize(workspace, crops_per_source)
    evaluation = _evaluate(rows, QUALITY_CONTRACT)
    fallback_tail = evaluation["shaping_fallback"].get("tail", {})
    frozen_safety_gates = {
        "all_standard_external_gates": bool(evaluation["accepted"]),
        "fallback_pass_fraction": fallback_tail.get("pass_fraction", 0.0) >= 0.75,
        "fallback_median_tail_reduction": (
            fallback_tail.get("median_tail_excess_reduction", 0.0) >= 0.30
        ),
        "fallback_never_adds_reverb": fallback_tail.get("added_reverb_fraction", 1.0) == 0.0,
    }
    accepted = all(frozen_safety_gates.values())
    return {
        "schema": 1,
        "status": (
            "accepted-profile-shaping-external-rooms"
            if accepted else "rejected-profile-shaping-external-rooms"
        ),
        "accepted": accepted,
        "promotable": False,
        "promotion_blockers": [
            "partitioned streaming runtime equivalence and timing are not sealed",
            "listening acceptance and locked-final remain unopened",
        ],
        "mechanism": "ambience",
        "quality_contract": QUALITY_CONTRACT,
        "frozen_candidate": FROZEN_CANDIDATE,
        "frozen_safety_gates": frozen_safety_gates,
        "external_validation": evaluation,
        "inventory": inventory,
        "profile_contract": {
            "profile_required": True,
            "profile_inputs": ["rir", "mix", "room_gain_db"],
            "clean_input": False,
            "unknown_profile_decision": "abstain",
            "chain_order_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        },
        "selection_contract": {
            "dataset": SOURCE_ID,
            "external_data_used_for_parameter_selection": False,
            "candidate_count": 1,
            "parameters_frozen_before_external_data_evaluation": True,
            "calibration_seal": calibration,
        },
        "runtime": {
            "total_seconds": elapsed,
            "mean_seconds_per_attempt": elapsed / len(rows),
            "ordinary_cpu": True,
            "audio_callback": False,
        },
        "provenance": {
            "rir_source": SOURCE_ID,
            "receiver_channel": 0,
            "locked_final_opened": False,
            "generated_audio_written": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--crops-per-source", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/foundation/product4-reverb-profile-shaping-ok5-external-v1/metrics.json"
        ),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace OK5 external audit: {output}")
    report = audit(args.workspace.resolve(), args.crops_per_source)
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "metrics": str(output),
        "status": report["status"],
        "accepted": report["accepted"],
        "attempted_examples": report["inventory"]["attempted_examples"],
        "modes": report["external_validation"]["modes"],
        "gates": report["external_validation"]["gates"],
        "frozen_safety_gates": report["frozen_safety_gates"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
