#!/usr/bin/env python3
"""Calibration-only Clean-oracle audit of frequency-dependent RIR shortening."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .ambience2 import AmbienceAbstention
from .ambience4 import AmbiencePairsV4, TARGET_FRAMES_MINIMUM
from .ambience4_frequency_shaping import (
    FREQUENCY_SHAPING_CANDIDATES,
    profile_frequency_shortening_inverse,
)
from .product_data import _rir
from .quality2 import measure as measure_universal
from .reverb_quality import measure as measure_tail
from .train_ambience2 import _summaries


SEED = 20260908
CANDIDATES = FREQUENCY_SHAPING_CANDIDATES


def _oracle_rank(wet: np.ndarray, restored: np.ndarray, clean: np.ndarray) -> tuple:
    universal = measure_universal("ambience", wet, restored, clean)
    tail = measure_tail(wet, restored, clean)
    required = ("spectrum", "high_band", "transient")
    reductions = [universal["metrics"][name]["reduction"] for name in required]
    nonregression = [
        universal["metrics"][name]["nonregression"]
        for name in ("spectrum", "high_band", "transient", "dynamics")
    ]
    return (
        int(universal["gates"]["no_new_clipping"]),
        int(all(nonregression)),
        int(tail.get("passed", False)),
        int(universal["passed"]),
        sum(bool(value) for value in nonregression),
        sum(bool(universal["gates"][name]) for name in required),
        float(tail.get("tail_envelope_esr_improvement", -1.0)),
        float(tail.get("tail_excess_reduction", -1.0)),
        min(reductions),
        float(np.mean(reductions)),
    )


def _summary(rows: list[dict], contract: str) -> dict:
    return _summaries((
        [row["wet"] for row in rows],
        [row["restored"] for row in rows],
        [row["clean"] for row in rows],
    ), contract)


def audit(workspace: Path, samples: int) -> dict:
    dataset = AmbiencePairsV4(
        workspace, "calibration", samples, TARGET_FRAMES_MINIMUM, SEED,
        include_late_base=False,
    )
    rows = []
    selections: Counter[str] = Counter()
    abstentions: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    elapsed = 0.0
    for index in range(len(dataset)):
        row = dataset[index]
        start = int(row["target_start"])
        wet_full = row["wet"].numpy()
        wet = wet_full[start:]
        clean = row["clean"][start:].numpy()
        impulse = _rir(dataset.rir_root / row["rir"])
        candidates = [("wet", wet)]
        for candidate in CANDIDATES:
            parameters = {key: value for key, value in candidate.items() if key != "id"}
            started = time.perf_counter()
            try:
                restored, _ = profile_frequency_shortening_inverse(
                    wet_full, impulse, row["control_values"], **parameters
                )
                candidates.append((candidate["id"], restored[start:]))
            except AmbienceAbstention as error:
                abstentions[f"{candidate['id']}: {error}"] += 1
            except (ValueError, RuntimeError) as error:
                errors[f"{candidate['id']}: {error}"] += 1
            elapsed += time.perf_counter() - started
        selected_id, selected = max(
            candidates, key=lambda item: _oracle_rank(wet, item[1], clean)
        )
        selections[selected_id] += 1
        rows.append({
            "wet": wet,
            "restored": np.asarray(selected, dtype=np.float32),
            "clean": clean,
            "source": str(row["source_id"]),
            "room": dataset.room_group(row),
            "decay": dataset.decay_stratum(
                float(row["control_values"]["decay_p999_seconds"])
            ),
        })

    aggregate = _summary(rows, dataset.quality_contract)
    groups = {}
    for axis in ("source", "room", "decay"):
        buckets = defaultdict(list)
        for row in rows:
            buckets[row[axis]].append(row)
        groups[axis] = {
            name: _summary(values, dataset.quality_contract)
            for name, values in sorted(buckets.items())
        }
    gates = {
        "aggregate": bool(aggregate["accepted"]),
        "each_source": all(row["accepted"] for row in groups["source"].values()),
        "each_room": all(row["accepted"] for row in groups["room"].values()),
        "each_decay_stratum": all(row["accepted"] for row in groups["decay"].values()),
        "nonidentity_reachable": selections["wet"] < samples,
        "no_unexpected_candidate_error": not errors,
    }
    return {
        "schema": 1,
        "status": "diagnostic-calibration-clean-oracle-not-deployable",
        "accepted": all(gates.values()),
        "promotable": False,
        "purpose": "frequency-dependent-known-profile-family-viability-only",
        "quality_contract": dataset.quality_contract,
        "selection": {
            "split": "calibration",
            "clean_input_at_runtime": True,
            "candidate_parameters_frozen_before_audit": True,
            "development_opened": False,
            "selected_examples": dict(sorted(selections.items())),
        },
        "profile_contract": {
            "profile_required": True,
            "profile_inputs": ["rir", "mix", "room_gain_db"],
            "candidate_runtime_clean_input": False,
            "chain_order_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        },
        "method": {
            "candidates": list(CANDIDATES),
            "frequency_decay_estimator": "per-bin Schroeder energy decay T30 extrapolated to T60",
            "desired_response": "frequency-bin-specific exponential fade preserving direct and early response",
            "inverse": "finite-lookahead bounded regularized profile deconvolution",
            "implementation_origin": "local clean-room implementation from published method description",
            "reference": {
                "title": "Dereverberation Filter by Deconvolution with Frequency Bin Specific Faded Impulse Response",
                "url": "https://arxiv.org/abs/2601.06662",
            },
            "third_party_runtime_code": False,
        },
        "samples": samples,
        "gates": gates,
        "aggregate": aggregate,
        "sources": groups["source"],
        "rooms": groups["room"],
        "decay_strata": groups["decay"],
        "errors": dict(sorted(errors.items())),
        "fail_closed_abstentions": dict(sorted(abstentions.items())),
        "runtime": {
            "total_candidate_seconds": elapsed,
            "mean_seconds_per_example_all_candidates": elapsed / samples,
            "ordinary_cpu": True,
            "audio_callback": False,
        },
        "provenance": {
            "clean_audio": "existing product-authorized Product4 calibration rows",
            "rir_source": "but-reverbdb",
            "generated_audio_written": False,
            "locked_final_opened": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/foundation/product4-reverb-frequency-shaping-oracle-v1/metrics.json"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace frequency shaping audit: {output}")
    report = audit(args.workspace.resolve(), args.samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "metrics": str(output),
        "accepted": report["accepted"],
        "gates": report["gates"],
        "selected_examples": report["selection"]["selected_examples"],
        "aggregate": {
            "universal_pass_fraction": report["aggregate"]["universal"]["pass_fraction"],
            "tail": report["aggregate"]["tail"],
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
