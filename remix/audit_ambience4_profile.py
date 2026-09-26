#!/usr/bin/env python3
"""Audit the deployable known-profile Reverb inverse on Product4 data."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .ambience2 import AmbienceAbstention
from .ambience3 import profile_inverse_base
from .ambience4 import AmbiencePairsV4, TARGET_FRAMES_MINIMUM
from .product_data import _rir
from .train_ambience2 import _summaries


SEED = 20260907
MINIMUM_SELECTIVE_COVERAGE = 0.50


def _empty():
    return ([], [], [])


def _add(collection, wet, restored, clean) -> None:
    collection[0].append(wet)
    collection[1].append(restored)
    collection[2].append(clean)


def _summary(rows, contract: str) -> dict:
    if not rows[0]:
        return {"accepted": False, "reason": "no examples"}
    return _summaries(rows, contract)


def audit(workspace: Path, samples: int) -> dict:
    dataset = AmbiencePairsV4(
        workspace, "development", samples, TARGET_FRAMES_MINIMUM, SEED,
        include_late_base=False,
    )
    accepted = _empty()
    exact = _empty()
    fallback = _empty()
    sources = defaultdict(_empty)
    rooms = defaultdict(_empty)
    decays = defaultdict(_empty)
    attempted_sources = Counter()
    accepted_sources = Counter()
    attempted_rooms = Counter()
    accepted_rooms = Counter()
    attempted_decays = Counter()
    accepted_decays = Counter()
    modes = Counter()
    abstentions = []
    elapsed = 0.0
    for index in range(len(dataset)):
        row = dataset[index]
        source = str(row["source_id"])
        room = dataset.room_group(row)
        decay = dataset.decay_stratum(
            float(row["control_values"]["decay_p999_seconds"])
        )
        attempted_sources[source] += 1
        attempted_rooms[room] += 1
        attempted_decays[decay] += 1
        impulse = _rir(dataset.rir_root / row["rir"])
        started = time.perf_counter()
        try:
            restored, profile = profile_inverse_base(
                row["wet"].numpy(), impulse, row["control_values"]
            )
        except AmbienceAbstention as error:
            abstentions.append({
                "index": index,
                "source_id": source,
                "room": room,
                "decay_stratum": decay,
                "reason": str(error),
            })
            continue
        elapsed += time.perf_counter() - started
        mode = str(profile["mode"])
        # The regularized path remains diagnostic until it independently passes;
        # the exact path is the only initially admitted selective profile inverse.
        if mode != "exact":
            start = int(row["target_start"])
            _add(
                fallback,
                row["wet"][start:].numpy(), restored[start:], row["clean"][start:].numpy(),
            )
            modes[mode] += 1
            continue
        modes[mode] += 1
        accepted_sources[source] += 1
        accepted_rooms[room] += 1
        accepted_decays[decay] += 1
        start = int(row["target_start"])
        values = (
            row["wet"][start:].numpy(),
            restored[start:].astype(np.float32),
            row["clean"][start:].numpy(),
        )
        for collection in (accepted, exact, sources[source], rooms[room], decays[decay]):
            _add(collection, *values)
    contract = dataset.quality_contract
    aggregate = _summary(accepted, contract)
    exact_report = _summary(exact, contract)
    fallback_report = _summary(fallback, contract)

    def grouped(rows, attempted, admitted):
        result = {}
        for name in sorted(attempted):
            report = _summary(rows[name], contract)
            report["attempted_examples"] = attempted[name]
            report["accepted_examples"] = admitted[name]
            report["coverage"] = admitted[name] / attempted[name]
            result[name] = report
        return result

    source_reports = grouped(sources, attempted_sources, accepted_sources)
    room_reports = grouped(rooms, attempted_rooms, accepted_rooms)
    decay_reports = grouped(decays, attempted_decays, accepted_decays)
    coverage = len(accepted[0]) / len(dataset)
    gates = {
        "aggregate": bool(aggregate.get("accepted")),
        "exact_path": bool(exact_report.get("accepted")),
        "coverage": coverage >= MINIMUM_SELECTIVE_COVERAGE,
        "each_source": bool(source_reports) and all(
            row.get("accepted", False) for row in source_reports.values()
        ),
        "each_room": bool(room_reports) and all(
            row.get("accepted", False) for row in room_reports.values()
        ),
        "each_decay_stratum": bool(decay_reports) and all(
            row.get("accepted", False) for row in decay_reports.values()
        ),
        "fallback_not_admitted": True,
        "no_unhandled_abstention": not abstentions,
    }
    return {
        "schema": 1,
        "status": "accepted-selective-known-profile-development" if all(gates.values()) else "rejected-known-profile-development",
        "accepted": all(gates.values()),
        "mechanism": "ambience",
        "split": "development",
        "samples": samples,
        "target_frames": TARGET_FRAMES_MINIMUM,
        "quality_contract": contract,
        "profile_contract": {
            "profile_required": True,
            "profile_inputs": ["rir", "mix", "room_gain_db"],
            "clean_input": False,
            "wet_only_when_profile_is_fixed": True,
            "unknown_profile_decision": "abstain",
            "regularized_fallback_admitted": False,
            "chain_order_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        },
        "coverage": coverage,
        "accepted_examples": len(accepted[0]),
        "accepted_modes": dict(modes),
        "gates": gates,
        "aggregate": aggregate,
        "exact": exact_report,
        "fallback_diagnostic": fallback_report,
        "sources": source_reports,
        "rooms": room_reports,
        "decay_strata": decay_reports,
        "abstentions": abstentions,
        "runtime": {
            "total_seconds": elapsed,
            "mean_seconds_per_attempt": elapsed / samples,
            "ordinary_cpu": True,
            "audio_callback": False,
        },
        "provenance": {
            "rir_source": "but-reverbdb",
            "authorization": dataset.authorization,
            "locked_final_opened": False,
            "generated_audio_written": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product4-reverb-known-profile-v1/metrics.json"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace profile audit: {output}")
    report = audit(args.workspace.resolve(), args.samples)
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "metrics": str(output),
        "status": report["status"],
        "coverage": report["coverage"],
        "gates": report["gates"],
        "modes": report["accepted_modes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
