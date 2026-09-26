#!/usr/bin/env python3
"""Clean-oracle upper bound for the frozen frequency-profile candidate bank."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .ambience4 import AmbiencePairsV4MixedFrequencyProfileBank
from .ambience4_frequency_shaping import FREQUENCY_SHAPING_CANDIDATES
from .reverb_quality2 import measure, summarize
from .train_ambience4_frequency_profile import SEED


def _score(row: dict) -> float:
    return (
        float(row["tail"].get("tail_excess_reduction", 0.0))
        + float(row["active_foreground"].get("reduction", 0.0))
        + sum(float(value["reduction"]) for value in row["universal"]["metrics"].values())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/foundation/product4-reverb-frequency-profile-"
            "acceptance2-oracle-calibration-v1.json"
        ),
    )
    parser.add_argument("--samples", type=int, default=192)
    args = parser.parse_args()
    if args.samples < 24:
        raise ValueError("acceptance2 oracle needs at least 24 calibration examples")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace acceptance2 oracle: {output}")
    dataset = AmbiencePairsV4MixedFrequencyProfileBank(
        args.workspace.resolve(), "calibration", args.samples, 65_536, SEED
    )
    selected_results = []
    compact_rows = []
    groups = defaultdict(list)
    selections = Counter()
    for index in range(len(dataset)):
        item = dataset[index]
        start = int(item["target_start"])
        wet = item["wet"][start:].numpy()
        clean = item["clean"][start:].numpy()
        candidates = []
        for candidate_index, restored in enumerate(item["late_base"][:, start:].numpy()):
            result = measure(wet, restored, clean)
            if result["decision"] == "effective-restoration":
                candidates.append((_score(result), candidate_index, result))
        if candidates:
            _, selected_index, selected = max(candidates, key=lambda row: (row[0], -row[1]))
            selected_id = FREQUENCY_SHAPING_CANDIDATES[selected_index]["id"]
        else:
            selected_index = None
            selected_id = "wet-hard-bypass"
            selected = measure(wet, wet.copy(), clean)
        selected_results.append(selected)
        selections[selected_id] += 1
        decay = dataset.decay_stratum(
            float(item["control_values"]["decay_p999_seconds"])
        )
        room = dataset.room_group(item)
        groups[f"source:{item['source_id']}"].append(selected)
        groups[f"room:{room}"].append(selected)
        groups[f"decay:{decay}"].append(selected)
        compact_rows.append({
            "index": index,
            "source_id": item["source_id"],
            "rir_source_id": item["rir_source_id"],
            "room_group": room,
            "decay_stratum": decay,
            "profile_mode": item["profile_mode"],
            "selected_candidate_index": selected_index,
            "selected_candidate": selected_id,
            "decision": selected["decision"],
            "evidence_measurable": selected["evidence_measurable"],
        })
    aggregate = summarize(selected_results)
    group_reports = {name: summarize(values) for name, values in sorted(groups.items())}
    accepted = bool(aggregate["accepted"] and all(row["accepted"] for row in group_reports.values()))
    report = {
        "schema": 1,
        "status": (
            "accepted-training-only-clean-oracle"
            if accepted else "rejected-training-only-clean-oracle"
        ),
        "accepted": accepted,
        "partition": "calibration",
        "candidate_count": len(FREQUENCY_SHAPING_CANDIDATES),
        "candidate_variants": list(FREQUENCY_SHAPING_CANDIDATES),
        "selection": "Clean may select only an individually effective candidate; otherwise exact Wet bypass",
        "deployable": False,
        "clean_input_at_runtime": False,
        "development_or_listening_rows_used": False,
        "aggregate": aggregate,
        "groups": group_reports,
        "selection_counts": dict(sorted(selections.items())),
        "rows": compact_rows,
        "locked_final_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "accepted": accepted,
        "aggregate": aggregate,
        "failed_groups": [name for name, row in group_reports.items() if not row["accepted"]],
        "selection_counts": report["selection_counts"],
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
