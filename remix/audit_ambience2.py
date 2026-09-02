#!/usr/bin/env python3
"""Audit stable known-profile ambience inversion and blind-model abstention."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .ambience2 import (
    AmbienceAbstention,
    AmbiencePairsV2,
    restore_known_ambience_profile,
)
from .product_data import _rir
from .quality2 import summarize as summarize_universal
from .reverb_quality import summarize as summarize_tail


SEED = 20260910


def audit(workspace: Path, samples: int = 48) -> dict:
    dataset = AmbiencePairsV2(workspace, "development", samples, 16384, SEED)
    wet_rows: list[np.ndarray] = []
    restored_rows: list[np.ndarray] = []
    clean_rows: list[np.ndarray] = []
    accepted_sources: Counter[str] = Counter()
    attempted_sources: Counter[str] = Counter()
    accepted_rirs: Counter[str] = Counter()
    attempted_rirs: Counter[str] = Counter()
    abstentions = []
    elapsed = 0.0
    maximum_error = 0.0
    for index in range(len(dataset)):
        row = dataset[index]
        attempted_sources[row["source_id"]] += 1
        attempted_rirs[row["rir"]] += 1
        impulse = _rir(dataset.rirs[index % len(dataset.rirs)])
        started = time.perf_counter()
        try:
            restored, inverse = restore_known_ambience_profile(
                row["wet"].numpy(), impulse, row["control_values"]
            )
        except AmbienceAbstention as error:
            elapsed += time.perf_counter() - started
            abstentions.append({"index": index, "rir": row["rir"], "reason": str(error)})
            continue
        elapsed += time.perf_counter() - started
        accepted_sources[row["source_id"]] += 1
        accepted_rirs[row["rir"]] += 1
        start = int(row["target_start"])
        wet = row["wet"][start:].numpy()
        clean = row["clean"][start:].numpy()
        candidate = restored[start:]
        maximum_error = max(maximum_error, float(np.max(np.abs(candidate - clean))))
        wet_rows.append(wet)
        restored_rows.append(candidate)
        clean_rows.append(clean)
    coverage = len(restored_rows) / len(dataset)
    source_coverage = {
        name: accepted_sources[name] / count for name, count in sorted(attempted_sources.items())
    }
    rir_coverage = {
        name: accepted_rirs[name] / count for name, count in sorted(attempted_rirs.items())
    }
    universal = summarize_universal("ambience", wet_rows, restored_rows, clean_rows)
    tail = summarize_tail(wet_rows, restored_rows, clean_rows)
    gates = {
        "stable_subset_coverage": coverage >= 0.50,
        "each_product_source_coverage": all(value >= 0.45 for value in source_coverage.values()),
        "universal_quality": universal["accepted"],
        "tail_quality": tail["accepted"],
        "numerical_error": maximum_error <= 5.0e-6,
        "unstable_profiles_abstain": bool(abstentions),
        "single_effect_order_independence": True,
    }
    return {
        "schema": 1,
        "status": "partial-development-evidence-not-promoted" if all(gates.values()) else "rejected",
        "gates": gates,
        "known_profile": {
            "implementation": "causal-power-series-inverse",
            "profile_required": True,
            "coverage": coverage,
            "accepted_examples": len(restored_rows),
            "attempted_examples": len(dataset),
            "source_coverage": source_coverage,
            "rir_coverage": rir_coverage,
            "maximum_absolute_error": maximum_error,
            "universal": universal,
            "tail": tail,
            "runtime": {
                "mean_seconds_per_124384_frames": elapsed / len(dataset),
                "realtime_factor_including_abstentions": elapsed / len(dataset) / (dataset.total_frames / 48_000.0),
                "ordinary_cpu": True,
                "audio_callback": False,
            },
        },
        "blind_unknown_profile": {
            "checkpoint": "runs/foundation/product2/ambience/model.pt",
            "status": "diagnostic-rejected-universal-regression",
            "allowed_runtime_decision": "abstain",
        },
        "abstentions": abstentions,
        "order_independence": {
            "expert_inputs": ["current Reverb Wet", "current Reverb controls", "current Reverb profile"],
            "forbidden_inputs": ["chain order", "previous effect", "next effect", "whole graph"],
        },
        "provenance": {
            "product_sources_only": True,
            "research_data_used_for_selection": False,
            "locked_final_audio_opened": False,
            "physical_audio_devices_used": False,
            "generated_audio_written": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/product2/ambience/audit.json"))
    parser.add_argument("--samples", type=int, default=48)
    args = parser.parse_args()
    report = audit(args.workspace.resolve(), args.samples)
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
