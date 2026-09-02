#!/usr/bin/env python3
"""Calibrate, challenge, and package the safe Stone-style Phaser subdomain."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .phaser3 import (
    StonePhaserPairsV3, estimate_phase, estimate_phase_diagnostics,
    invert_known_phase, manifest, _analog_triangle
)
from .quality2 import measure as quality_measure
from .train_chorus3 import trajrow
from .train_modulation3 import _summarize_rows


SEED = 20260927
MARGIN_THRESHOLDS = (0.02, 0.04, 0.075, 0.10, 0.15)
CONSENSUS_THRESHOLDS = (0.35, 0.70, 1.05, 1.40)
MINIMUM_COVERAGE = 0.25
MINIMUM_PER_SOURCE = 4


def _precompute(data: StonePhaserPairsV3) -> list[dict]:
    rows = []
    for index in range(len(data)):
        source = data[index]
        wet = source["wet"].numpy()
        clean = source["clean"].numpy()
        values = source["control_values"]
        diagnostics = estimate_phase_diagnostics(wet, values)
        phase, score, margin = diagnostics["phase"], diagnostics["score"], diagnostics["margin"]
        restored = invert_known_phase(wet, values, phase)
        time_axis = np.arange(len(wet), dtype=np.float64) / 48_000
        predicted_lfo = _analog_triangle(values["rate_hz"] * time_axis + phase / (2.0 * np.pi))
        quality = quality_measure("modulation", wet, restored, clean)
        restored_clip_fraction = float(np.mean(np.abs(restored) >= 0.999))
        wet_clip_fraction = float(np.mean(np.abs(wet) >= 0.999))
        output_safe = bool(
            float(np.max(np.abs(restored))) <= 1.0
            and restored_clip_fraction <= wet_clip_fraction + 1.0e-4
        )
        rows.append({
            "index": index,
            "source_id": source["source_id"],
            "group": source["group"],
            "wet": wet,
            "restored": restored,
            "clean": clean,
            "trajectory": trajrow(predicted_lfo, source["lfo"].numpy()),
            "quality": quality,
            "rate_hz": values["rate_hz"],
            "mix": values["mix"],
            "color": values["color"],
            "phase": phase,
            "phase_error": abs(((phase - values["hidden_phase_radians"] + math.pi) % (2.0 * math.pi)) - math.pi),
            "score": score,
            "margin": margin,
            "band_consensus_p90_radians": diagnostics["band_consensus_p90_radians"],
            "band_consensus_median_radians": diagnostics["band_consensus_median_radians"],
            "output_safe": output_safe,
            "restored_peak": float(np.max(np.abs(restored))),
            "wet_clip_fraction": wet_clip_fraction,
            "restored_clip_fraction": restored_clip_fraction,
            "values": values,
        })
    return rows


def _report(
    rows: list[dict], margin_threshold: float, consensus_threshold: float, attempted: int
) -> dict:
    admitted = [
        row for row in rows
        if row["margin"] >= margin_threshold
        and row["band_consensus_p90_radians"] <= consensus_threshold
        and row["output_safe"]
    ]
    if not admitted:
        return {"accepted": False, "coverage": 0.0, "reason": "no admitted examples"}
    aggregate = ([row["wet"] for row in admitted], [row["restored"] for row in admitted], [row["clean"] for row in admitted])
    trajectories = [row["trajectory"] for row in admitted]
    report = _summarize_rows(aggregate, trajectories)
    report["attempted_examples"] = attempted
    report["accepted_examples"] = len(admitted)
    report["coverage"] = len(admitted) / attempted
    report["all_individual_quality_passed"] = all(row["quality"]["passed"] for row in admitted)
    report["phase_error_median_radians"] = float(np.median([row["phase_error"] for row in admitted]))
    report["phase_error_p90_radians"] = float(np.quantile([row["phase_error"] for row in admitted], 0.90))
    report["minimum_margin"] = float(min(row["margin"] for row in admitted))
    report["sources"] = {}
    report["strata"] = {}
    attempted_sources = Counter(row["source_id"] for row in rows)
    attempted_strata = Counter(f"color-{int(row['color'])}" for row in rows)
    for destination, key_name, attempted_counts in (
        (report["sources"], "source_id", attempted_sources),
        (report["strata"], "color", attempted_strata),
    ):
        keys = sorted(attempted_counts)
        for key in keys:
            selected = [row for row in admitted if (row[key_name] if key_name == "source_id" else f"color-{int(row['color'])}") == key]
            if not selected:
                destination[key] = {"accepted": False, "coverage": 0.0, "accepted_examples": 0, "attempted_examples": attempted_counts[key]}
                continue
            subset = _summarize_rows(
                ([row["wet"] for row in selected], [row["restored"] for row in selected], [row["clean"] for row in selected]),
                [row["trajectory"] for row in selected],
            )
            subset["accepted_examples"] = len(selected)
            subset["attempted_examples"] = attempted_counts[key]
            subset["coverage"] = len(selected) / attempted_counts[key]
            subset["all_individual_quality_passed"] = all(row["quality"]["passed"] for row in selected)
            subset["accepted"] = bool(subset["accepted"] and subset["all_individual_quality_passed"])
            destination[key] = subset
    report["safe_subdomain_gates"] = {
        "aggregate_quality": bool(report["accepted"]),
        "coverage": report["coverage"] >= MINIMUM_COVERAGE,
        "all_individual_quality": report["all_individual_quality_passed"],
        "all_sources_quality": all(
            item["accepted"] and item["accepted_examples"] >= MINIMUM_PER_SOURCE
            for item in report["sources"].values()
        ),
        "all_color_strata_quality": all(item["accepted"] for item in report["strata"].values()),
    }
    report["accepted"] = all(report["safe_subdomain_gates"].values())
    report["admission"] = {
        "margin_threshold": margin_threshold,
        "band_consensus_p90_threshold_radians": consensus_threshold,
        "output_safety_gate": "restored peak <= 1.0 and no increase in >=0.999 sample fraction",
    }
    report["admitted_rows"] = [{key: row[key] for key in (
        "index", "source_id", "group", "rate_hz", "mix", "color", "phase_error",
        "score", "margin", "band_consensus_p90_radians", "output_safe", "restored_peak"
    )} for row in admitted]
    return report


def _runtime(row: dict) -> dict:
    wet, values = row["wet"], row["values"]
    estimate_phase(wet, values)
    started = time.perf_counter()
    phase, _, _ = estimate_phase(wet, values)
    restored = invert_known_phase(wet, values, phase)
    seconds = time.perf_counter() - started
    return {
        "frames": len(restored),
        "mean_seconds": seconds,
        "realtime_factor": seconds / (len(restored) / 48_000.0),
        "ordinary_cpu": True,
        "bounded_phase_window": True,
        "python_reference_inverse": True,
        "audio_callback": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/product3-phaser-safe"))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    calibration = StonePhaserPairsV3(workspace, "calibration", 48, SEED + 2)
    challenge = StonePhaserPairsV3(workspace, "development", 120, SEED + 2003)
    calibration_rows = _precompute(calibration)
    candidates = {}
    passing = []
    for margin_threshold in MARGIN_THRESHOLDS:
        for consensus_threshold in CONSENSUS_THRESHOLDS:
            key = f"margin={margin_threshold};band_p90={consensus_threshold}"
            candidates[key] = _report(
                calibration_rows, margin_threshold, consensus_threshold, len(calibration)
            )
            if candidates[key]["accepted"]:
                passing.append((margin_threshold, consensus_threshold, candidates[key]["coverage"]))
    selected_margin, selected_consensus, _ = (
        max(passing, key=lambda item: (item[2], -item[0], item[1]))
        if passing else (MARGIN_THRESHOLDS[-1], CONSENSUS_THRESHOLDS[0], 0.0)
    )
    challenge_rows = _precompute(challenge)
    development = _report(
        challenge_rows, selected_margin, selected_consensus, len(challenge)
    )
    admitted_runtime_row = next((row for row in challenge_rows if row["margin"] >= selected_margin), challenge_rows[0])
    runtime = _runtime(admitted_runtime_row)
    accepted = bool(passing and development["accepted"] and runtime["realtime_factor"] <= 0.5)
    target = args.output.resolve() / "phaser"
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "model.pt"
    architecture = manifest(selected_margin)
    architecture["band_consensus_p90_threshold_radians"] = selected_consensus
    architecture["output_safety_gate"] = "restored peak <= 1.0 and no increase in >=0.999 sample fraction"
    torch.save({"schema": 1, "sample_rate": 48_000, "architecture": architecture}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 1,
        "status": "accepted-safe-subdomain" if accepted else "diagnostic-not-promoted",
        "accepted": accepted,
        "mechanism": "modulation",
        "family": "phaser",
        "model": {**architecture, "checkpoint": str(checkpoint), "sha256": digest},
        "calibration": {
            "selected_margin_threshold": selected_margin,
            "selected_band_consensus_p90_threshold_radians": selected_consensus,
            "minimum_coverage": MINIMUM_COVERAGE,
            "minimum_admitted_per_source": MINIMUM_PER_SOURCE,
            "candidates": candidates,
        },
        "development_control_challenge": {
            "seed": SEED + 2003,
            "same_held_out_source_groups_new_controls_and_crops": True,
            "fresh_after_margin-only-diagnostic": True,
            "report": development,
        },
        "runtime": runtime,
        "provenance": {
            "product_sources_only": True,
            "authorized_source_ids": calibration.authorization["sources"],
            "required_attribution": calibration.authorization["required_attribution"],
            "research_source_ids": [],
            "generated_audio_written": False,
            "physical_audio_devices_used": False,
            "locked_final_audio_opened": False,
            "demo_generated": False,
        },
    }
    (target / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "family": "phaser", "status": report["status"], "selected_margin_threshold": selected_margin,
        "selected_band_consensus_p90_threshold_radians": selected_consensus,
        "development_coverage": development.get("coverage", 0.0), "runtime_rtf": runtime["realtime_factor"],
        "checkpoint": str(checkpoint), "metrics": str(target / "metrics.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
