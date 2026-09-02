#!/usr/bin/env python3
"""Calibrate and gate the safe-subdomain Tremolo phase inverse."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .modulation3 import TremoloPairsV3
from .modulation_phase3 import TremoloPhaseInverseV3
from .train_modulation3 import SEED, _summarize_rows, _trajectory_row


THRESHOLDS = (0.10, 0.12, 0.14, 0.16, 0.18, 0.20)
MINIMUM_COVERAGE = 0.40


def _quality(model: TremoloPhaseInverseV3, dataset: TremoloPairsV3) -> dict:
    aggregate = ([], [], [])
    trajectories = []
    sources = defaultdict(lambda: (([], [], []), []))
    strata = defaultdict(lambda: (([], [], []), []))
    attempted_sources = Counter()
    accepted_sources = Counter()
    abstentions = []
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            source = row["source_id"]
            attempted_sources[source] += 1
            restored, residual, accepted, gain = model(
                row["wet"].unsqueeze(0), row["controls"].unsqueeze(0)
            )
            if not bool(accepted[0]):
                abstentions.append({
                    "index": index,
                    "source_id": source,
                    "group": row["group"],
                    "rate_hz": row["control_values"]["rate_hz"],
                    "waveform": row["control_values"]["waveform"],
                    "harmonic_residual_ratio": float(residual[0]),
                })
                continue
            accepted_sources[source] += 1
            start, end = int(row["target_start"]), int(row["target_end"])
            audio = (
                row["wet"][start:end].numpy(),
                restored[0, start:end].numpy().astype(np.float32),
                row["clean"][start:end].numpy(),
            )
            trajectory = _trajectory_row(
                gain[0, start:end].numpy(), row["inverse_log_gain"][start:end].numpy()
            )
            key = f"{row['control_values']['waveform']}-fast-rate"
            for audio_rows, trajectory_rows in (
                (aggregate, trajectories), sources[source], strata[key]
            ):
                for target, value in zip(audio_rows, audio, strict=True):
                    target.append(value)
                trajectory_rows.append(trajectory)
    if not trajectories:
        return {"accepted": False, "coverage": 0.0, "reason": "no admitted examples"}
    report = _summarize_rows(aggregate, trajectories)
    report["attempted_examples"] = len(dataset)
    report["accepted_examples"] = len(trajectories)
    report["coverage"] = len(trajectories) / len(dataset)
    report["abstentions"] = abstentions
    report["trajectory"]["minimum_correlation"] = min(row["correlation"] for row in trajectories)
    report["trajectory"]["maximum_normalized_mae"] = max(row["normalized_mae"] for row in trajectories)
    report["sources"] = {}
    for name in sorted(attempted_sources):
        rows = sources[name]
        source_report = _summarize_rows(*rows) if rows[1] else {"accepted": False}
        source_report["attempted_examples"] = attempted_sources[name]
        source_report["accepted_examples"] = accepted_sources[name]
        source_report["coverage"] = accepted_sources[name] / attempted_sources[name]
        report["sources"][name] = source_report
    report["strata"] = {name: _summarize_rows(*rows) for name, rows in sorted(strata.items())}
    report["safe_subdomain_gates"] = {
        "aggregate_quality": bool(report["accepted"]),
        "coverage": report["coverage"] >= MINIMUM_COVERAGE,
        "every_admitted_trajectory_correlation": report["trajectory"]["minimum_correlation"] >= 0.75,
        "every_admitted_trajectory_error": report["trajectory"]["maximum_normalized_mae"] <= 0.35,
        "all_sources_quality": bool(report["sources"] and all(row["accepted"] for row in report["sources"].values())),
        "all_strata_quality": bool(report["strata"] and all(row["accepted"] for row in report["strata"].values())),
    }
    report["accepted"] = all(report["safe_subdomain_gates"].values())
    return report


def _runtime(model: TremoloPhaseInverseV3, dataset: TremoloPairsV3) -> dict:
    row = dataset[0]
    wet, controls = row["wet"].unsqueeze(0), row["controls"].unsqueeze(0)
    with torch.inference_mode():
        model(wet, controls)
        started = time.perf_counter()
        repeats = 3
        for _ in range(repeats):
            model(wet, controls)
        seconds = (time.perf_counter() - started) / repeats
    return {
        "frames": len(row["wet"]),
        "mean_seconds": seconds,
        "realtime_factor": seconds / (len(row["wet"]) / 48_000.0),
        "ordinary_cpu": True,
        "bounded_context": True,
        "audio_callback": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/product3-modulation-safe"))
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    calibration = TremoloPairsV3(workspace, "calibration", 48, 96_000, SEED + 2)
    development = TremoloPairsV3(workspace, "development", 72, 96_000, SEED + 3)
    calibration_candidates = {}
    selected = None
    for threshold in THRESHOLDS:
        report = _quality(TremoloPhaseInverseV3(threshold), calibration)
        calibration_candidates[str(threshold)] = report
        if report["accepted"]:
            selected = threshold
    if selected is None:
        selected = THRESHOLDS[0]
    model = TremoloPhaseInverseV3(selected)
    development_report = _quality(model, development)
    runtime = _runtime(model, development)
    accepted = bool(
        any(row["accepted"] for row in calibration_candidates.values())
        and development_report["accepted"]
        and runtime["realtime_factor"] <= 0.5
    )
    target = output / "tremolo"
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "model.pt"
    torch.save({"schema": 1, "sample_rate": 48_000, "architecture": model.manifest()}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    learned_metrics = workspace / "runs/foundation/product3-modulation/tremolo/metrics.json"
    learned = json.loads(learned_metrics.read_text())
    report = {
        "schema": 1,
        "status": "accepted-safe-subdomain" if accepted else "diagnostic-not-promoted",
        "accepted": accepted,
        "mechanism": "modulation",
        "family": "tremolo",
        "model": {**model.manifest(), "checkpoint": str(checkpoint), "sha256": digest},
        "calibration": {
            "selection_split": "calibration",
            "selected_residual_threshold": selected,
            "minimum_coverage": MINIMUM_COVERAGE,
            "candidates": calibration_candidates,
        },
        "development": development_report,
        "runtime": runtime,
        "rejected_learned_candidate": {
            "metrics": str(learned_metrics),
            "checkpoint": learned["model"]["checkpoint"],
            "sha256": learned["model"]["sha256"],
            "selected_epoch": learned["training"]["selected_epoch"],
            "reason": "DAFx continuous-phrase and slow-rate trajectory coverage failed",
        },
        "provenance": {
            "product_sources_only": True,
            "generic_repository_owned_dsp": True,
            "named_physical_device_claim": False,
            "authorized_source_ids": calibration.authorization["sources"],
            "required_attribution": calibration.authorization["required_attribution"],
            "research_source_ids": [],
            "generated_audio_written": False,
            "physical_audio_devices_used": False,
            "locked_final_audio_opened": False,
        },
    }
    (target / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "mechanism": "modulation",
        "family": "tremolo",
        "status": report["status"],
        "selected_residual_threshold": selected,
        "development_coverage": development_report["coverage"],
        "checkpoint": str(checkpoint),
        "metrics": str(target / "metrics.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
