#!/usr/bin/env python3
"""Calibrate a Wet-only uncertainty selector for one Amp+cab+mic profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .amp_cab_model2 import AmpCabInverseExpert2
from .quality2 import measure, summarize
from .train_amp_cab import PROFILES, SEED, _pairs


MIN_COVERAGE = 0.25
MIN_PASS_FRACTION = 0.80


def _rows(model: torch.nn.Module, dataset) -> list[dict]:
    rows = []
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            restored, uncertainty, _ = model(
                row["wet"].unsqueeze(0), row["profile"].unsqueeze(0)
            )
            start = int(row["target_start"])
            wet = row["wet"][start:].numpy()
            clean = row["clean"][start:].numpy()
            candidate = restored[0, start:].numpy().astype(np.float32)
            uncertainty_values = uncertainty[0, start:].numpy()
            rows.append({
                "score": float(np.mean(uncertainty_values) + 0.5 * np.quantile(uncertainty_values, 0.90)),
                "quality": measure("amp", wet, candidate, clean),
                "wet": wet,
                "candidate": candidate,
                "clean": clean,
            })
    return rows


def _summary(rows: list[dict], threshold: float) -> dict:
    selected = [row for row in rows if row["score"] <= threshold]
    if not selected:
        return {
            "examples": len(rows), "admitted": 0, "coverage": 0.0,
            "pass_fraction": 0.0, "accepted": False,
        }
    report = summarize(
        "amp",
        [row["wet"] for row in selected],
        [row["candidate"] for row in selected],
        [row["clean"] for row in selected],
    )
    pass_fraction = float(np.mean([row["quality"]["passed"] for row in selected]))
    coverage = len(selected) / len(rows)
    accepted = bool(
        coverage >= MIN_COVERAGE
        and pass_fraction >= MIN_PASS_FRACTION
        and report["accepted"]
    )
    return {
        **report,
        "admitted": len(selected),
        "coverage": coverage,
        "individual_pass_fraction": pass_fraction,
        "accepted": accepted,
    }


def _select(rows: list[dict]) -> tuple[float | None, dict | None]:
    best = None
    for threshold in sorted({row["score"] for row in rows}):
        report = _summary(rows, threshold)
        if report["accepted"]:
            best = (threshold, report)
    return best if best is not None else (None, None)


def calibrate(args: argparse.Namespace) -> dict:
    metrics = json.loads(args.metrics.read_text())
    model_report = metrics["model"]
    profiles = metrics["data"]["profiles"]
    if len(profiles) != 1:
        raise ValueError("selector requires one independently trained profile")
    profile_index = next(
        index for index, profile in enumerate(PROFILES) if profile.id == profiles[0]
    )
    if model_report.get("p3_input_or_profile") is not False:
        raise PermissionError("P3 model boundary is missing")
    model = AmpCabInverseExpert2(model_report["hidden_size"], model_report["depth"])
    payload = torch.load(Path(model_report["checkpoint"]), map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    target_frames = metrics["training"]["target_frames"]
    calibration = _pairs(
        args.workspace, "calibration", metrics["training"]["calibration_samples"],
        target_frames, SEED + 2, profile_index,
    )
    development = _pairs(
        args.workspace, "development", metrics["training"]["development_samples"],
        target_frames, SEED + 3, profile_index,
    )
    calibration_rows = _rows(model, calibration)
    threshold, calibration_report = _select(calibration_rows)
    if threshold is None:
        development_report = None
        accepted = False
    else:
        development_report = _summary(_rows(model, development), threshold)
        accepted = bool(development_report["accepted"])
    report = {
        "schema": 1,
        "status": "accepted-development-selective" if accepted else "rejected-selector",
        "accepted": accepted,
        "mechanism": "amp",
        "profile": profiles[0],
        "checkpoint": model_report["checkpoint"],
        "checkpoint_sha256": model_report["sha256"],
        "selector": {
            "input": "Wet-only model uncertainty",
            "score": "mean + 0.5 * p90 pointwise uncertainty",
            "threshold": threshold,
            "selected_on": "calibration only",
            "minimum_coverage": MIN_COVERAGE,
            "minimum_individual_pass_fraction": MIN_PASS_FRACTION,
            "graph_order_input": False,
            "neighbor_effect_input": False,
            "clean_input": False,
            "p3_input": False,
        },
        "calibration": calibration_report,
        "development": development_report,
        "quality": {
            "demo_generated": False,
            "source_audio_modified": False,
            "physical_audio_devices_used": False,
            "development_used_for_selection": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "threshold": threshold,
        "calibration": calibration_report,
        "development": development_report,
    }, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibrate(args)


if __name__ == "__main__":
    main()
