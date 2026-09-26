#!/usr/bin/env python3
"""Frozen inverse-forward replay audit for generic Drive control identifiability.

This is not a control estimator and does not train or promote a model. It asks
whether a bounded top-k control set can be recovered from Wet alone by applying
the accepted inverse at each candidate and replaying the repository-owned
forward effect. Truth is used only to score the frozen synthetic audit.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfilt

from .foundation_data import RATE
from .foundation_expert_runtime import FoundationExpertRuntime


SEED = 20260930
FRAMES = 8_192
CROP = 512
TOP_K = 12
CHUNK = 36
AXES = {
    "drive": (2.0, 4.0, 6.0, 8.0),
    "bias": (-0.10, 0.0, 0.10),
    "cutoff_hz": (2_200.0, 5_500.0, 10_000.0),
    "level": (0.50, 0.65, 0.80),
    "shape": (0.0, 1.0, 2.0),
}
CONTINUOUS = ("drive", "bias", "cutoff_hz", "level")
THRESHOLDS = {
    "top_k_exact_tuple_coverage_minimum": 0.80,
    "each_control_interval_coverage_minimum": 0.90,
    "each_continuous_median_normalized_width_maximum": 0.50,
    "shape_top1_accuracy_minimum": 0.80,
    "median_top_k_boundary_margin_minimum": 0.02,
}


def candidate_grid() -> list[dict[str, float]]:
    names = tuple(AXES)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(AXES[name] for name in names))
    ]


def normalized_controls(rows: list[dict[str, float]]) -> np.ndarray:
    values = []
    for row in rows:
        values.append((
            (row["drive"] - 1.8) / (8.0 - 1.8),
            (row["bias"] + 0.12) / 0.24,
            math.log(row["cutoff_hz"] / 1_800.0) / math.log(11_000.0 / 1_800.0),
            (row["level"] - 0.45) / 0.40,
            row["shape"] / 2.0,
        ))
    return np.asarray(values, dtype=np.float32)


def clean_example(index: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + index * 977)
    time = np.arange(FRAMES, dtype=np.float64) / RATE
    base = 82.0 + 11.0 * index
    signal = (
        0.10 * np.sin(2.0 * np.pi * base * time + 0.13 * index)
        + 0.055 * np.sin(2.0 * np.pi * (base * 5.7) * time + 0.4)
        + 0.025 * np.sin(2.0 * np.pi * (2_200.0 + 137.0 * index) * time + 0.9)
        + 0.007 * rng.standard_normal(FRAMES)
    )
    envelope = 0.30 + 0.70 * np.square(np.sin(2.0 * np.pi * (2.1 + 0.07 * index) * time))
    signal *= envelope
    signal[:: max(127, 421 - index * 11)] += 0.08
    return np.clip(signal, -0.45, 0.45).astype(np.float32)


def truth_controls(index: int) -> dict[str, float]:
    return {
        "drive": AXES["drive"][(index // 3) % len(AXES["drive"])],
        "bias": AXES["bias"][(index * 2) % len(AXES["bias"])],
        "cutoff_hz": AXES["cutoff_hz"][(index * 2 + index // 3) % len(AXES["cutoff_hz"])],
        "level": AXES["level"][(index + index // 2) % len(AXES["level"])],
        "shape": AXES["shape"][index % len(AXES["shape"])],
    }


def render_batch(clean: np.ndarray, controls: list[dict[str, float]]) -> np.ndarray:
    audio = np.asarray(clean, dtype=np.float64)
    if audio.ndim == 1:
        audio = np.broadcast_to(audio, (len(controls), audio.shape[0]))
    elif audio.ndim != 2 or audio.shape[0] != len(controls):
        raise ValueError("Drive renderer expects one Clean or one Clean per control row")
    frames = audio.shape[1]
    oversampled = resample_poly(audio, 2, 1, axis=1)
    drive = np.asarray([row["drive"] for row in controls])[:, None]
    bias = np.asarray([row["bias"] for row in controls])[:, None]
    shaped = oversampled * drive + bias
    shape = np.asarray([row["shape"] for row in controls])
    output = np.empty_like(shaped)
    for value in AXES["shape"]:
        mask = shape == value
        if value == 0.0:
            output[mask] = np.tanh(shaped[mask]) - np.tanh(bias[mask])
        elif value == 1.0:
            output[mask] = (
                (2.0 / np.pi) * np.arctan(shaped[mask] * 1.8)
                - (2.0 / np.pi) * np.arctan(bias[mask] * 1.8)
            )
        else:
            limited = np.clip(shaped[mask], -1.5, 1.5)
            output[mask] = limited - limited**3 / 6.75
    output = resample_poly(output, 1, 2, axis=1)[:, :frames]
    for index, row in enumerate(controls):
        output[index] = sosfilt(
            butter(2, row["cutoff_hz"], btype="lowpass", fs=RATE, output="sos"),
            output[index],
        )
    level = np.asarray([row["level"] for row in controls])[:, None]
    return np.asarray(output * level, dtype=np.float32)


def summarize_rows(rows: list[dict]) -> dict:
    exact = np.asarray([row["truth_in_top_k"] for row in rows], dtype=np.float64)
    shape = np.asarray([row["shape_top1"] for row in rows], dtype=np.float64)
    margins = np.asarray([row["top_k_boundary_margin"] for row in rows], dtype=np.float64)
    coverage = {
        name: float(np.mean([row["intervals"][name]["covers_truth"] for row in rows]))
        for name in AXES
    }
    widths = {
        name: float(np.median([row["intervals"][name]["normalized_width"] for row in rows]))
        for name in CONTINUOUS
    }
    metrics = {
        "examples": len(rows),
        "top_k_exact_tuple_coverage": float(exact.mean()),
        "shape_top1_accuracy": float(shape.mean()),
        "median_top_k_boundary_margin": float(np.median(margins)),
        "control_interval_coverage": coverage,
        "continuous_median_normalized_width": widths,
    }
    gates = {
        "top_k_exact_tuple_coverage": metrics["top_k_exact_tuple_coverage"]
        >= THRESHOLDS["top_k_exact_tuple_coverage_minimum"],
        "each_control_interval_coverage": min(coverage.values())
        >= THRESHOLDS["each_control_interval_coverage_minimum"],
        "each_continuous_interval_narrow": max(widths.values())
        <= THRESHOLDS["each_continuous_median_normalized_width_maximum"],
        "shape_top1_accuracy": metrics["shape_top1_accuracy"]
        >= THRESHOLDS["shape_top1_accuracy_minimum"],
        "top_k_boundary_separated": metrics["median_top_k_boundary_margin"]
        >= THRESHOLDS["median_top_k_boundary_margin_minimum"],
    }
    return {"accepted": all(gates.values()), "metrics": metrics, "gates": gates}


def audit(checkpoint: Path, device: str = "mps") -> dict:
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the frozen Drive identifiability audit")
    runtime = FoundationExpertRuntime("nonlinear", checkpoint)
    assert runtime.model is not None
    runtime.model = runtime.model.to(torch.device(device))
    candidates = candidate_grid()
    normalized = normalized_controls(candidates)
    rows = []
    for example_index in range(12):
        clean = clean_example(example_index)
        truth = truth_controls(example_index)
        wet = render_batch(clean, [truth])[0]
        restored_parts = []
        uncertainty_parts = []
        for offset in range(0, len(candidates), CHUNK):
            count = min(CHUNK, len(candidates) - offset)
            wet_tensor = torch.from_numpy(wet).to(device).unsqueeze(0).expand(count, -1)
            control_tensor = torch.from_numpy(normalized[offset : offset + count]).to(device)
            with torch.inference_mode():
                restored, uncertainty, _ = runtime.model(wet_tensor, control_tensor)
            restored_parts.append(restored.cpu().numpy())
            uncertainty_parts.append(uncertainty.median(dim=1).values.cpu().numpy())
        restored = np.concatenate(restored_parts)
        uncertainty = np.concatenate(uncertainty_parts)
        replay = render_batch(restored, candidates)
        selection = slice(CROP, -CROP)
        denominator = float(np.mean(np.square(wet[selection]))) + 1.0e-12
        scores = np.mean(np.square(replay[:, selection] - wet[None, selection]), axis=1) / denominator
        ranking = np.argsort(scores, kind="stable")
        chosen = ranking[:TOP_K]
        top_controls = [candidates[int(index)] for index in chosen]
        truth_index = candidates.index(truth)
        intervals = {}
        for name, axis in AXES.items():
            selected = [row[name] for row in top_controls]
            if name == "shape":
                intervals[name] = {
                    "values": sorted(set(selected)),
                    "covers_truth": truth[name] in selected,
                }
            else:
                low, high = min(selected), max(selected)
                intervals[name] = {
                    "minimum": low,
                    "maximum": high,
                    "covers_truth": low <= truth[name] <= high,
                    "normalized_width": (high - low) / (max(axis) - min(axis)),
                }
        boundary = float(scores[ranking[TOP_K]])
        best = float(scores[ranking[0]])
        rows.append({
            "example": example_index,
            "truth": truth,
            "top1": candidates[int(ranking[0])],
            "truth_rank": int(np.flatnonzero(ranking == truth_index)[0]) + 1,
            "truth_in_top_k": truth_index in chosen,
            "shape_top1": candidates[int(ranking[0])]["shape"] == truth["shape"],
            "top1_replay_nmse": best,
            "top1_median_uncertainty": float(uncertainty[ranking[0]]),
            "top_k_boundary_margin": (boundary - best) / max(best, 1.0e-12),
            "intervals": intervals,
        })
    summary = summarize_rows(rows)
    return {
        "schema": 1,
        "status": "development-identifiability-pass-not-promoted"
        if summary["accepted"] else "rejected-unidentifiable",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "compute": device,
        "candidate_count": len(candidates),
        "top_k": TOP_K,
        "thresholds": THRESHOLDS,
        "summary": summary,
        "examples": rows,
        "scope": {
            "wet_only_at_inference": True,
            "truth_used_for_scoring_only": True,
            "synthetic_repository_owned_drive": True,
            "point_estimate_promoted": False,
            "physical_drive": False,
            "amp_included": False,
            "reverb_included": False,
            "locked_final_opened": False,
            "source_audio_modified": False,
            "training_performed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("runs/foundation/product3-safe/nonlinear/model.pt"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product7-drive-control-identifiability/audit.json"),
    )
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    report = audit(args.checkpoint.resolve(), args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
