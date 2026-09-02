#!/usr/bin/env python3
"""Refit only the bias-free stable-effect output layer with peak-aware LS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .asrnn_effects import effect_files
from .evaluate_asrnn_effect import EffectClips
from .stable_effect import load_stable_effect


PEAK_WEIGHTS = (0.0, 64.0, 256.0, 1024.0)


def _partition(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    fit, calibrate = [], []
    for path in paths:
        take_id = int(path.stem.split(",")[-1])
        (calibrate if take_id % 5 == 0 else fit).append(path)
    if not fit or not calibrate:
        raise ValueError("output refit requires non-empty fit and calibrate partitions")
    return fit, calibrate


@torch.inference_mode()
def _hidden(model, dry: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
    state = None
    chunks = []
    for start in range(0, dry.shape[1], 2_048):
        value, state = model.encode(dry[:, start : start + 2_048], controls, state)
        chunks.append(value)
    return torch.cat(chunks, dim=1)[:, 1_024:]


@torch.inference_mode()
def _normal_equations(model, paths: list[Path], device: str) -> tuple:
    width = model.hidden_size
    waveform_xx = torch.zeros(width, width, dtype=torch.float64)
    waveform_xy = torch.zeros(width, dtype=torch.float64)
    peak_xx = torch.zeros_like(waveform_xx)
    peak_xy = torch.zeros_like(waveform_xy)
    loader = DataLoader(EffectClips(paths, device), batch_size=4)
    for index, batch in enumerate(loader):
        hidden = _hidden(model, batch["dry"], batch["controls"]).double()
        wet = batch["wet"][:, 1_024:].double()
        flat_hidden = hidden.reshape(-1, width)
        flat_wet = wet.reshape(-1)
        waveform_xx += flat_hidden.T @ flat_hidden
        waveform_xy += flat_hidden.T @ flat_wet
        emphasized_hidden = hidden[:, 1:] - 0.95 * hidden[:, :-1]
        emphasized_wet = wet[:, 1:] - 0.95 * wet[:, :-1]
        flat_emphasized = emphasized_hidden.reshape(-1, width)
        waveform_xx += 0.25 * (flat_emphasized.T @ flat_emphasized)
        waveform_xy += 0.25 * (flat_emphasized.T @ emphasized_wet.reshape(-1))
        peak_indices = wet.abs().topk(k=64, dim=1).indices
        peak_hidden = hidden.gather(
            1, peak_indices.unsqueeze(-1).expand(-1, -1, width)
        ).reshape(-1, width)
        peak_wet = wet.gather(1, peak_indices).reshape(-1)
        peak_xx += peak_hidden.T @ peak_hidden
        peak_xy += peak_hidden.T @ peak_wet
        if index % 8 == 0 or index + 1 == len(loader):
            print(json.dumps({"stage": "fit-readout", "batch": index + 1, "batches": len(loader)}), flush=True)
    return waveform_xx, waveform_xy, peak_xx, peak_xy


def _solve(base_xx, base_xy, peak_xx, peak_xy, peak_weight: float) -> torch.Tensor:
    matrix = base_xx + peak_weight * peak_xx
    target = base_xy + peak_weight * peak_xy
    ridge = max(float(torch.trace(matrix)) / len(matrix) * 1.0e-9, 1.0e-12)
    return torch.linalg.solve(matrix + torch.eye(len(matrix)) * ridge, target).float()


@torch.inference_mode()
def _calibrate(model, paths: list[Path], device: str, candidates: dict) -> dict:
    accumulators = {
        name: {"error": 0.0, "energy": 0.0, "per_file": [], "peak_errors": []}
        for name in candidates
    }
    loader = DataLoader(EffectClips(paths, device), batch_size=8)
    for batch in loader:
        hidden = _hidden(model, batch["dry"], batch["controls"])
        wet = batch["wet"][:, 1_024:]
        energy = wet.square().mean(1).clamp_min(1.0e-8)
        target_peak = wet.abs().amax(1)
        for name, weight in candidates.items():
            predicted = hidden @ weight
            error = (predicted - wet).square()
            accumulator = accumulators[name]
            accumulator["error"] += float(error.sum())
            accumulator["energy"] += float(wet.square().sum())
            accumulator["per_file"].extend((error.mean(1) / energy).tolist())
            accumulator["peak_errors"].extend(
                (predicted.abs().amax(1) - target_peak).abs().tolist()
            )
    reports = {}
    for name, values in accumulators.items():
        reports[name] = {
            "global_esr": values["error"] / values["energy"],
            "mean_per_file_esr": float(np.mean(values["per_file"])),
            "p95_per_file_esr": float(np.quantile(values["per_file"], 0.95)),
            "absolute_peak_error_p95": float(
                np.quantile(values["peak_errors"], 0.95)
            ),
        }
        reports[name]["passes_selection_gate"] = bool(
            reports[name]["global_esr"] <= 0.05
            and reports[name]["mean_per_file_esr"] <= 0.10
            and reports[name]["p95_per_file_esr"] <= 0.25
            and reports[name]["absolute_peak_error_p95"] <= 0.02
        )
    return reports


def fit(checkpoint: Path, corpus: Path, device: str, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"output-refit checkpoint already exists: {output}")
    model, payload = load_stable_effect(checkpoint)
    if payload["device"] != device:
        raise ValueError("output-refit device differs from checkpoint")
    fit_paths, calibrate_paths = _partition(effect_files(corpus, device, "train"))
    base_xx, base_xy, peak_xx, peak_xy = _normal_equations(model, fit_paths, device)
    candidates = {
        "original": model.output_layer.weight.detach().flatten().clone(),
        **{
            f"peak-weight-{int(peak_weight)}": _solve(
                base_xx, base_xy, peak_xx, peak_xy, peak_weight
            )
            for peak_weight in PEAK_WEIGHTS
        },
    }
    calibration = _calibrate(model, calibrate_paths, device, candidates)
    selected_name = min(
        candidates,
        key=lambda name: (
            not calibration[name]["passes_selection_gate"],
            calibration[name]["absolute_peak_error_p95"],
            calibration[name]["global_esr"],
        ),
    )
    admitted = calibration[selected_name]["passes_selection_gate"]
    state = payload["state_dict"].copy()
    state["output_layer.weight"] = candidates[selected_name].reshape(1, -1)
    output_payload = dict(payload)
    output_payload["state_dict"] = state
    output_payload["output_refit"] = {
        "method": "bias-free peak-aware linear least squares",
        "fit_examples": len(fit_paths),
        "calibrate_examples": len(calibrate_paths),
        "selected_candidate": selected_name,
        "official_eval_opened_for_selection": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if admitted:
        torch.save(output_payload, output)
    report = {
        "schema": 1,
        "device": device,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "output_checkpoint": str(output) if admitted else None,
        "output_checkpoint_sha256": (
            hashlib.sha256(output.read_bytes()).hexdigest() if admitted else None
        ),
        "fit_examples": len(fit_paths),
        "calibrate_examples": len(calibrate_paths),
        "candidates": calibration,
        "selected_candidate": selected_name,
        "admitted_for_official_eval": admitted,
        "selected_on_official_eval": False,
        "recurrent_core_frozen": True,
        "output_bias_present": False,
        "physical_audio_devices_used": False,
    }
    (output.parent / "output-refit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("dfz", "cs3"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = fit(args.checkpoint, args.corpus.resolve(), args.device, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["admitted_for_official_eval"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
