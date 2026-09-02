#!/usr/bin/env python3
"""Refit a stable RAT readout without changing its recurrent core."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .asrnn_data import audit_asrnn, rat_files
from .evaluate_asrnn_stable import evaluate
from .stable_rat import load_stable_rat
from .train_asrnn_rat_adapter import RatClips, _partition


PEAK_WEIGHTS = (0.0, 128.0, 512.0, 2048.0)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def _hidden(model, dry: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
    state = None
    chunks = []
    for start in range(0, dry.shape[1], 2_048):
        value, state = model.encode(dry[:, start : start + 2_048], controls, state)
        chunks.append(value)
    return torch.cat(chunks, dim=1)[:, 1_024:]


@torch.inference_mode()
def _normal_equations(model, paths: list[Path]) -> tuple[torch.Tensor, ...]:
    width = model.hidden_size
    waveform_xx = torch.zeros(width, width, dtype=torch.float64)
    waveform_xy = torch.zeros(width, dtype=torch.float64)
    peak_xx = torch.zeros_like(waveform_xx)
    peak_xy = torch.zeros_like(waveform_xy)
    loader = DataLoader(RatClips(paths), batch_size=16)
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
        waveform_xy += 0.25 * (
            flat_emphasized.T @ emphasized_wet.reshape(-1)
        )
        peak_indices = wet.abs().topk(k=64, dim=1).indices
        peak_hidden = hidden.gather(
            1, peak_indices.unsqueeze(-1).expand(-1, -1, width)
        ).reshape(-1, width)
        peak_wet = wet.gather(1, peak_indices).reshape(-1)
        peak_xx += peak_hidden.T @ peak_hidden
        peak_xy += peak_hidden.T @ peak_wet
        if index % 8 == 0 or index + 1 == len(loader):
            print(
                json.dumps(
                    {"stage": "fit", "batch": index + 1, "batches": len(loader)}
                ),
                flush=True,
            )
    return waveform_xx, waveform_xy, peak_xx, peak_xy


def _solve(
    waveform_xx: torch.Tensor,
    waveform_xy: torch.Tensor,
    peak_xx: torch.Tensor,
    peak_xy: torch.Tensor,
    peak_weight: float,
) -> torch.Tensor:
    matrix = waveform_xx + peak_weight * peak_xx
    target = waveform_xy + peak_weight * peak_xy
    ridge = max(float(torch.trace(matrix)) / len(matrix) * 1.0e-9, 1.0e-12)
    return torch.linalg.solve(
        matrix + torch.eye(len(matrix), dtype=matrix.dtype) * ridge,
        target,
    ).float()


@torch.inference_mode()
def _calibrate(model, paths: list[Path], candidates: dict[str, torch.Tensor]) -> dict:
    values = {
        name: {
            "error": 0.0,
            "energy": 0.0,
            "files": [],
            "peak_errors": [],
            "peak_ratios": [],
            "quiet_peaks": [],
        }
        for name in candidates
    }
    loader = DataLoader(RatClips(paths), batch_size=16)
    for batch in loader:
        hidden = _hidden(model, batch["dry"], batch["controls"])
        wet = batch["wet"][:, 1_024:]
        energy = wet.square().mean(1).clamp_min(1.0e-8)
        target_peak = wet.abs().amax(1)
        audible = target_peak >= 1.0e-3
        for name, weight in candidates.items():
            prediction = hidden @ weight
            error = (prediction - wet).square()
            predicted_peak = prediction.abs().amax(1)
            row = values[name]
            row["error"] += float(error.sum())
            row["energy"] += float(wet.square().sum())
            row["files"].extend((error.mean(1) / energy).tolist())
            row["peak_errors"].extend((predicted_peak - target_peak).abs().tolist())
            row["peak_ratios"].extend(
                (predicted_peak[audible] / target_peak[audible]).tolist()
            )
            row["quiet_peaks"].extend(predicted_peak[~audible].tolist())
    reports = {}
    for name, row in values.items():
        report = {
            "global_esr": row["error"] / max(row["energy"], 1.0e-12),
            "mean_per_file_esr": float(np.mean(row["files"])),
            "median_per_file_esr": float(np.median(row["files"])),
            "p95_per_file_esr": float(np.quantile(row["files"], 0.95)),
            "peak_ratio_median": float(np.median(row["peak_ratios"])),
            "peak_ratio_p95": float(np.quantile(row["peak_ratios"], 0.95)),
            "absolute_peak_error_p95": float(
                np.quantile(row["peak_errors"], 0.95)
            ),
            "quiet_prediction_peak_maximum": max(row["quiet_peaks"], default=0.0),
        }
        report["passes_selection_gate"] = bool(
            report["global_esr"] <= 0.05
            and report["mean_per_file_esr"] <= 0.10
            and report["median_per_file_esr"] <= 0.05
            and report["p95_per_file_esr"] <= 0.25
            and 0.75 <= report["peak_ratio_median"] <= 1.25
            and report["peak_ratio_p95"] <= 1.35
            and report["absolute_peak_error_p95"] <= 0.02
            and report["quiet_prediction_peak_maximum"] <= 1.0e-3
        )
        reports[name] = report
    return reports


def refit(checkpoint: Path, corpus: Path, output: Path) -> dict:
    if output.exists() or (output.parent / "metrics.json").exists():
        raise ValueError(f"round-three output already exists: {output.parent}")
    audit = audit_asrnn(corpus, scan_audio=True)
    model = load_stable_rat(checkpoint).cpu().eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    fit_paths, calibrate_paths = _partition(rat_files(corpus, "train"))
    equations = _normal_equations(model, fit_paths)
    candidates = {
        "original": model.output_layer.weight.detach().flatten().clone(),
        **{
            f"peak-{int(weight)}": _solve(*equations, weight)
            for weight in PEAK_WEIGHTS
        },
    }
    calibration = _calibrate(model, calibrate_paths, candidates)
    passing = [
        name for name, report in calibration.items() if report["passes_selection_gate"]
    ]
    if not passing:
        selected = "original"
    else:
        selected = min(
            passing,
            key=lambda name: (
                calibration[name]["global_esr"] / 0.05
                + calibration[name]["absolute_peak_error_p95"] / 0.02
                + calibration[name]["mean_per_file_esr"] / 0.10,
                calibration[name]["absolute_peak_error_p95"],
            ),
        )
    state = {name: value.detach().cpu().clone() for name, value in payload["state_dict"].items()}
    state["output_layer.weight"] = candidates[selected].reshape(1, -1)
    result_payload = dict(payload)
    result_payload["state_dict"] = state
    result_payload["readout_refit"] = {
        "method": "frozen-core peak-aware linear least squares",
        "fit_examples": len(fit_paths),
        "calibrate_examples": len(calibrate_paths),
        "selected": selected,
        "official_eval_opened_for_selection": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result_payload, output)
    final = evaluate(output, corpus)
    (output.parent / "metrics.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n"
    )
    status = (
        "candidate"
        if final["accepted"] and selected != "original"
        else "retained"
        if final["accepted"]
        else "rejected"
    )
    report = {
        "schema": 1,
        "status": status,
        "source": str(checkpoint),
        "source_sha256": _digest(checkpoint),
        "output": str(output),
        "output_sha256": _digest(output),
        "recurrent_core_frozen": True,
        "output_bias_present": False,
        "fit_examples": len(fit_paths),
        "calibrate_examples": len(calibrate_paths),
        "official_eval_examples": final["official_eval"]["examples"],
        "selected": selected,
        "candidates": calibration,
        "final": final,
        "dataset_audit": audit,
        "quality_policy": {
            "source_files_read_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
        },
    }
    (output.parent / "refit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = refit(args.checkpoint, args.corpus.resolve(), args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected": report["selected"],
                "official_eval": report["final"]["official_eval"],
            },
            sort_keys=True,
        )
    )
    if report["status"] == "rejected":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
