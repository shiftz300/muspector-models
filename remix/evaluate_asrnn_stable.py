#!/usr/bin/env python3
"""Evaluate the canonical stable RAT pilot under Muspector quality gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .asrnn_data import LICENSE, RECORD_URL, rat_files
from .forward_drive import multiresolution_spectral_loss, pre_emphasis
from .stable_rat import load_stable_rat
from .train import device
from .train_asrnn_rat_adapter import RatClips


@torch.inference_mode()
def evaluate(checkpoint: Path, corpus: Path, frames: int = 2_048) -> dict:
    target = device()
    model = load_stable_rat(checkpoint).to(target)
    loader = DataLoader(RatClips(rat_files(corpus, "eval")), batch_size=32)
    per_file, ratios, peak_errors, quiet_peaks = [], [], [], []
    total_error = total_energy = total_bypass_error = 0.0
    total_absolute_error = total_bypass_absolute_error = 0.0
    total_emphasized_error = total_emphasized_bypass_error = 0.0
    total_emphasized_energy = 0.0
    spectral_model, spectral_bypass = [], []
    for batch in loader:
        dry = batch["dry"].to(target)
        wet = batch["wet"].to(target)
        controls = batch["controls"].to(target)
        state = None
        chunks = []
        for start in range(0, dry.shape[1], frames):
            stop = min(start + frames, dry.shape[1])
            rendered, state = model(dry[:, start:stop], controls, state)
            chunks.append(rendered)
        rendered = torch.cat(chunks, dim=1)[:, 1_024:]
        dry = dry[:, 1_024:]
        wet = wet[:, 1_024:]
        error = (rendered - wet).square()
        bypass_error = (dry - wet).square()
        energy = wet.square()
        total_error += float(error.sum())
        total_bypass_error += float(bypass_error.sum())
        total_energy += float(energy.sum())
        total_absolute_error += float((rendered - wet).abs().sum())
        total_bypass_absolute_error += float((dry - wet).abs().sum())
        emphasized_wet = pre_emphasis(wet)
        total_emphasized_error += float((pre_emphasis(rendered) - emphasized_wet).square().sum())
        total_emphasized_bypass_error += float((pre_emphasis(dry) - emphasized_wet).square().sum())
        total_emphasized_energy += float(emphasized_wet.square().sum())
        spectral_model.append(float(multiresolution_spectral_loss(rendered, wet)))
        spectral_bypass.append(float(multiresolution_spectral_loss(dry, wet)))
        per_file.extend((error.mean(1) / energy.mean(1).clamp_min(1.0e-8)).cpu().tolist())
        target_peak = wet.abs().amax(1)
        rendered_peak = rendered.abs().amax(1)
        peak_errors.extend((rendered_peak - target_peak).abs().cpu().tolist())
        audible = target_peak >= 1.0e-3
        ratios.extend((rendered_peak[audible] / target_peak[audible]).cpu().tolist())
        quiet_peaks.extend(rendered_peak[~audible].cpu().tolist())

    silence = torch.zeros(3, 4_096, device=target)
    static_controls = torch.tensor(
        ((0.0, 1.0, 0.0), (0.5, 0.5, 0.5), (1.0, 0.0, 1.0)), device=target
    )
    static_output, _ = model(silence, static_controls)
    time = torch.linspace(0.0, 1.0, 4_096, device=target)
    dynamic_controls = torch.stack(
        (
            torch.sin(time * 7.0).mul(0.5).add(0.5),
            torch.cos(time * 11.0).mul(0.5).add(0.5),
            torch.sin(time * 5.0).mul(0.5).add(0.5),
        ),
        dim=1,
    ).unsqueeze(0)
    dynamic_output, _ = model(torch.zeros(1, 4_096, device=target), dynamic_controls)
    torch.manual_seed(20260830)
    probe = torch.randn(2, 6_173, device=target) * 0.04
    probe_controls = torch.tensor(((0.2, 0.7, 0.3), (0.8, 0.4, 0.9)), device=target)
    whole, _ = model(probe, probe_controls)
    state = None
    streamed = []
    for start, stop in ((0, 17), (17, 513), (513, 2_121), (2_121, 6_173)):
        value, state = model(probe[:, start:stop], probe_controls, state)
        streamed.append(value)
    stream_error = whole - torch.cat(streamed, dim=1)

    global_esr = total_error / max(total_energy, 1.0e-12)
    bypass_esr = total_bypass_error / max(total_energy, 1.0e-12)
    global_mae = total_absolute_error / (len(per_file) * (48_000 - 1_024))
    bypass_mae = total_bypass_absolute_error / (len(per_file) * (48_000 - 1_024))
    emphasized_esr = total_emphasized_error / max(total_emphasized_energy, 1.0e-12)
    bypass_emphasized_esr = total_emphasized_bypass_error / max(
        total_emphasized_energy, 1.0e-12
    )
    spectral = float(np.mean(spectral_model))
    bypass_spectral = float(np.mean(spectral_bypass))
    metrics = {
        "examples": len(per_file),
        "global_esr": global_esr,
        "bypass_global_esr": bypass_esr,
        "global_esr_relative_improvement": 1.0 - global_esr / max(bypass_esr, 1.0e-12),
        "global_mae": global_mae,
        "bypass_global_mae": bypass_mae,
        "global_mae_relative_improvement": 1.0 - global_mae / max(bypass_mae, 1.0e-12),
        "preemphasis_esr": emphasized_esr,
        "bypass_preemphasis_esr": bypass_emphasized_esr,
        "preemphasis_esr_relative_improvement": 1.0 - emphasized_esr / max(
            bypass_emphasized_esr, 1.0e-12
        ),
        "multiresolution_spectral_loss": spectral,
        "bypass_multiresolution_spectral_loss": bypass_spectral,
        "spectral_loss_relative_improvement": 1.0 - spectral / max(bypass_spectral, 1.0e-12),
        "mean_per_file_esr": float(np.mean(per_file)),
        "median_per_file_esr": float(np.median(per_file)),
        "p95_per_file_esr": float(np.quantile(per_file, 0.95)),
        "peak_ratio_median": float(np.median(ratios)),
        "peak_ratio_p95": float(np.quantile(ratios, 0.95)),
        "absolute_peak_error_p95": float(np.quantile(peak_errors, 0.95)),
        "quiet_prediction_peak_maximum": max(quiet_peaks, default=0.0),
        "static_silence_max_absolute_output": float(static_output.abs().max()),
        "dynamic_control_silence_max_absolute_output": float(dynamic_output.abs().max()),
        "stream_max_absolute_error": float(stream_error.abs().max()),
    }
    accepted = bool(
        metrics["global_esr"] <= 0.05
        and metrics["global_esr_relative_improvement"] >= 0.50
        and metrics["global_mae_relative_improvement"] >= 0.50
        and metrics["preemphasis_esr_relative_improvement"] >= 0.50
        and metrics["spectral_loss_relative_improvement"] >= 0.30
        and metrics["mean_per_file_esr"] <= 0.10
        and metrics["median_per_file_esr"] <= 0.05
        and metrics["p95_per_file_esr"] <= 0.25
        and 0.75 <= metrics["peak_ratio_median"] <= 1.25
        and metrics["peak_ratio_p95"] <= 1.35
        and metrics["absolute_peak_error_p95"] <= 0.02
        and metrics["quiet_prediction_peak_maximum"] <= 1.0e-3
        and metrics["static_silence_max_absolute_output"] == 0.0
        and metrics["dynamic_control_silence_max_absolute_output"] == 0.0
        and metrics["stream_max_absolute_error"] <= 2.0e-6
    )
    return {
        "schema": 1,
        "status": "accepted-internal-noncommercial-pilot" if accepted else "rejected",
        "accepted": accepted,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "dataset_record": RECORD_URL,
        "dataset_license": LICENSE,
        "official_eval": metrics,
        "quality_policy": {
            "source_files_read_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
        },
        "release_limitations": [
            "CC-BY-NC-4.0 data and weights prohibit commercial promotion",
            "evaluation output is write-once and must not be used for training selection",
            "no Rust or UI integration is authorized by this pilot",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.corpus)
    if args.output.exists():
        raise ValueError(f"stable RAT evaluation already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
