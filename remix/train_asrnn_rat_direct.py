#!/usr/bin/env python3
"""Train a direct non-commercial ProCo RAT pilot on the public ASRNN corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .asrnn_data import LICENSE, RECORD_URL, audit_asrnn, rat_files
from .rat_direct import RatDirectRenderer
from .train import device
from .train_asrnn_rat_adapter import (
    RatClips,
    _bounded,
    _detach_state,
    _partition,
    _rat_loss,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/asrnn-rat-direct-pilot"


@torch.inference_mode()
def _evaluate(
    model: RatDirectRenderer,
    loader: DataLoader,
    target: torch.device,
    frames: int,
    warmup: int,
) -> dict:
    model.eval()
    baseline_errors, model_errors = [], []
    peak_ratios, peak_errors, quiet_peaks = [], [], []
    for batch in loader:
        dry = batch["dry"].to(target)
        wet = batch["wet"].to(target)
        controls = batch["controls"].to(target)
        state = None
        chunks = []
        for start in range(0, dry.shape[1], frames):
            stop = min(start + frames, dry.shape[1])
            prediction, state = model(dry[:, start:stop], controls, state)
            chunks.append(prediction)
        prediction = torch.cat(chunks, dim=1)[:, warmup:]
        dry = dry[:, warmup:]
        wet = wet[:, warmup:]
        baseline = dry * controls[:, 2:3].square()
        energy = wet.square().mean(dim=1).clamp_min(1.0e-8)
        baseline_errors.extend(((baseline - wet).square().mean(dim=1) / energy).cpu().tolist())
        model_errors.extend(((prediction - wet).square().mean(dim=1) / energy).cpu().tolist())
        target_peak = wet.abs().amax(dim=1)
        prediction_peak = prediction.abs().amax(dim=1)
        peak_errors.extend((prediction_peak - target_peak).abs().cpu().tolist())
        audible = target_peak >= 1.0e-3
        peak_ratios.extend((prediction_peak[audible] / target_peak[audible]).cpu().tolist())
        quiet_peaks.extend(prediction_peak[~audible].cpu().tolist())
    baseline_esr = float(np.mean(baseline_errors))
    model_esr = float(np.mean(model_errors))
    ratios = np.asarray(peak_ratios, dtype=np.float64)
    return {
        "examples": len(model_errors),
        "mean_physical_baseline_esr": baseline_esr,
        "mean_model_esr": model_esr,
        "mean_relative_improvement": 1.0 - model_esr / max(baseline_esr, 1.0e-12),
        "median_model_esr": float(np.median(model_errors)),
        "p95_model_esr": float(np.quantile(model_errors, 0.95)),
        "peak_ratio_examples": len(peak_ratios),
        "peak_ratio_median": float(np.median(ratios)),
        "peak_ratio_p95": float(np.quantile(ratios, 0.95)),
        "worst_peak_ratio": float(np.max(ratios)),
        "absolute_peak_error_p95": float(np.quantile(peak_errors, 0.95)),
        "quiet_target_examples": len(quiet_peaks),
        "quiet_prediction_peak_maximum": max(quiet_peaks, default=0.0),
    }


@torch.inference_mode()
def _runtime_checks(model: RatDirectRenderer) -> dict:
    torch.manual_seed(20260830)
    dry = torch.randn(3, 6_173) * 0.04
    controls = torch.tensor(((0.2, 0.7, 0.3), (0.8, 0.4, 0.9), (0.5, 0.5, 0.0)))
    whole, _ = model(dry, controls)
    state = None
    chunks = []
    for start, stop in ((0, 17), (17, 513), (513, 2_121), (2_121, 6_173)):
        rendered, state = model(dry[:, start:stop], controls, state)
        chunks.append(rendered)
    streamed = torch.cat(chunks, dim=1)
    silence = torch.zeros(3, 4_096)
    silent_output, _ = model(silence, controls)
    error = whole - streamed
    return {
        "max_absolute_error": float(error.abs().max()),
        "rms_error": float(error.square().mean().sqrt()),
        "silence_max_absolute_output": float(silent_output.abs().max()),
        "zero_volume_max_absolute_output": float(whole[2].abs().max()),
    }


def train(
    corpus: Path,
    output: Path,
    *,
    epochs: int = 12,
    batch_size: int = 32,
    frames: int = 2_048,
    warmup: int = 1_024,
    hidden_size: int = 32,
    layers: int = 4,
    init_checkpoint: Path | None = None,
    maximum_train_files: int | None = None,
    maximum_eval_files: int | None = None,
) -> dict:
    if (output / "rat-direct.pt").exists() or (output / "metrics.json").exists():
        raise ValueError(f"ASRNN RAT direct output already exists: {output}")
    audit = audit_asrnn(corpus, scan_audio=True)
    fit_paths, calibrate_paths = _partition(rat_files(corpus, "train"))
    fit_paths = _bounded(fit_paths, maximum_train_files)
    eval_paths = _bounded(rat_files(corpus, "eval"), maximum_eval_files)
    torch.manual_seed(20260830)
    np.random.seed(20260830)
    target = device()
    model = RatDirectRenderer(hidden_size=hidden_size, layers=layers).to(target)
    initial_hash = None
    if init_checkpoint is not None:
        initial = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        if (
            initial.get("schema") != 1
            or initial.get("sample_rate") != 48_000
            or initial.get("hidden_size") != hidden_size
            or initial.get("layers") != layers
        ):
            raise ValueError("initial RAT direct checkpoint is incompatible")
        model.load_state_dict(initial["state_dict"], strict=True)
        initial_hash = hashlib.sha256(init_checkpoint.read_bytes()).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    train_loader = DataLoader(RatClips(fit_paths), batch_size=batch_size, shuffle=True)
    calibrate_loader = DataLoader(RatClips(calibrate_paths), batch_size=batch_size)
    eval_loader = DataLoader(RatClips(eval_paths), batch_size=batch_size)
    best_score = float("inf")
    best_state = None
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            dry = batch["dry"].to(target)
            wet = batch["wet"].to(target)
            controls = batch["controls"].to(target)
            state = None
            if warmup:
                with torch.no_grad():
                    _, state = model(dry[:, :warmup], controls, state)
            for start in range(warmup, dry.shape[1], frames):
                stop = min(start + frames, dry.shape[1])
                prediction, state = model(dry[:, start:stop], controls, state)
                loss, _ = _rat_loss(prediction, wet[:, start:stop])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                state = _detach_state(state)
                losses.append(float(loss.detach()))
        calibration = _evaluate(model, calibrate_loader, target, frames, warmup)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "calibrate_model_esr": calibration["mean_model_esr"],
            "calibrate_median_esr": calibration["median_model_esr"],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if calibration["mean_model_esr"] < best_score:
            best_score = calibration["mean_model_esr"]
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("ASRNN RAT direct training produced no checkpoint")
    model.load_state_dict(best_state)
    evaluation = _evaluate(model, eval_loader, target, frames, warmup)
    runtime = _runtime_checks(model.cpu().eval())
    accepted = bool(
        evaluation["mean_model_esr"] <= 0.25
        and evaluation["median_model_esr"] <= 0.15
        and evaluation["p95_model_esr"] <= 1.0
        and 0.70 <= evaluation["peak_ratio_median"] <= 1.30
        and evaluation["peak_ratio_p95"] <= 1.50
        and evaluation["absolute_peak_error_p95"] <= 0.02
        and evaluation["quiet_prediction_peak_maximum"] <= 5.0e-4
        and runtime["max_absolute_error"] <= 2.0e-6
        and runtime["silence_max_absolute_output"] == 0.0
        and runtime["zero_volume_max_absolute_output"] == 0.0
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "rat-direct.pt"
    torch.save(
        {
            "schema": 1,
            "sample_rate": 48_000,
            "hidden_size": hidden_size,
            "layers": layers,
            "state_dict": best_state,
            "device_domain": "ASRNN ProCo RAT",
            "dataset_record": RECORD_URL,
            "dataset_license": LICENSE,
        },
        checkpoint,
    )
    report = {
        "schema": 1,
        "status": "accepted-real-hardware-public-pilot" if accepted else "rejected",
        "accepted": accepted,
        "scope": {
            "real_hardware_data": True,
            "internal_noncommercial_research_only": True,
            "product_or_ui_promotion_allowed": False,
            "independent_locked_final_available": False,
        },
        "dataset": {"record_url": RECORD_URL, "license": LICENSE, "audit": audit},
        "partitions": {
            "fit_files": len(fit_paths),
            "calibrate_files": len(calibrate_paths),
            "official_eval_files": len(eval_paths),
            "calibrate_from_official_train": True,
        },
        "architecture": {
            "type": "bias-free deep LSTM direct renderer",
            "hidden_size": hidden_size,
            "layers": layers,
            "volume_factorization": "volume_squared_times_learned_positive_gain",
            "zero_input_zero_output_by_construction": True,
        },
        "training_geometry": {
            "clip_frames": 48_000,
            "warmup_frames": warmup,
            "tbptt_frames": frames,
            "batch_size": batch_size,
        },
        "initial_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "initial_checkpoint_sha256": initial_hash,
        "official_eval": evaluation,
        "runtime": runtime,
        "history": history,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "quality_policy": {
            "source_files_read_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
        },
        "evaluation_limitations": [
            "the public corpus has no independent locked-final partition",
            "official eval was observed during development architecture selection",
            "acceptance is limited to an internal non-commercial pilot",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frames", type=int, default=2_048)
    parser.add_argument("--warmup", type=int, default=1_024)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--maximum-train-files", type=int)
    parser.add_argument("--maximum-eval-files", type=int)
    args = parser.parse_args()
    report = train(
        args.corpus,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        frames=args.frames,
        warmup=args.warmup,
        hidden_size=args.hidden_size,
        layers=args.layers,
        init_checkpoint=args.init_checkpoint,
        maximum_train_files=args.maximum_train_files,
        maximum_eval_files=args.maximum_eval_files,
    )
    print(json.dumps({"accepted": report["accepted"], "official_eval": report["official_eval"]}, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
