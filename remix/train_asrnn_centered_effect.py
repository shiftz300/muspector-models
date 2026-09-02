#!/usr/bin/env python3
"""Peak-aware recurrent fine-tuning for a zero-centered stable ASRNN effect."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .asrnn_effects import effect_files, read_effect_pair
from .centered_stable_effect import CenteredStableEffectRenderer
from .forward_drive import error_to_signal_ratio, pre_emphasis
from .stable_effect import load_stable_effect
from .train import device


def _partition(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    fit, calibrate = [], []
    for path in paths:
        take_id = int(path.stem.split(",")[-1])
        (calibrate if take_id % 5 == 0 else fit).append(path)
    if not fit or not calibrate:
        raise ValueError("centered training requires fit and calibrate partitions")
    return fit, calibrate


def _rows(paths: list[Path], device_key: str, frames: int | None) -> list[dict]:
    rows = []
    for path in paths:
        dry, wet, controls = read_effect_pair(path, device_key)
        starts = (0,)
        if frames is not None:
            maximum = max(0, len(wet) - frames)
            centers = (int(np.argmax(np.abs(wet))), int(np.argmax(np.abs(dry))))
            starts = tuple(
                min(max(center - frames // 3, 0), maximum) for center in centers
            )
        for start in starts:
            stop = len(wet) if frames is None else start + frames
            rows.append(
                {
                    "dry": torch.from_numpy(dry[start:stop].copy()),
                    "wet": torch.from_numpy(wet[start:stop].copy()),
                    "controls": torch.from_numpy(controls.copy()),
                }
            )
    return rows


def _loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction[:, 1_024:]
    target = target[:, 1_024:]
    waveform = error_to_signal_ratio(prediction, target)
    emphasized = error_to_signal_ratio(pre_emphasis(prediction), pre_emphasis(target))
    normalized_l1 = (prediction - target).abs().mean() / target.abs().mean().clamp_min(1.0e-6)
    absolute_peak = (
        prediction.abs().amax(1) - target.abs().amax(1)
    ).abs().mean()
    relative_peak = (
        (prediction.abs().amax(1) - target.abs().amax(1)).abs()
        / target.abs().amax(1).clamp_min(1.0e-5)
    ).mean()
    return (
        waveform
        + 0.5 * emphasized
        + 0.25 * normalized_l1
        + 3.0 * relative_peak
        + 10.0 * absolute_peak
    )


@torch.inference_mode()
def _evaluate(model, rows: list[dict], target: torch.device, batch_size: int) -> dict:
    model.eval()
    error_sum = energy_sum = 0.0
    per_file, peak_errors, ratios = [], [], []
    for batch in DataLoader(rows, batch_size=batch_size):
        dry = batch["dry"].to(target)
        wet = batch["wet"].to(target)
        controls = batch["controls"].to(target)
        state = None
        chunks = []
        for start in range(0, dry.shape[1], 2_048):
            value, state = model(dry[:, start : start + 2_048], controls, state)
            chunks.append(value)
        prediction = torch.cat(chunks, dim=1)[:, 1_024:]
        wet = wet[:, 1_024:]
        error = (prediction - wet).square()
        energy = wet.square()
        error_sum += float(error.sum())
        energy_sum += float(energy.sum())
        per_file.extend((error.mean(1) / energy.mean(1).clamp_min(1.0e-8)).cpu().tolist())
        target_peak = wet.abs().amax(1)
        prediction_peak = prediction.abs().amax(1)
        peak_errors.extend((prediction_peak - target_peak).abs().cpu().tolist())
        ratios.extend((prediction_peak / target_peak.clamp_min(1.0e-8)).cpu().tolist())
    metrics = {
        "examples": len(rows),
        "global_esr": error_sum / energy_sum,
        "mean_per_file_esr": float(np.mean(per_file)),
        "p95_per_file_esr": float(np.quantile(per_file, 0.95)),
        "absolute_peak_error_p95": float(np.quantile(peak_errors, 0.95)),
        "peak_ratio_p05": float(np.quantile(ratios, 0.05)),
        "peak_ratio_median": float(np.median(ratios)),
        "peak_ratio_p95": float(np.quantile(ratios, 0.95)),
    }
    metrics["passes_selection_gate"] = bool(
        metrics["global_esr"] <= 0.05
        and metrics["mean_per_file_esr"] <= 0.10
        and metrics["p95_per_file_esr"] <= 0.25
        and metrics["absolute_peak_error_p95"] <= 0.02
        and 0.75 <= metrics["peak_ratio_median"] <= 1.25
        and metrics["peak_ratio_p95"] <= 1.35
    )
    return metrics


@torch.inference_mode()
def _runtime(model: CenteredStableEffectRenderer) -> dict:
    torch.manual_seed(20260901)
    dry = torch.randn(2, 6_173) * 0.04
    controls = torch.tensor(((0.2,), (0.8,)))
    whole, _ = model(dry, controls)
    state = None
    chunks = []
    for start, stop in ((0, 17), (17, 513), (513, 2_121), (2_121, 6_173)):
        value, state = model(dry[:, start:stop], controls, state)
        chunks.append(value)
    silence, _ = model(torch.zeros_like(dry), controls)
    return {
        "stream_max_absolute_error": float(
            (whole - torch.cat(chunks, dim=1)).abs().max()
        ),
        "static_silence_max_absolute_output": float(silence.abs().max()),
        "candidate_recurrent_infinity_norms": model.project_candidate_recurrence(),
    }


def train(args) -> dict:
    if args.output.exists():
        raise ValueError(f"centered training output already exists: {args.output}")
    torch.manual_seed(20260901)
    np.random.seed(20260901)
    target = device()
    base, payload = load_stable_effect(args.base)
    if payload["device"] != args.device:
        raise ValueError("centered training device differs from base checkpoint")
    model = CenteredStableEffectRenderer(base).to(target)
    fit_paths, calibrate_paths = _partition(
        effect_files(args.corpus.resolve(), args.device, "train")
    )
    fit_rows = _rows(fit_paths, args.device, args.frames)
    calibrate_rows = _rows(calibrate_paths, args.device, None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-7)
    initial_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    initial = _evaluate(model, calibrate_rows, target, args.batch_size)
    history = [{"epoch": 0, "train_loss": None, "calibrate": initial}]
    best_state = initial_state
    best_key = (
        not initial["passes_selection_gate"],
        initial["absolute_peak_error_p95"],
        initial["global_esr"],
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(20260901 + epoch)
        loader = DataLoader(
            fit_rows, batch_size=args.batch_size, shuffle=True, generator=generator
        )
        losses = []
        for batch in loader:
            dry = batch["dry"].to(target)
            wet = batch["wet"].to(target)
            controls = batch["controls"].to(target)
            prediction, _ = model(dry, controls)
            loss = _loss(prediction, wet)
            anchor = sum(
                (parameter - initial_state[name].to(target)).square().mean()
                for name, parameter in model.named_parameters()
            )
            loss = loss + 1.0e-4 * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            model.project_candidate_recurrence()
            losses.append(float(loss.detach()))
        if epoch % args.evaluate_every != 0 and epoch != args.epochs:
            print(
                json.dumps({"epoch": epoch, "train_loss": float(np.mean(losses))}),
                flush=True,
            )
            continue
        metrics = _evaluate(model, calibrate_rows, target, args.batch_size)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "calibrate": metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = (
            not metrics["passes_selection_gate"],
            metrics["absolute_peak_error_p95"],
            metrics["global_esr"],
        )
        if key < best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    calibration = _evaluate(model, calibrate_rows, target, args.batch_size)
    runtime = _runtime(model.cpu().eval())
    admitted = bool(
        calibration["passes_selection_gate"]
        and runtime["stream_max_absolute_error"] <= 2.0e-6
        and runtime["static_silence_max_absolute_output"] == 0.0
        and max(runtime["candidate_recurrent_infinity_norms"]) <= 0.995001
    )
    args.output.mkdir(parents=True)
    checkpoint = args.output / "centered-stable-effect.pt"
    if admitted:
        torch.save(
            {
                "schema": 1,
                "sample_rate": 48_000,
                "architecture": "zero-centered-stable-conditioned-lstm",
                "device": args.device,
                "device_name": payload["device_name"],
                "control_count": int(payload["control_count"]),
                "control_names": payload["control_names"],
                "inverted_controls": payload["inverted_controls"],
                "hidden_size": int(payload["hidden_size"]),
                "layers": int(payload["layers"]),
                "input_coef": float(payload["input_coef"]),
                "state_dict": model.base.state_dict(),
                "source_checkpoint_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
                "dataset_license": payload["dataset_license"],
                "scope": "internal non-commercial development pilot",
            },
            checkpoint,
        )
    report = {
        "schema": 1,
        "status": "admitted-for-one-shot-official-eval" if admitted else "rejected-on-calibrate",
        "admitted_for_official_eval": admitted,
        "device": args.device,
        "base_checkpoint_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
        "checkpoint": str(checkpoint) if admitted else None,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if admitted else None,
        "fit_files": len(fit_paths),
        "fit_windows": len(fit_rows),
        "calibrate_files": len(calibrate_paths),
        "official_eval_opened": False,
        "calibration": calibration,
        "runtime": runtime,
        "history": history,
        "quality_policy": {
            "source_files_read_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
        },
    }
    (args.output / "training.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cs3",), required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8_192)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--evaluate-every", type=int, default=2)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["admitted_for_official_eval"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
