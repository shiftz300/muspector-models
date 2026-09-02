#!/usr/bin/env python3
"""Train a peak-aware residual adapter while freezing a stable ASRNN effect."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .asrnn_effects import effect_files, read_effect_pair
from .forward_drive import error_to_signal_ratio, pre_emphasis
from .stable_effect import load_stable_effect
from .stable_effect_adapter import StableEffectResidualAdapter
from .train import device


def _partition(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    fit, calibrate = [], []
    for path in paths:
        take_id = int(path.stem.split(",")[-1])
        (calibrate if take_id % 5 == 0 else fit).append(path)
    if not fit or not calibrate:
        raise ValueError("adapter requires non-empty fit and calibrate partitions")
    return fit, calibrate


def _window_starts(wet: np.ndarray, dry: np.ndarray, frames: int) -> tuple[int, ...]:
    maximum = max(0, len(wet) - frames)
    centers = (int(np.argmax(np.abs(wet))), int(np.argmax(np.abs(dry))))
    return tuple(min(max(center - frames // 3, 0), maximum) for center in centers)


@torch.inference_mode()
def _precompute_windows(
    base,
    paths: list[Path],
    device_key: str,
    frames: int,
    target: torch.device,
) -> list[dict]:
    rows = []
    base = base.to(target).eval()
    for index, path in enumerate(paths):
        dry, wet, controls = read_effect_pair(path, device_key)
        for start in _window_starts(wet, dry, frames):
            dry_window = torch.from_numpy(dry[start : start + frames]).unsqueeze(0).to(target)
            control = torch.from_numpy(controls).unsqueeze(0).to(target)
            rendered, _ = base(dry_window, control)
            rows.append(
                {
                    "dry": dry_window.squeeze(0).cpu(),
                    "base": rendered.squeeze(0).cpu(),
                    "wet": torch.from_numpy(wet[start : start + frames].copy()),
                    "controls": control.squeeze(0).cpu(),
                }
            )
        if (index + 1) % 64 == 0:
            print(json.dumps({"precomputed_files": index + 1, "total": len(paths)}))
    return rows


@torch.inference_mode()
def _precompute_full(
    base, paths: list[Path], device_key: str, target: torch.device
) -> list[dict]:
    rows = []
    base = base.to(target).eval()
    for path in paths:
        dry, wet, controls = read_effect_pair(path, device_key)
        dry_tensor = torch.from_numpy(dry).unsqueeze(0).to(target)
        control = torch.from_numpy(controls).unsqueeze(0).to(target)
        state = None
        chunks = []
        for start in range(0, len(dry), 2_048):
            rendered, state = base(dry_tensor[:, start : start + 2_048], control, state)
            chunks.append(rendered)
        rows.append(
            {
                "dry": dry_tensor.squeeze(0).cpu(),
                "base": torch.cat(chunks, dim=1).squeeze(0).cpu(),
                "wet": torch.from_numpy(wet.copy()),
                "controls": control.squeeze(0).cpu(),
            }
        )
    return rows


def _loss(prediction: torch.Tensor, wet: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    prediction = prediction[:, 1_024:]
    wet = wet[:, 1_024:]
    base = base[:, 1_024:]
    waveform = error_to_signal_ratio(prediction, wet)
    emphasized = error_to_signal_ratio(pre_emphasis(prediction), pre_emphasis(wet))
    normalized_l1 = (prediction - wet).abs().mean() / wet.abs().mean().clamp_min(1.0e-6)
    peak_error = (
        (prediction.abs().amax(1) - wet.abs().amax(1)).abs()
        / wet.abs().amax(1).clamp_min(1.0e-5)
    ).mean()
    residual = (prediction - base).square().mean() / wet.square().mean().clamp_min(1.0e-6)
    return waveform + 0.5 * emphasized + 0.25 * normalized_l1 + 2.0 * peak_error + 0.01 * residual


@torch.inference_mode()
def _evaluate(adapter, rows: list[dict], target: torch.device, batch_size: int) -> dict:
    adapter.eval()
    error_sum = energy_sum = base_error_sum = 0.0
    per_file, peak_errors, ratios = [], [], []
    for batch in DataLoader(rows, batch_size=batch_size):
        dry = batch["dry"].to(target)
        base = batch["base"].to(target)
        wet = batch["wet"].to(target)
        controls = batch["controls"].to(target)
        state = None
        chunks = []
        for start in range(0, dry.shape[1], 2_048):
            value, state = adapter(
                dry[:, start : start + 2_048],
                base[:, start : start + 2_048],
                controls,
                state,
            )
            chunks.append(value)
        predicted = torch.cat(chunks, dim=1)[:, 1_024:]
        base = base[:, 1_024:]
        wet = wet[:, 1_024:]
        error = (predicted - wet).square()
        energy = wet.square()
        error_sum += float(error.sum())
        energy_sum += float(energy.sum())
        base_error_sum += float((base - wet).square().sum())
        per_file.extend((error.mean(1) / energy.mean(1).clamp_min(1.0e-8)).cpu().tolist())
        target_peak = wet.abs().amax(1)
        predicted_peak = predicted.abs().amax(1)
        peak_errors.extend((predicted_peak - target_peak).abs().cpu().tolist())
        ratios.extend((predicted_peak / target_peak.clamp_min(1.0e-8)).cpu().tolist())
    metrics = {
        "examples": len(rows),
        "global_esr": error_sum / energy_sum,
        "base_global_esr": base_error_sum / energy_sum,
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
def _runtime(adapter: StableEffectResidualAdapter) -> dict:
    torch.manual_seed(20260831)
    dry = torch.randn(2, 6_173) * 0.04
    base = torch.tanh(dry * 2.0)
    controls = torch.tensor(((0.2,), (0.8,)))
    whole, _ = adapter(dry, base, controls)
    state = None
    chunks = []
    for start, stop in ((0, 17), (17, 513), (513, 2_121), (2_121, 6_173)):
        value, state = adapter(dry[:, start:stop], base[:, start:stop], controls, state)
        chunks.append(value)
    silence, _ = adapter(torch.zeros_like(dry), torch.zeros_like(base), controls)
    return {
        "stream_max_absolute_error": float(
            (whole - torch.cat(chunks, dim=1)).abs().max()
        ),
        "static_silence_max_absolute_output": float(silence.abs().max()),
    }


def train(args) -> dict:
    if args.output.exists():
        raise ValueError(f"adapter output already exists: {args.output}")
    torch.manual_seed(20260831)
    np.random.seed(20260831)
    target = device()
    base, payload = load_stable_effect(args.base)
    if payload["device"] != args.device:
        raise ValueError("adapter device differs from base checkpoint")
    fit_paths, calibrate_paths = _partition(
        effect_files(args.corpus.resolve(), args.device, "train")
    )
    fit_rows = _precompute_windows(
        base, fit_paths, args.device, args.frames, target
    )
    calibrate_rows = _precompute_full(base, calibrate_paths, args.device, target)
    adapter = StableEffectResidualAdapter(
        int(payload["control_count"]), args.hidden_size
    ).to(target)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=1.0e-6)
    initial = _evaluate(adapter, calibrate_rows, target, args.batch_size)
    history = [{"epoch": 0, "train_loss": None, "calibrate": initial}]
    best = {
        name: value.detach().cpu().clone()
        for name, value in adapter.state_dict().items()
    }
    best_key = (
        not initial["passes_selection_gate"],
        initial["absolute_peak_error_p95"],
        initial["global_esr"],
    )
    for epoch in range(args.epochs):
        adapter.train()
        losses = []
        generator = torch.Generator().manual_seed(20260831 + epoch)
        loader = DataLoader(
            fit_rows, batch_size=args.batch_size, shuffle=True, generator=generator
        )
        for batch in loader:
            dry = batch["dry"].to(target)
            base_audio = batch["base"].to(target)
            wet = batch["wet"].to(target)
            controls = batch["controls"].to(target)
            predicted, _ = adapter(dry, base_audio, controls)
            loss = _loss(predicted, wet, base_audio)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = _evaluate(adapter, calibrate_rows, target, args.batch_size)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "calibrate": metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        key = (
            not metrics["passes_selection_gate"],
            metrics["absolute_peak_error_p95"],
            metrics["global_esr"],
        )
        if best_key is None or key < best_key:
            best_key = key
            best = {
                name: value.detach().cpu().clone()
                for name, value in adapter.state_dict().items()
            }
    if best is None:
        raise RuntimeError("adapter training produced no candidate")
    adapter.load_state_dict(best)
    calibration = _evaluate(adapter, calibrate_rows, target, args.batch_size)
    runtime = _runtime(adapter.cpu().eval())
    admitted = bool(
        calibration["passes_selection_gate"]
        and runtime["stream_max_absolute_error"] <= 2.0e-6
        and runtime["static_silence_max_absolute_output"] == 0.0
    )
    args.output.mkdir(parents=True)
    checkpoint = args.output / "effect-residual-adapter.pt"
    if admitted:
        torch.save(
            {
                "schema": 1,
                "sample_rate": 48_000,
                "device": args.device,
                "control_count": adapter.control_count,
                "hidden_size": adapter.hidden_size,
                "state_dict": best,
                "base_checkpoint_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
            },
            checkpoint,
        )
    report = {
        "schema": 1,
        "status": "admitted-for-one-shot-official-eval" if admitted else "rejected-on-calibrate",
        "admitted_for_official_eval": admitted,
        "device": args.device,
        "base_checkpoint": str(args.base),
        "base_checkpoint_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
        "checkpoint": str(checkpoint) if admitted else None,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if admitted else None,
        "fit_files": len(fit_paths),
        "fit_windows": len(fit_rows),
        "calibrate_files": len(calibrate_paths),
        "official_eval_opened": False,
        "base_frozen": True,
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
    parser.add_argument("--hidden-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["admitted_for_official_eval"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
