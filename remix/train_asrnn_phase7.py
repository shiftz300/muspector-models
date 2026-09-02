#!/usr/bin/env python3
"""Group-robust full-clip CS-3 recurrent fine-tuning for Phase 7."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from .asrnn_effects import effect_files, read_effect_pair
from .centered_stable_effect import (
    CenteredStableEffectRenderer,
    detach_centered_state,
)
from .stable_effect import load_stable_effect
from .train import device


def _partition(paths: list[Path], audit: dict) -> tuple[list[Path], list[Path]]:
    fit_names = set(audit["fit_files"])
    calibration_names = set(audit["calibration_files"])
    available = {path.name for path in paths}
    if fit_names & calibration_names or fit_names | calibration_names != available:
        raise ValueError("Phase-7 audit partition does not exactly cover the train split")
    if audit.get("fit_calibration_near_performance_pairs") != 0:
        raise ValueError("Phase-7 requires a clean near-performance-group audit")
    return (
        [path for path in paths if path.name in fit_names],
        [path for path in paths if path.name in calibration_names],
    )


def _rows(paths: list[Path], device_key: str) -> list[dict]:
    rows = []
    for path in paths:
        dry, wet, controls = read_effect_pair(path, device_key)
        rows.append(
            {
                "dry": torch.from_numpy(dry.copy()),
                "wet": torch.from_numpy(wet.copy()),
                "controls": torch.from_numpy(controls.copy()),
                "attack": int(path.stem.split(",")[0]),
                "name": path.name,
            }
        )
    return rows


def _spectral_per_example(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for size in (256, 1_024):
        window = torch.hann_window(size, device=prediction.device)
        predicted = torch.stft(
            prediction,
            n_fft=size,
            hop_length=size // 4,
            window=window,
            center=False,
            return_complex=True,
        ).abs()
        expected = torch.stft(
            target,
            n_fft=size,
            hop_length=size // 4,
            window=window,
            center=False,
            return_complex=True,
        ).abs()
        losses.append(
            (torch.log1p(predicted) - torch.log1p(expected)).abs().mean((1, 2))
            / torch.log1p(expected).abs().mean((1, 2)).clamp_min(1.0e-5)
        )
    return torch.stack(losses).mean(0)


def _per_example_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = prediction - target
    waveform = error.square().mean(1) / target.square().mean(1).clamp_min(1.0e-7)
    emphasized_prediction = prediction[:, 1:] - 0.95 * prediction[:, :-1]
    emphasized_target = target[:, 1:] - 0.95 * target[:, :-1]
    emphasized = (emphasized_prediction - emphasized_target).square().mean(1) / (
        emphasized_target.square().mean(1).clamp_min(1.0e-7)
    )
    normalized_l1 = error.abs().mean(1) / target.abs().mean(1).clamp_min(1.0e-5)
    predicted_peak = prediction.abs().amax(1)
    target_peak = target.abs().amax(1)
    absolute_peak = (predicted_peak - target_peak).abs()
    audible = target_peak >= 1.0e-3
    relative_peak = torch.where(
        audible,
        absolute_peak / target_peak.clamp_min(1.0e-5),
        absolute_peak * 100.0,
    )
    predicted_envelope = functional.max_pool1d(
        prediction.abs().unsqueeze(1), 256, stride=128
    ).squeeze(1)
    target_envelope = functional.max_pool1d(
        target.abs().unsqueeze(1), 256, stride=128
    ).squeeze(1)
    envelope = (predicted_envelope - target_envelope).abs().mean(1) / (
        target_envelope.mean(1).clamp_min(1.0e-5)
    )
    spectral = _spectral_per_example(prediction, target)
    return (
        waveform
        + 0.5 * emphasized
        + 0.25 * normalized_l1
        + envelope
        + 2.0 * relative_peak
        + 8.0 * absolute_peak
        + 0.05 * spectral
    ).clamp_max(20.0)


def _group_cvar(losses: torch.Tensor, fraction: float = 0.25) -> torch.Tensor:
    count = max(1, int(np.ceil(losses.numel() * fraction)))
    return 0.5 * losses.mean() + 0.5 * losses.topk(count).values.mean()


@torch.inference_mode()
def _evaluate(model, rows: list[dict], target: torch.device, batch_size: int) -> dict:
    model.eval()
    error_sum = energy_sum = 0.0
    per_file, peak_errors, ratios = [], [], []
    attacks: dict[int, list[dict]] = defaultdict(list)
    for batch in DataLoader(rows, batch_size=batch_size):
        dry = batch["dry"].to(target)
        wet = batch["wet"].to(target)
        controls = batch["controls"].to(target)
        state = None
        chunks = []
        for start in range(0, dry.shape[1], 4_096):
            value, state = model(dry[:, start : start + 4_096], controls, state)
            chunks.append(value)
        prediction = torch.cat(chunks, dim=1)[:, 1_024:]
        expected = wet[:, 1_024:]
        error = (prediction - expected).square()
        energy = expected.square()
        file_esr = error.mean(1) / energy.mean(1).clamp_min(1.0e-8)
        expected_peak = expected.abs().amax(1)
        predicted_peak = prediction.abs().amax(1)
        file_peak_error = (predicted_peak - expected_peak).abs()
        file_ratio = predicted_peak / expected_peak.clamp_min(1.0e-8)
        error_sum += float(error.sum())
        energy_sum += float(energy.sum())
        per_file.extend(file_esr.cpu().tolist())
        peak_errors.extend(file_peak_error.cpu().tolist())
        ratios.extend(file_ratio.cpu().tolist())
        for attack, esr, peak in zip(
            batch["attack"].tolist(), file_esr.cpu().tolist(), file_peak_error.cpu().tolist()
        ):
            attacks[int(attack)].append({"esr": esr, "peak_error": peak})
    by_attack = {}
    for attack, values in sorted(attacks.items()):
        by_attack[str(attack)] = {
            "files": len(values),
            "mean_esr": float(np.mean([value["esr"] for value in values])),
            "absolute_peak_error_p95": float(
                np.quantile([value["peak_error"] for value in values], 0.95)
            ),
        }
    worst_attack = max(
        value["absolute_peak_error_p95"] for value in by_attack.values()
    )
    metrics = {
        "examples": len(rows),
        "global_esr": error_sum / energy_sum,
        "mean_per_file_esr": float(np.mean(per_file)),
        "p95_per_file_esr": float(np.quantile(per_file, 0.95)),
        "absolute_peak_error_p95": float(np.quantile(peak_errors, 0.95)),
        "worst_attack_absolute_peak_error_p95": worst_attack,
        "peak_ratio_p05": float(np.quantile(ratios, 0.05)),
        "peak_ratio_median": float(np.median(ratios)),
        "peak_ratio_p95": float(np.quantile(ratios, 0.95)),
        "by_attack": by_attack,
    }
    metrics["passes_selection_gate"] = bool(
        metrics["global_esr"] <= 0.05
        and metrics["mean_per_file_esr"] <= 0.10
        and metrics["p95_per_file_esr"] <= 0.25
        and metrics["absolute_peak_error_p95"] <= 0.02
        and metrics["worst_attack_absolute_peak_error_p95"] <= 0.03
        and 0.75 <= metrics["peak_ratio_median"] <= 1.25
        and metrics["peak_ratio_p95"] <= 1.35
    )
    return metrics


@torch.inference_mode()
def _runtime(model: CenteredStableEffectRenderer) -> dict:
    torch.manual_seed(20260902)
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


def _calibrate(model, rows: list[dict]) -> dict:
    evaluation_model = copy.deepcopy(model).cpu().eval()
    return _evaluate(evaluation_model, rows, torch.device("cpu"), 32)


def train(args) -> dict:
    if args.output.exists():
        raise ValueError(f"Phase-7 output already exists: {args.output}")
    torch.manual_seed(20260902)
    np.random.seed(20260902)
    target = device()
    print(
        json.dumps({"phase": "phase-7", "compute_device": str(target), "epochs": args.epochs}),
        flush=True,
    )
    base, payload = load_stable_effect(args.base)
    if payload["device"] != "cs3":
        raise ValueError("Phase 7 requires a CS-3 base checkpoint")
    model = CenteredStableEffectRenderer(base).to(target)
    audit = json.loads(args.group_audit.read_text())
    fit_paths, calibration_paths = _partition(
        effect_files(args.corpus.resolve(), "cs3", "train"), audit
    )
    fit_rows = _rows(fit_paths, "cs3")
    calibration_rows = _rows(calibration_paths, "cs3")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-7
    )
    initial_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    anchor_state = {
        name: value.to(target) for name, value in initial_state.items()
    }
    initial = _calibrate(model, calibration_rows)
    history = [{"epoch": 0, "train_loss": None, "calibration": initial}]
    best_state = initial_state
    best_key = (
        not initial["passes_selection_gate"],
        initial["worst_attack_absolute_peak_error_p95"],
        initial["absolute_peak_error_p95"],
        initial["global_esr"],
    )
    optimizer_updates = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(20260902 + epoch)
        loader = DataLoader(
            fit_rows, batch_size=args.batch_size, shuffle=True, generator=generator
        )
        losses = []
        for batch_index, batch in enumerate(loader, start=1):
            dry = batch["dry"].to(target)
            wet = batch["wet"].to(target)
            controls = batch["controls"].to(target)
            state = None
            batch_losses = []
            chunk_starts = list(range(0, dry.shape[1], args.frames))
            for index, start in enumerate(chunk_starts):
                optimizer.zero_grad(set_to_none=True)
                prediction, state = model(
                    dry[:, start : start + args.frames], controls, state
                )
                expected = wet[:, start : start + args.frames]
                if index == 0:
                    prediction = prediction[:, 1_024:]
                    expected = expected[:, 1_024:]
                loss = _group_cvar(_per_example_loss(prediction, expected))
                anchor = sum(
                    (parameter - anchor_state[name]).square().mean()
                    for name, parameter in model.named_parameters()
                )
                (loss + args.anchor_weight * anchor).backward()
                batch_losses.append(float(loss.detach()))
                state = detach_centered_state(state)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                model.project_candidate_recurrence()
                optimizer_updates += 1
            losses.append(float(np.mean(batch_losses)))
            if batch_index % 4 == 0 or batch_index == len(loader):
                print(
                    json.dumps({"epoch": epoch, "batch": batch_index, "batches": len(loader), "optimizer_updates": optimizer_updates}),
                    flush=True,
                )
        metrics = _calibrate(model, calibration_rows)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "calibration": metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = (
            not metrics["passes_selection_gate"],
            metrics["worst_attack_absolute_peak_error_p95"],
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
    calibration = _calibrate(model, calibration_rows)
    runtime = _runtime(model.cpu().eval())
    admitted = bool(
        calibration["passes_selection_gate"]
        and runtime["stream_max_absolute_error"] <= 2.0e-6
        and runtime["static_silence_max_absolute_output"] == 0.0
        and max(runtime["candidate_recurrent_infinity_norms"]) <= 0.995001
    )
    args.output.mkdir(parents=True)
    checkpoint = args.output / "phase7-centered-stable-effect.pt"
    torch.save(
        {
                "schema": 1,
                "sample_rate": 48_000,
                "architecture": "zero-centered-stable-conditioned-lstm",
                "device": "cs3",
                "device_name": payload["device_name"],
                "control_count": int(payload["control_count"]),
                "control_names": payload["control_names"],
                "inverted_controls": payload["inverted_controls"],
                "hidden_size": int(payload["hidden_size"]),
                "layers": int(payload["layers"]),
                "input_coef": float(payload["input_coef"]),
                "state_dict": model.base.state_dict(),
                "source_checkpoint_sha256": hashlib.sha256(
                    args.base.read_bytes()
                ).hexdigest(),
                "dataset_license": payload["dataset_license"],
                "scope": "internal non-commercial development pilot",
                "diagnostic_only": not admitted,
        },
        checkpoint,
    )
    report = {
        "schema": 1,
        "phase": "phase-7-group-cvar-full-clip",
        "status": "admitted-for-development-challenge" if admitted else "rejected-on-calibration",
        "admitted_for_development_challenge": admitted,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "diagnostic_checkpoint_only": not admitted,
        "compute_device": str(target),
        "configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "anchor_weight": args.anchor_weight,
            "optimizer_update_granularity": "each causal TBPTT chunk",
            "optimizer_updates": optimizer_updates,
            "calibration_compute_device": "cpu",
        },
        "fit_files": len(fit_paths),
        "calibration_files": len(calibration_paths),
        "chunk_frames": args.frames,
        "history": history,
        "calibration": calibration,
        "runtime": runtime,
        "development_challenge_opened": False,
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
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--group-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16_384)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--anchor-weight", type=float, default=5.0e-4)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["admitted_for_development_challenge"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
