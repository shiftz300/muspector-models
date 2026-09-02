#!/usr/bin/env python3
"""Non-commercial real-hardware pilot for the public ASRNN ProCo RAT data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .asrnn_data import LICENSE, RECORD_URL, audit_asrnn, rat_files, read_rat_pair
from .drive_adapter import DriveDeviceAdapter
from .forward_chain import DRIVE_CHECKPOINT, _load_drive
from .forward_drive import pre_emphasis
from .train import device
from .train_drive_adapter import _stream_parity


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/asrnn-rat-adapter-pilot"


def _partition(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    fit, calibrate = [], []
    for path in paths:
        bucket = int(hashlib.sha256(path.name.encode()).hexdigest()[:8], 16) % 10
        (calibrate if bucket == 0 else fit).append(path)
    if not fit or not calibrate:
        raise ValueError("ASRNN train split cannot form deterministic fit/calibrate partitions")
    return fit, calibrate


def _bounded(paths: list[Path], maximum: int | None) -> list[Path]:
    if maximum is None or maximum >= len(paths):
        return paths
    if maximum <= 0:
        raise ValueError("maximum file count must be positive")
    # Filename order is correlated with the control values. A stable hash gives a
    # deterministic pilot subset without collapsing control coverage.
    ranked = sorted(paths, key=lambda path: hashlib.sha256(path.name.encode()).digest())
    return sorted(ranked[:maximum])


class RatClips(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        if not paths:
            raise ValueError("RAT clip dataset is empty")
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        dry, wet, controls = read_rat_pair(path)
        return {
            "dry": torch.from_numpy(dry),
            "wet": torch.from_numpy(wet),
            "controls": torch.from_numpy(controls),
            "file": path.name,
        }


def _detach_state(state):
    if isinstance(state, tuple):
        return tuple(value.detach() for value in state)
    return state.detach()


def _rat_loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Phase-preserving, quiet-target-safe objective for raw capture levels."""
    mae = torch.nn.functional.l1_loss(prediction, target)
    emphasized_mae = torch.nn.functional.l1_loss(
        pre_emphasis(prediction), pre_emphasis(target)
    )
    target_peak = target.abs().amax(dim=1, keepdim=True)
    absolute_overshoot = torch.relu(
        prediction.abs() - (target_peak * 1.10 + 1.0e-4)
    ).mean()
    quiet = target_peak.squeeze(1) < 1.0e-3
    quiet_output = (
        prediction[quiet].abs().mean()
        if bool(quiet.any())
        else prediction.square().mean() * 0.0
    )
    total = mae + 0.25 * emphasized_mae + 0.25 * absolute_overshoot + quiet_output
    return total, {
        "mae": float(mae.detach()),
        "preemphasis_mae": float(emphasized_mae.detach()),
        "absolute_overshoot": float(absolute_overshoot.detach()),
        "quiet_output": float(quiet_output.detach()),
    }


@torch.inference_mode()
def _evaluate(
    base: torch.nn.Module,
    adapter: DriveDeviceAdapter,
    loader: DataLoader,
    target: torch.device,
    frames: int,
    warmup: int,
) -> dict:
    base.eval()
    adapter.eval()
    base_errors, model_errors, peak_ratios = [], [], []
    absolute_peak_errors, quiet_prediction_peaks = [], []
    for batch in loader:
        dry = batch["dry"].to(target)
        wet = batch["wet"].to(target)
        controls = batch["controls"].to(target)
        base_state = None
        adapter_state = None
        generic_chunks, prediction_chunks = [], []
        for start in range(0, dry.shape[1], frames):
            stop = min(start + frames, dry.shape[1])
            generic, base_state = base(dry[:, start:stop], controls, base_state)
            prediction, adapter_state = adapter(
                dry[:, start:stop], generic, controls, adapter_state
            )
            generic_chunks.append(generic)
            prediction_chunks.append(prediction)
        generic = torch.cat(generic_chunks, dim=1)[:, warmup:]
        prediction = torch.cat(prediction_chunks, dim=1)[:, warmup:]
        wet = wet[:, warmup:]
        energy = wet.square().mean(dim=1).clamp_min(1.0e-8)
        base_errors.extend(((generic - wet).square().mean(dim=1) / energy).cpu().tolist())
        model_errors.extend(((prediction - wet).square().mean(dim=1) / energy).cpu().tolist())
        target_peak = wet.abs().amax(dim=1)
        prediction_peak = prediction.abs().amax(dim=1)
        absolute_peak_errors.extend((prediction_peak - target_peak).abs().cpu().tolist())
        audible = target_peak >= 1.0e-3
        peak_ratios.extend((prediction_peak[audible] / target_peak[audible]).cpu().tolist())
        quiet_prediction_peaks.extend(prediction_peak[~audible].cpu().tolist())
    base_esr = float(np.mean(base_errors))
    model_esr = float(np.mean(model_errors))
    ordered = sorted(float(value) for value in peak_ratios)
    ordered_absolute = sorted(float(value) for value in absolute_peak_errors)
    return {
        "examples": len(model_errors),
        "mean_base_esr": base_esr,
        "mean_model_esr": model_esr,
        "mean_relative_improvement": 1.0 - model_esr / max(base_esr, 1.0e-12),
        "median_model_esr": float(np.median(model_errors)),
        "p95_model_esr": float(np.quantile(model_errors, 0.95)),
        "peak_ratio_examples": len(ordered),
        "peak_ratio_p95": ordered[max(0, int(np.ceil(0.95 * len(ordered))) - 1)],
        "worst_peak_ratio": ordered[-1],
        "absolute_peak_error_p95": ordered_absolute[
            max(0, int(np.ceil(0.95 * len(ordered_absolute))) - 1)
        ],
        "quiet_target_examples": len(quiet_prediction_peaks),
        "quiet_prediction_peak_maximum": max(quiet_prediction_peaks, default=0.0),
    }


def train_rat_adapter(
    corpus: Path,
    output: Path,
    *,
    base_path: Path = DRIVE_CHECKPOINT,
    epochs: int = 8,
    batch_size: int = 32,
    frames: int = 2_048,
    warmup: int = 1_024,
    hidden_size: int = 32,
    maximum_train_files: int | None = None,
    maximum_eval_files: int | None = None,
) -> dict:
    if (output / "rat-device-adapter.pt").exists() or (output / "metrics.json").exists():
        raise ValueError(f"ASRNN RAT output already exists: {output}")
    audit = audit_asrnn(corpus, scan_audio=True)
    official_train = rat_files(corpus, "train")
    official_eval = rat_files(corpus, "eval")
    fit_paths, calibrate_paths = _partition(official_train)
    fit_paths = _bounded(fit_paths, maximum_train_files)
    eval_paths = _bounded(official_eval, maximum_eval_files)
    torch.manual_seed(20260830)
    np.random.seed(20260830)
    target = device()
    base = _load_drive(base_path).to(target).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    if warmup < 0 or frames <= 0:
        raise ValueError("warmup must be non-negative and TBPTT frames must be positive")
    train_loader = DataLoader(RatClips(fit_paths), batch_size=batch_size, shuffle=True)
    calibrate_loader = DataLoader(RatClips(calibrate_paths), batch_size=batch_size)
    eval_loader = DataLoader(RatClips(eval_paths), batch_size=batch_size)
    adapter = DriveDeviceAdapter(hidden_size=hidden_size).to(target)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1.0e-3)
    best_score = float("inf")
    best_state = None
    history = []
    for epoch in range(epochs):
        adapter.train()
        losses = []
        for batch in train_loader:
            dry = batch["dry"].to(target)
            wet = batch["wet"].to(target)
            controls = batch["controls"].to(target)
            base_state = None
            adapter_state = None
            if warmup:
                with torch.no_grad():
                    generic, base_state = base(dry[:, :warmup], controls, base_state)
                    _, adapter_state = adapter(
                        dry[:, :warmup], generic, controls, adapter_state
                    )
            for start in range(warmup, dry.shape[1], frames):
                stop = min(start + frames, dry.shape[1])
                with torch.no_grad():
                    generic, base_state = base(
                        dry[:, start:stop], controls, base_state
                    )
                prediction, adapter_state = adapter(
                    dry[:, start:stop], generic, controls, adapter_state
                )
                loss, _ = _rat_loss(prediction, wet[:, start:stop])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                base_state = _detach_state(base_state)
                adapter_state = _detach_state(adapter_state)
                losses.append(float(loss.detach()))
        calibration = _evaluate(base, adapter, calibrate_loader, target, frames, warmup)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "calibrate_model_esr": calibration["mean_model_esr"],
            "calibrate_improvement": calibration["mean_relative_improvement"],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if calibration["mean_model_esr"] < best_score:
            best_score = calibration["mean_model_esr"]
            best_state = {
                name: value.detach().cpu().clone() for name, value in adapter.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("ASRNN RAT training produced no checkpoint")
    adapter.load_state_dict(best_state)
    evaluation = _evaluate(base, adapter, eval_loader, target, frames, warmup)
    runtime = _stream_parity(adapter.cpu().eval())
    accepted = bool(
        evaluation["mean_relative_improvement"] >= 0.20
        and evaluation["peak_ratio_p95"] <= 1.35
        and evaluation["worst_peak_ratio"] <= 1.75
        and evaluation["absolute_peak_error_p95"] <= 0.02
        and evaluation["quiet_prediction_peak_maximum"] <= 5.0e-4
        and runtime["max_absolute_error"] <= 2.0e-6
        and runtime["silence_max_absolute_output"] == 0.0
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "rat-device-adapter.pt"
    base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
    torch.save(
        {
            "schema": 1,
            "sample_rate": 48_000,
            "hidden_size": adapter.hidden_size,
            "state_dict": best_state,
            "base_checkpoint_sha256": base_hash,
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
        "dataset": {
            "record_url": RECORD_URL,
            "license": LICENSE,
            "device": "ProCo RAT",
            "official_eval_source_disjoint": True,
            "audit": audit,
        },
        "partitions": {
            "fit_files": len(fit_paths),
            "calibrate_files": len(calibrate_paths),
            "official_eval_files": len(eval_paths),
            "calibrate_from_official_train": True,
        },
        "training_geometry": {
            "clip_frames": 48_000,
            "warmup_frames": warmup,
            "tbptt_frames": frames,
            "hidden_size": hidden_size,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "base_checkpoint_sha256": base_hash,
        "control_mapping": {
            "gain": "RAT distortion / 100",
            "tone": "1 - RAT filter / 100",
            "level": "RAT volume / 100",
        },
        "official_eval": evaluation,
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
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--base", type=Path, default=DRIVE_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frames", type=int, default=2_048)
    parser.add_argument("--warmup", type=int, default=1_024)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--maximum-train-files", type=int)
    parser.add_argument("--maximum-eval-files", type=int)
    args = parser.parse_args()
    report = train_rat_adapter(
        args.corpus,
        args.output,
        base_path=args.base,
        epochs=args.epochs,
        batch_size=args.batch_size,
        frames=args.frames,
        warmup=args.warmup,
        hidden_size=args.hidden_size,
        maximum_train_files=args.maximum_train_files,
        maximum_eval_files=args.maximum_eval_files,
    )
    print(json.dumps({"accepted": report["accepted"], "official_eval": report["official_eval"]}, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
