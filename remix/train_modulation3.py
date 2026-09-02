#!/usr/bin/env python3
"""Train and gate the hidden-trajectory Tremolo inverse on product-safe pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .inverse2 import _pre_emphasis, _spectral_loss
from .modulation3 import TremoloPairsV3
from .modulation_model3 import TremoloInverseV3
from .quality2 import summarize


SEED = 20260925


def _collate(rows: list[dict]) -> dict:
    starts = {int(row["target_start"]) for row in rows}
    ends = {int(row["target_end"]) for row in rows}
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("one Tremolo batch must share target geometry")
    return {
        "wet": torch.stack([row["wet"] for row in rows]),
        "clean": torch.stack([row["clean"] for row in rows]),
        "controls": torch.stack([row["controls"] for row in rows]),
        "inverse_log_gain": torch.stack([row["inverse_log_gain"] for row in rows]),
        "target_start": starts.pop(),
        "target_end": ends.pop(),
    }


def _device_batch(batch: dict, device: torch.device) -> dict:
    return {
        **batch,
        "wet": batch["wet"].to(device),
        "clean": batch["clean"].to(device),
        "controls": batch["controls"].to(device),
        "inverse_log_gain": batch["inverse_log_gain"].to(device),
    }


def _loss(
    restored: torch.Tensor,
    uncertainty: torch.Tensor,
    predicted_gain: torch.Tensor,
    batch: dict,
) -> tuple[torch.Tensor, dict]:
    start, end = batch["target_start"], batch["target_end"]
    candidate = restored[:, start:end]
    clean = batch["clean"][:, start:end]
    target_gain = batch["inverse_log_gain"][:, start:end]
    predicted_gain = predicted_gain[:, start:end]
    uncertainty = uncertainty[:, start:end]
    scale = clean.abs().mean().clamp_min(1.0e-5)
    trajectory_scale = target_gain.amax(dim=1).clamp_min(0.05)
    trajectory = ((predicted_gain - target_gain).abs().mean(dim=1) / trajectory_scale).mean()
    waveform = F.l1_loss(candidate, clean) / scale
    emphasized = F.l1_loss(_pre_emphasis(candidate), _pre_emphasis(clean)) / _pre_emphasis(clean).abs().mean().clamp_min(1.0e-5)
    envelope_clean = F.avg_pool1d(clean.square()[:, None], 480, 120, padding=240).sqrt()
    envelope_candidate = F.avg_pool1d(candidate.square()[:, None], 480, 120, padding=240).sqrt()
    envelope = F.l1_loss(envelope_candidate, envelope_clean) / envelope_clean.mean().clamp_min(1.0e-5)
    spectral = _spectral_loss(candidate, clean)
    uncertainty_target = (candidate - clean).abs().detach()
    uncertainty_loss = F.l1_loss(uncertainty, uncertainty_target) / scale
    loss = 2.0 * trajectory + waveform + 0.25 * emphasized + 0.35 * envelope + 0.10 * spectral + 0.03 * uncertainty_loss
    return loss, {
        "trajectory": float(trajectory.detach()),
        "waveform": float(waveform.detach()),
        "preemphasis": float(emphasized.detach()),
        "envelope": float(envelope.detach()),
        "spectral": float(spectral.detach()),
        "uncertainty": float(uncertainty_loss.detach()),
    }


def _mean_loss(model: TremoloInverseV3, dataset: TremoloPairsV3, batch_size: int, device: torch.device) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total = 0.0
    examples = 0
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            batch = _device_batch(raw, device)
            restored, uncertainty, gain = model(batch["wet"], batch["controls"])
            loss, _ = _loss(restored, uncertainty, gain, batch)
            count = len(batch["wet"])
            total += float(loss) * count
            examples += count
    return total / max(examples, 1)


def _trajectory_row(predicted: np.ndarray, target: np.ndarray) -> dict:
    scale = max(float(np.max(target)), 0.05)
    nmae = float(np.mean(np.abs(predicted - target)) / scale)
    left = predicted - float(np.mean(predicted))
    right = target - float(np.mean(target))
    correlation = float(np.sum(left * right) / max(np.sqrt(np.sum(left * left) * np.sum(right * right)), 1.0e-10))
    return {"normalized_mae": nmae, "correlation": correlation}


def _summarize_rows(audio: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]], trajectories: list[dict]) -> dict:
    report = summarize("modulation", *audio)
    nmae = [row["normalized_mae"] for row in trajectories]
    correlation = [row["correlation"] for row in trajectories]
    report["trajectory"] = {
        "examples": len(trajectories),
        "median_normalized_mae": float(np.median(nmae)),
        "p90_normalized_mae": float(np.quantile(nmae, 0.90)),
        "median_correlation": float(np.median(correlation)),
        "correlation_ge_0_75_fraction": float(np.mean(np.asarray(correlation) >= 0.75)),
    }
    report["trajectory_gates"] = {
        "median_normalized_mae": report["trajectory"]["median_normalized_mae"] <= 0.20,
        "p90_normalized_mae": report["trajectory"]["p90_normalized_mae"] <= 0.35,
        "median_correlation": report["trajectory"]["median_correlation"] >= 0.80,
        "coverage": report["trajectory"]["correlation_ge_0_75_fraction"] >= 0.70,
    }
    report["accepted"] = bool(report["accepted"] and all(report["trajectory_gates"].values()))
    return report


def _quality(model: TremoloInverseV3, dataset: TremoloPairsV3) -> dict:
    aggregate = ([], [], [])
    aggregate_trajectories = []
    sources = defaultdict(lambda: (([], [], []), []))
    strata = defaultdict(lambda: (([], [], []), []))
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            restored, _, predicted = model(row["wet"].unsqueeze(0), row["controls"].unsqueeze(0))
            start, end = int(row["target_start"]), int(row["target_end"])
            audio = (
                row["wet"][start:end].numpy(),
                restored[0, start:end].numpy().astype(np.float32),
                row["clean"][start:end].numpy(),
            )
            trajectory = _trajectory_row(
                predicted[0, start:end].numpy(), row["inverse_log_gain"][start:end].numpy()
            )
            rate = float(row["control_values"]["rate_hz"])
            rate_stratum = "slow-rate" if rate < 3.2 else "fast-rate"
            waveform = row["control_values"]["waveform"]
            for audio_rows, trajectory_rows in (
                (aggregate, aggregate_trajectories),
                sources[row["source_id"]],
                strata[f"{waveform}-{rate_stratum}"],
            ):
                for target, value in zip(audio_rows, audio, strict=True):
                    target.append(value)
                trajectory_rows.append(trajectory)
    report = _summarize_rows(aggregate, aggregate_trajectories)
    report["sources"] = {name: _summarize_rows(*rows) for name, rows in sorted(sources.items())}
    report["strata"] = {name: _summarize_rows(*rows) for name, rows in sorted(strata.items())}
    report["all_sources_accepted"] = bool(report["sources"] and all(row["accepted"] for row in report["sources"].values()))
    report["all_strata_accepted"] = bool(report["strata"] and all(row["accepted"] for row in report["strata"].values()))
    report["accepted"] = bool(report["accepted"] and report["all_sources_accepted"] and report["all_strata_accepted"])
    return report


def _runtime(model: TremoloInverseV3, frames: int = 192_000) -> dict:
    wet = torch.zeros(1, frames)
    controls = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0]])
    model.eval()
    with torch.inference_mode():
        model(wet, controls)
        started = time.perf_counter()
        repeats = 3
        for _ in range(repeats):
            model(wet, controls)
        seconds = (time.perf_counter() - started) / repeats
    return {
        "frames": frames,
        "mean_seconds": seconds,
        "realtime_factor": seconds / (frames / 48_000.0),
        "ordinary_cpu": True,
        "bounded_context": True,
        "audio_callback": False,
    }


def train(args: argparse.Namespace) -> dict:
    workspace, output = args.workspace.resolve(), args.output.resolve()
    fit = TremoloPairsV3(workspace, "fit", args.train_samples, args.target_frames, SEED + 1)
    calibration = TremoloPairsV3(workspace, "calibration", args.calibration_samples, args.target_frames, SEED + 2)
    development = TremoloPairsV3(workspace, "development", args.development_samples, args.target_frames, SEED + 3)
    device = torch.device(args.device)
    model = TremoloInverseV3(args.channels, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5)
    loader = DataLoader(
        fit,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        collate_fn=_collate,
    )
    initial = _mean_loss(model, calibration, args.batch_size, device)
    history = [{"epoch": 0, "train": None, "calibration_loss": initial}]
    best_loss = initial
    best_epoch = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    print(json.dumps({"mechanism": "modulation", "family": "tremolo", "epoch": 0, "calibration_loss": initial}), flush=True)
    for epoch in range(args.epochs):
        model.train()
        totals = defaultdict(float)
        examples = 0
        for raw in loader:
            batch = _device_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            restored, uncertainty, gain = model(batch["wet"], batch["controls"])
            loss, parts = _loss(restored, uncertainty, gain, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            count = len(batch["wet"])
            totals["loss"] += float(loss.detach()) * count
            for name, value in parts.items():
                totals[name] += value * count
            examples += count
        calibration_loss = _mean_loss(model, calibration, args.batch_size, device)
        history.append({
            "epoch": epoch + 1,
            "train": {name: value / max(examples, 1) for name, value in sorted(totals.items())},
            "calibration_loss": calibration_loss,
        })
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_epoch = epoch + 1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(json.dumps({"mechanism": "modulation", "family": "tremolo", "epoch": epoch + 1, "calibration_loss": calibration_loss}), flush=True)
    model = model.cpu()
    model.load_state_dict(best_state)
    calibration_report = _quality(model, calibration)
    development_report = _quality(model, development)
    runtime = _runtime(model)
    accepted = bool(
        not args.quick and calibration_report["accepted"] and development_report["accepted"]
        and runtime["realtime_factor"] <= 0.5
    )
    target = output / "tremolo"
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "model.pt"
    torch.save({
        "schema": 1,
        "sample_rate": 48_000,
        "architecture": model.manifest(),
        "context_frames_each_side": fit.context_frames,
        "state_dict": model.state_dict(),
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 1,
        "status": "accepted" if accepted else "diagnostic-not-promoted",
        "accepted": accepted,
        "quick": args.quick,
        "mechanism": "modulation",
        "family": "tremolo",
        "model": {**model.manifest(), "checkpoint": str(checkpoint), "sha256": digest},
        "training": {
            "seed": SEED,
            "epochs": args.epochs,
            "selected_epoch": best_epoch,
            "selected_calibration_loss": best_loss,
            "fit_samples_per_epoch": args.train_samples,
            "calibration_samples": args.calibration_samples,
            "development_samples": args.development_samples,
            "target_frames": args.target_frames,
            "context_frames_each_side": fit.context_frames,
            "history": history,
            "accelerator": device.type,
        },
        "calibration": calibration_report,
        "development": development_report,
        "runtime": runtime,
        "provenance": {
            "product_sources_only": True,
            "generic_repository_owned_dsp": True,
            "named_physical_device_claim": False,
            "realized_source_counts": {
                "fit": fit.realized_source_counts(),
                "calibration": calibration.realized_source_counts(),
                "development": development.realized_source_counts(),
            },
            "authorized_source_ids": fit.authorization["sources"],
            "required_attribution": fit.authorization["required_attribution"],
            "research_source_ids": [],
            "generated_audio_written": False,
            "physical_audio_devices_used": False,
            "locked_final_audio_opened": False,
        },
    }
    (target / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/product3-modulation"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train-samples", type=int, default=192)
    parser.add_argument("--calibration-samples", type=int, default=48)
    parser.add_argument("--development-samples", type=int, default=72)
    parser.add_argument("--target-frames", type=int, default=96_000)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--channels", type=int, default=20)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    args = parser.parse_args()
    if args.quick:
        args.epochs = 3
        args.train_samples = 36
        args.calibration_samples = 12
        args.development_samples = 16
        args.batch_size = 2
        args.channels = 10
        args.depth = 7
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))
    report = train(args)
    print(json.dumps({
        "mechanism": "modulation",
        "family": "tremolo",
        "status": report["status"],
        "selected_epoch": report["training"]["selected_epoch"],
        "checkpoint": report["model"]["checkpoint"],
        "metrics": str(args.output.resolve() / "tremolo/metrics.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
