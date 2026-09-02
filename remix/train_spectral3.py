#!/usr/bin/env python3
"""Train and gate the independent product-safe Spectral/EQ v3 expert."""

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
from torch.utils.data import DataLoader

from .quality2 import summarize
from .spectral3 import SpectralPairsV3
from .spectral_model3 import SpectralInverseV3


SEED = 20260924


def _collate(rows: list[dict]) -> dict:
    starts = {int(row["target_start"]) for row in rows}
    ends = {int(row["target_end"]) for row in rows}
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("one spectral batch must share target geometry")
    return {
        "wet": torch.stack([row["wet"] for row in rows]),
        "clean": torch.stack([row["clean"] for row in rows]),
        "controls": torch.stack([row["controls"] for row in rows]),
        "target_start": starts.pop(),
        "target_end": ends.pop(),
    }


def _loss(restored: torch.Tensor, uncertainty: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict]:
    start, end = batch["target_start"], batch["target_end"]
    candidate = restored[:, start:end]
    clean = batch["clean"][:, start:end]
    wet = batch["wet"][:, start:end]
    scale = clean.square().mean(dim=1).sqrt().clamp_min(1.0e-4)
    waveform = ((candidate - clean).abs().mean(dim=1) / scale).mean()
    candidate_spectrum = torch.fft.rfft(candidate)
    clean_spectrum = torch.fft.rfft(clean)
    spectral = torch.mean(
        torch.abs(torch.log1p(candidate_spectrum.abs()) - torch.log1p(clean_spectrum.abs()))
    )
    correction = (((candidate - wet).abs().mean(dim=1) / scale).mean())
    confidence = uncertainty[:, start:end].mean()
    loss = waveform + 0.5 * spectral + 1.0e-5 * correction + 1.0e-4 * confidence
    return loss, {
        "waveform": float(waveform.detach()),
        "spectral": float(spectral.detach()),
        "correction": float(correction.detach()),
        "uncertainty": float(confidence.detach()),
    }


def _device_batch(batch: dict, device: torch.device) -> dict:
    return {
        **batch,
        "wet": batch["wet"].to(device),
        "clean": batch["clean"].to(device),
        "controls": batch["controls"].to(device),
    }


def _mean_loss(model: SpectralInverseV3, dataset: SpectralPairsV3, batch_size: int, device: torch.device) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total = 0.0
    examples = 0
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            batch = _device_batch(raw, device)
            restored, uncertainty = model(batch["wet"], batch["controls"])
            loss, _ = _loss(restored, uncertainty, batch)
            count = len(batch["wet"])
            total += float(loss) * count
            examples += count
    return total / max(examples, 1)


def _stratum(values: dict) -> str:
    gains = {
        "low-dominant": abs(float(values["low_gain_db"])),
        "mid-dominant": abs(float(values["mid_gain_db"])),
        "high-dominant": abs(float(values["high_gain_db"])),
    }
    return max(gains, key=gains.get)


def _quality(model: SpectralInverseV3, dataset: SpectralPairsV3) -> dict:
    aggregate = ([], [], [])
    sources = defaultdict(lambda: ([], [], []))
    strata = defaultdict(lambda: ([], [], []))
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            restored, _ = model(row["wet"].unsqueeze(0), row["controls"].unsqueeze(0))
            start, end = int(row["target_start"]), int(row["target_end"])
            values = (
                row["wet"][start:end].numpy(),
                restored[0, start:end].numpy().astype(np.float32),
                row["clean"][start:end].numpy(),
            )
            for collection in (aggregate, sources[row["source_id"]], strata[_stratum(row["control_values"])]):
                for target, value in zip(collection, values, strict=True):
                    target.append(value)
    report = summarize("spectral", *aggregate)
    report["sources"] = {name: summarize("spectral", *rows) for name, rows in sorted(sources.items())}
    report["strata"] = {name: summarize("spectral", *rows) for name, rows in sorted(strata.items())}
    report["all_sources_accepted"] = bool(report["sources"] and all(row["accepted"] for row in report["sources"].values()))
    report["all_strata_accepted"] = bool(report["strata"] and all(row["accepted"] for row in report["strata"].values()))
    report["accepted"] = bool(report["accepted"] and report["all_sources_accepted"] and report["all_strata_accepted"])
    return report


def _runtime(model: SpectralInverseV3, frames: int = 52_096) -> dict:
    wet = torch.zeros(1, frames)
    controls = torch.full((1, 5), 0.5)
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
    fit = SpectralPairsV3(workspace, "fit", args.train_samples, args.target_frames, SEED + 1)
    calibration = SpectralPairsV3(workspace, "calibration", args.calibration_samples, args.target_frames, SEED + 2)
    development = SpectralPairsV3(workspace, "development", args.development_samples, args.target_frames, SEED + 3)
    device = torch.device(args.device)
    model = SpectralInverseV3(args.channels, args.depth).to(device)
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
    print(json.dumps({"mechanism": "spectral", "epoch": 0, "calibration_loss": initial}), flush=True)
    for epoch in range(args.epochs):
        model.train()
        totals = defaultdict(float)
        examples = 0
        for raw in loader:
            batch = _device_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            restored, uncertainty = model(batch["wet"], batch["controls"])
            loss, parts = _loss(restored, uncertainty, batch)
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
        print(json.dumps({"mechanism": "spectral", "epoch": epoch + 1, "calibration_loss": calibration_loss}), flush=True)

    model = model.cpu()
    model.load_state_dict(best_state)
    calibration_report = _quality(model, calibration)
    development_report = _quality(model, development)
    runtime = _runtime(model)
    accepted = bool(
        not args.quick
        and calibration_report["accepted"]
        and development_report["accepted"]
        and runtime["realtime_factor"] <= 0.5
    )
    target = output / "spectral"
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "model.pt"
    torch.save({
        "schema": 1,
        "sample_rate": 48_000,
        "architecture": model.manifest(),
        "context_frames": fit.context_frames,
        "state_dict": model.state_dict(),
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 1,
        "status": "accepted" if accepted else "diagnostic-not-promoted",
        "accepted": accepted,
        "quick": args.quick,
        "mechanism": "spectral",
        "model": {**model.manifest(), "checkpoint": str(checkpoint), "sha256": digest},
        "training": {
            "seed": SEED,
            "epochs": args.epochs,
            "selected_epoch": best_epoch,
            "selected_calibration_loss": best_loss,
            "target_frames": args.target_frames,
            "context_frames_each_side": fit.context_frames,
            "fit_samples_per_epoch": args.train_samples,
            "calibration_samples": args.calibration_samples,
            "development_samples": args.development_samples,
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
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/product3-spectral"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=144)
    parser.add_argument("--calibration-samples", type=int, default=54)
    parser.add_argument("--development-samples", type=int, default=72)
    parser.add_argument("--target-frames", type=int, default=16_384)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--channels", type=int, default=10)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    args = parser.parse_args()
    if args.quick:
        args.epochs = 2
        args.train_samples = 24
        args.calibration_samples = 12
        args.development_samples = 18
        args.target_frames = 8192
        args.batch_size = 2
        args.channels = 6
        args.depth = 3
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))
    report = train(args)
    print(json.dumps({
        "mechanism": "spectral",
        "status": report["status"],
        "selected_epoch": report["training"]["selected_epoch"],
        "checkpoint": report["model"]["checkpoint"],
        "metrics": str(args.output.resolve() / "spectral/metrics.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
