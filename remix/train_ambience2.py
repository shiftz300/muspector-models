#!/usr/bin/env python3
"""Train the independent long-tail product ambience expert."""

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

from .ambience2 import AmbiencePairsV2
from .ambience_model2 import AmbienceExpert, ambience_loss
from .quality2 import summarize as summarize_universal
from .reverb_quality import summarize as summarize_tail


SEED = 20260907


def _collate(rows: list[dict]) -> dict:
    starts = {int(row["target_start"]) for row in rows}
    if len(starts) != 1:
        raise ValueError("one ambience2 batch must share target geometry")
    return {
        "wet": torch.stack([row["wet"] for row in rows]),
        "clean": torch.stack([row["clean"] for row in rows]),
        "controls": torch.stack([row["controls"] for row in rows]),
        "late_base": torch.stack([row["late_base"] for row in rows]),
        "target_start": starts.pop(),
    }


def _device_batch(batch: dict, device: torch.device) -> dict:
    return {
        **batch,
        "wet": batch["wet"].to(device),
        "clean": batch["clean"].to(device),
        "controls": batch["controls"].to(device),
        "late_base": batch["late_base"].to(device),
    }


def _mean_loss(model: AmbienceExpert, dataset: AmbiencePairsV2, batch_size: int, device: torch.device) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total = 0.0
    examples = 0
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            batch = _device_batch(raw, device)
            restored, uncertainty = model(batch["wet"], batch["controls"], batch["late_base"])
            loss, _ = ambience_loss(
                restored, uncertainty, batch["wet"], batch["clean"], batch["target_start"]
            )
            count = len(batch["wet"])
            total += float(loss) * count
            examples += count
    return total / max(examples, 1)


def _decay_stratum(seconds: float) -> str:
    return "short-tail" if seconds < 1.85 else "long-tail"


def _summaries(values: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]) -> dict:
    universal = summarize_universal("ambience", *values)
    try:
        tail = summarize_tail(*values)
    except ValueError:
        tail = {"accepted": False, "reason": "no eligible temporal tails"}
    return {
        "universal": universal,
        "tail": tail,
        "accepted": bool(universal["accepted"] and tail["accepted"]),
    }


def _quality(model: AmbienceExpert, dataset: AmbiencePairsV2) -> dict:
    aggregate = ([], [], [])
    sources = defaultdict(lambda: ([], [], []))
    decays = defaultdict(lambda: ([], [], []))
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            restored, _ = model(
                row["wet"].unsqueeze(0), row["controls"].unsqueeze(0), row["late_base"].unsqueeze(0)
            )
            start = int(row["target_start"])
            wet = row["wet"][start:].numpy()
            candidate = restored[0, start:].numpy().astype(np.float32)
            clean = row["clean"][start:].numpy()
            stratum = _decay_stratum(float(row["control_values"]["decay_p999_seconds"]))
            for collection in (aggregate, sources[row["source_id"]], decays[stratum]):
                collection[0].append(wet)
                collection[1].append(candidate)
                collection[2].append(clean)
    result = _summaries(aggregate)
    result["sources"] = {name: _summaries(values) for name, values in sorted(sources.items())}
    result["decay_strata"] = {name: _summaries(values) for name, values in sorted(decays.items())}
    result["all_sources_accepted"] = bool(
        result["sources"] and all(row["accepted"] for row in result["sources"].values())
    )
    result["all_decay_strata_accepted"] = bool(
        result["decay_strata"] and all(row["accepted"] for row in result["decay_strata"].values())
    )
    return result


def _runtime(model: AmbienceExpert, frames: int) -> dict:
    wet = torch.zeros(1, frames)
    controls = torch.zeros(1, 3)
    late_base = wet.clone()
    model.eval()
    with torch.inference_mode():
        model(wet, controls, late_base)
        started = time.perf_counter()
        repeats = 2
        for _ in range(repeats):
            model(wet, controls, late_base)
        elapsed = (time.perf_counter() - started) / repeats
    return {
        "frames": frames,
        "mean_seconds": elapsed,
        "realtime_factor": elapsed / (frames / 48_000.0),
        "ordinary_cpu": True,
        "bounded_window": True,
        "audio_callback": False,
    }


def train(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    fit = AmbiencePairsV2(workspace, "fit", args.train_samples, args.target_frames, SEED + 1)
    calibration = AmbiencePairsV2(
        workspace, "calibration", args.calibration_samples, args.target_frames, SEED + 2
    )
    development = AmbiencePairsV2(
        workspace, "development", args.development_samples, args.target_frames, SEED + 3
    )
    device = torch.device(args.device)
    model = AmbienceExpert(args.channels, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5)
    loader = DataLoader(
        fit,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        collate_fn=_collate,
    )
    initial_loss = _mean_loss(model, calibration, args.batch_size, device)
    history = [{"epoch": 0, "train": None, "calibration_loss": initial_loss}]
    best_loss = initial_loss
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    print(json.dumps({"mechanism": "ambience", "epoch": 0, "calibration_loss": initial_loss}), flush=True)
    for epoch in range(args.epochs):
        model.train()
        totals = defaultdict(float)
        examples = 0
        for raw in loader:
            batch = _device_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            restored, uncertainty = model(batch["wet"], batch["controls"], batch["late_base"])
            loss, parts = ambience_loss(
                restored, uncertainty, batch["wet"], batch["clean"], batch["target_start"]
            )
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
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        print(json.dumps({"mechanism": "ambience", "epoch": epoch + 1, "calibration_loss": calibration_loss}), flush=True)
    model = model.cpu()
    model.load_state_dict(best_state)
    quality = _quality(model, development)
    runtime = _runtime(model, fit.total_frames)
    accepted = bool(
        not args.quick
        and quality["accepted"]
        and quality["all_sources_accepted"]
        and quality["all_decay_strata_accepted"]
        and runtime["realtime_factor"] <= 0.5
    )
    target = output / "ambience"
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "model.pt"
    torch.save({
        "schema": 1,
        "sample_rate": 48_000,
        "architecture": model.manifest(),
        "state_dict": model.state_dict(),
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 1,
        "status": "accepted" if accepted else "diagnostic-not-promoted",
        "accepted": accepted,
        "quick": args.quick,
        "mechanism": "ambience",
        "model": {**model.manifest(), "checkpoint": str(checkpoint), "sha256": digest},
        "training": {
            "epochs": args.epochs,
            "target_frames": args.target_frames,
            "history_frames": fit.history_frames,
            "fit_samples_per_epoch": args.train_samples,
            "calibration_samples": args.calibration_samples,
            "development_samples": args.development_samples,
            "selected_calibration_loss": best_loss,
            "history": history,
            "accelerator": device.type,
        },
        "provenance": {
            "product_sources_only": True,
            "realized_source_counts": {
                "fit": fit.realized_source_counts(),
                "calibration": calibration.realized_source_counts(),
                "development": development.realized_source_counts(),
            },
            "realized_rir_counts": {
                "fit": fit.realized_rir_counts(),
                "calibration": calibration.realized_rir_counts(),
                "development": development.realized_rir_counts(),
            },
            "authorized_source_ids": fit.authorization["sources"],
            "required_attribution": fit.authorization["required_attribution"],
            "research_source_ids": [],
            "generated_audio_written": False,
            "physical_audio_devices_used": False,
            "locked_final_audio_opened": False,
        },
        "development": quality,
        "runtime": runtime,
    }
    (target / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/product2"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=120)
    parser.add_argument("--calibration-samples", type=int, default=30)
    parser.add_argument("--development-samples", type=int, default=48)
    parser.add_argument("--target-frames", type=int, default=16_384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    args = parser.parse_args()
    if args.quick:
        args.epochs = 2
        args.train_samples = 24
        args.calibration_samples = 8
        args.development_samples = 12
        args.batch_size = 1
        args.channels = 6
        args.depth = 8
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))
    report = train(args)
    print(json.dumps({
        "mechanism": "ambience",
        "status": report["status"],
        "checkpoint": report["model"]["checkpoint"],
        "metrics": str(args.output.resolve() / "ambience/metrics.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
