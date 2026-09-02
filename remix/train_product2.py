#!/usr/bin/env python3
"""Train product-safe stateful inverse experts after v2 data admission."""

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

from .inverse2 import InverseExpert, TRAINABLE_MECHANISMS, inverse_loss
from .inverse3 import NonlinearInverseV3
from .product2 import ProductPairsV2
from .quality2 import summarize


SEED = 20260903


def _collate(rows: list[dict]) -> dict:
    starts = {int(row["target_start"]) for row in rows}
    if len(starts) != 1:
        raise ValueError("one inverse2 batch must share target geometry")
    return {
        "wet": torch.stack([row["wet"] for row in rows]),
        "clean": torch.stack([row["clean"] for row in rows]),
        "controls": torch.stack([row["controls"] for row in rows]),
        "inverse_log_gain": torch.stack([row["inverse_log_gain"] for row in rows]),
        "target_start": starts.pop(),
    }


def _mean_loss(
    model: InverseExpert,
    dataset: ProductPairsV2,
    batch_size: int,
    device: torch.device,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total = 0.0
    examples = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {
                **batch,
                "wet": batch["wet"].to(device),
                "clean": batch["clean"].to(device),
                "controls": batch["controls"].to(device),
                "inverse_log_gain": batch["inverse_log_gain"].to(device),
            }
            restored, uncertainty, _ = model(batch["wet"], batch["controls"])
            loss, _ = inverse_loss(
                restored,
                uncertainty,
                batch["clean"],
                batch["target_start"],
                wet=batch["wet"] if model.mechanism == "dynamics" else None,
                inverse_log_gain=batch["inverse_log_gain"] if model.mechanism == "dynamics" else None,
            )
            count = len(batch["wet"])
            total += float(loss) * count
            examples += count
    return total / max(examples, 1)


def _stratum(mechanism: str, values: dict) -> str:
    if mechanism == "nonlinear":
        return str(values["shape"])
    release = float(values["release_ms"])
    return "short-release" if release < 120.0 else "medium-release" if release < 300.0 else "long-release"


def _quality(model: InverseExpert, dataset: ProductPairsV2) -> dict:
    aggregate = ([], [], [])
    strata = defaultdict(lambda: ([], [], []))
    sources = defaultdict(lambda: ([], [], []))
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            restored, _, _ = model(row["wet"].unsqueeze(0), row["controls"].unsqueeze(0))
            start = int(row["target_start"])
            wet = row["wet"][start:].numpy()
            candidate = restored[0, start:].numpy().astype(np.float32)
            clean = row["clean"][start:].numpy()
            for collection in (aggregate, strata[_stratum(model.mechanism, row["control_values"])], sources[row["source_id"]]):
                collection[0].append(wet)
                collection[1].append(candidate)
                collection[2].append(clean)
    report = summarize(model.mechanism, *aggregate)
    report["strata"] = {name: summarize(model.mechanism, *values) for name, values in sorted(strata.items())}
    report["sources"] = {name: summarize(model.mechanism, *values) for name, values in sorted(sources.items())}
    report["all_strata_accepted"] = bool(report["strata"] and all(item["accepted"] for item in report["strata"].values()))
    report["all_sources_accepted"] = bool(report["sources"] and all(item["accepted"] for item in report["sources"].values()))
    return report


def _runtime(model: InverseExpert, frames: int = 48_000) -> dict:
    wet = torch.zeros(1, frames)
    controls = torch.zeros(1, 5)
    model.eval()
    with torch.inference_mode():
        model(wet, controls)
        started = time.perf_counter()
        repeats = 3
        for _ in range(repeats):
            model(wet, controls)
        elapsed = (time.perf_counter() - started) / repeats
    return {
        "frames": frames,
        "mean_seconds": elapsed,
        "realtime_factor": elapsed / (frames / 48_000.0),
        "ordinary_cpu": True,
        "single_expert_only": True,
        "audio_callback": False,
    }


def train_one(
    workspace: Path,
    output: Path,
    mechanism: str,
    *,
    epochs: int,
    train_samples: int,
    calibration_samples: int,
    development_samples: int,
    target_frames: int,
    batch_size: int,
    hidden_size: int,
    layers: int,
    quick: bool,
    device: torch.device,
    nonlinear_version: int = 2,
) -> dict:
    fit = ProductPairsV2(workspace, mechanism, "fit", train_samples, target_frames, SEED + 1)
    calibration = ProductPairsV2(workspace, mechanism, "calibration", calibration_samples, target_frames, SEED + 2)
    development = ProductPairsV2(workspace, mechanism, "development", development_samples, target_frames, SEED + 3)
    model = (
        NonlinearInverseV3(hidden_size=hidden_size, layers=layers)
        if mechanism == "nonlinear" and nonlinear_version == 3
        else InverseExpert(mechanism, hidden_size=hidden_size, layers=layers)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4.0e-4, weight_decay=1.0e-5)
    loader = DataLoader(
        fit,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        collate_fn=_collate,
    )
    initial_loss = _mean_loss(model, calibration, batch_size, device)
    history = [{"epoch": 0, "train": None, "calibration_loss": initial_loss}]
    best_loss = initial_loss
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    print(json.dumps({"mechanism": mechanism, "epoch": 0, "calibration_loss": initial_loss}), flush=True)
    for epoch in range(epochs):
        model.train()
        totals = {
            "loss": 0.0,
            "waveform": 0.0,
            "transient": 0.0,
            "attack": 0.0,
            "preemphasis": 0.0,
            "spectral": 0.0,
            "envelope": 0.0,
            "uncertainty": 0.0,
            "gain": 0.0,
        }
        examples = 0
        for batch in loader:
            batch = {
                **batch,
                "wet": batch["wet"].to(device),
                "clean": batch["clean"].to(device),
                "controls": batch["controls"].to(device),
                "inverse_log_gain": batch["inverse_log_gain"].to(device),
            }
            optimizer.zero_grad(set_to_none=True)
            target_start = batch["target_start"]
            if mechanism == "dynamics" and target_start:
                with torch.no_grad():
                    _, _, warm_state = model(batch["wet"][:, :target_start], batch["controls"])
                restored, uncertainty, _ = model(
                    batch["wet"][:, target_start:], batch["controls"], warm_state.detach()
                )
                clean_for_loss = batch["clean"][:, target_start:]
                loss_start = 0
            else:
                restored, uncertainty, _ = model(batch["wet"], batch["controls"])
                clean_for_loss = batch["clean"]
                loss_start = target_start
            loss, parts = inverse_loss(
                restored,
                uncertainty,
                clean_for_loss,
                loss_start,
                wet=(batch["wet"][:, target_start:] if mechanism == "dynamics" and target_start else batch["wet"])
                if mechanism == "dynamics" else None,
                inverse_log_gain=(
                    batch["inverse_log_gain"][:, target_start:]
                    if mechanism == "dynamics" and target_start
                    else batch["inverse_log_gain"]
                ) if mechanism == "dynamics" else None,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            count = len(batch["wet"])
            totals["loss"] += float(loss.detach()) * count
            for name, value in parts.items():
                totals[name] += value * count
            examples += count
        calibration_loss = _mean_loss(model, calibration, batch_size, device)
        history.append({
            "epoch": epoch + 1,
            "train": {name: value / max(examples, 1) for name, value in totals.items()},
            "calibration_loss": calibration_loss,
        })
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        print(json.dumps({"mechanism": mechanism, "epoch": epoch + 1, "calibration_loss": calibration_loss}), flush=True)
    if best_state is None:
        raise RuntimeError("inverse2 training produced no checkpoint")
    model = model.cpu()
    model.load_state_dict(best_state)
    quality = _quality(model, development)
    runtime = _runtime(model)
    accepted = bool(
        not quick
        and quality["accepted"]
        and quality["all_strata_accepted"]
        and quality["all_sources_accepted"]
        and runtime["realtime_factor"] <= 0.5
    )
    target = output / mechanism
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "model.pt"
    torch.save({
        "schema": 2,
        "sample_rate": 48_000,
        "architecture": model.manifest(),
        "state_dict": model.state_dict(),
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 2,
        "status": "accepted" if accepted else "diagnostic-not-promoted",
        "accepted": accepted,
        "quick": quick,
        "mechanism": mechanism,
        "model": {**model.manifest(), "checkpoint": str(checkpoint), "sha256": digest},
        "training": {
            "epochs": epochs,
            "target_frames": target_frames,
            "history_frames": fit.history_frames,
            "fit_samples_per_epoch": train_samples,
            "calibration_samples": calibration_samples,
            "development_samples": development_samples,
            "selected_calibration_loss": best_loss,
            "history": history,
            "accelerator": device.type,
            "nonlinear_version": nonlinear_version if mechanism == "nonlinear" else None,
        },
        "provenance": {
            "product_sources_only": True,
            "realized_source_counts": {
                "fit": fit.realized_source_counts(),
                "calibration": calibration.realized_source_counts(),
                "development": development.realized_source_counts(),
            },
            "authorized_source_ids": fit.authorization["sources"],
            "required_attribution": fit.authorization["required_attribution"],
            "research_source_ids": [],
            "research_data_isolated": True,
        },
        "development": quality,
        "runtime": runtime,
        "quality": {
            "metric_schema": 2,
            "generated_audio_written": False,
            "source_audio_modified": False,
            "physical_audio_devices_used": False,
            "locked_final_audio_opened": False,
        },
    }
    (target / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/product2"))
    parser.add_argument("--mechanisms", default=",".join(TRAINABLE_MECHANISMS))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train-samples", type=int, default=600)
    parser.add_argument("--calibration-samples", type=int, default=120)
    parser.add_argument("--development-samples", type=int, default=120)
    parser.add_argument("--target-frames", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=20)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--nonlinear-version", type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    if args.quick:
        args.epochs = 3
        args.train_samples = 60
        args.calibration_samples = 24
        args.development_samples = 24
        args.target_frames = 4096
        args.batch_size = 4
        args.hidden_size = 12
        args.layers = 2
    mechanisms = tuple(item.strip() for item in args.mechanisms.split(",") if item.strip())
    if not mechanisms or any(item not in TRAINABLE_MECHANISMS for item in mechanisms):
        raise ValueError(f"mechanisms must be selected from {TRAINABLE_MECHANISMS}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))
    requested_device = args.device
    if requested_device == "auto":
        requested_device = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this execution environment")
    device = torch.device(requested_device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for mechanism in mechanisms:
        reports.append(train_one(
            args.workspace.resolve(), output, mechanism,
            epochs=args.epochs,
            train_samples=args.train_samples,
            calibration_samples=args.calibration_samples,
            development_samples=args.development_samples,
            target_frames=args.target_frames,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            layers=args.layers,
            quick=args.quick,
            device=device,
            nonlinear_version=args.nonlinear_version,
        ))
    summary = {
        "schema": 2,
        "status": "accepted" if reports and all(report["accepted"] for report in reports) else "diagnostic-not-promoted",
        "mechanisms": [report["mechanism"] for report in reports],
        "reports": {report["mechanism"]: f"{report['mechanism']}/metrics.json" for report in reports},
        "product_sources_only": True,
        "research_data_isolated": True,
        "physical_audio_devices_used": False,
        "locked_final_audio_opened": False,
        "training_accelerator": device.type,
        "nonlinear_version": args.nonlinear_version if "nonlinear" in mechanisms else None,
        "ambience_model": None,
        "echo_runtime": "analytic repository-owned inverse; separately audited",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
