#!/usr/bin/env python3
"""Train equal-budget product-safe restoration foundation experts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .foundation_model import Expert, MECHANISMS, restoration_loss
from .product_data import ProductPairs, audit as audit_product_data
from .restoration_quality import summarize


SEED = 20260902


def _collate(rows: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "wet": torch.stack([row["wet"] for row in rows]),
        "clean": torch.stack([row["clean"] for row in rows]),
    }


def _mean_loss(model: Expert, dataset: ProductPairs, batch_size: int) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total = 0.0
    count = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            restored = model(batch["wet"])
            loss, _ = restoration_loss(restored, batch["clean"])
            total += float(loss) * len(batch["wet"])
            count += len(batch["wet"])
    return total / max(count, 1)


def _quality(model: Expert, dataset: ProductPairs) -> dict:
    wet, restored, clean = [], [], []
    strata = {}
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            source = row["wet"].unsqueeze(0)
            candidate = model(source)[0].numpy().astype(np.float32)
            wet.append(row["wet"].numpy())
            restored.append(candidate)
            clean.append(row["clean"].numpy())
            stratum = str(row["controls"].get("kind", row["controls"].get("shape", model.mechanism)))
            strata.setdefault(stratum, ([], [], []))
            strata[stratum][0].append(wet[-1])
            strata[stratum][1].append(restored[-1])
            strata[stratum][2].append(clean[-1])
    report = summarize(wet, restored, clean)
    report["strata"] = {
        name: summarize(*values) for name, values in sorted(strata.items()) if len(values[0]) >= 2
    }
    return report


def _runtime(model: Expert, frames: int) -> dict:
    value = torch.zeros(1, frames)
    model.eval()
    with torch.inference_mode():
        model(value)
        started = time.perf_counter()
        repeats = 5
        for _ in range(repeats):
            model(value)
        elapsed = time.perf_counter() - started
    seconds = frames / 48_000.0
    return {
        "frames": frames,
        "repeats": repeats,
        "mean_seconds": elapsed / repeats,
        "realtime_factor": elapsed / repeats / seconds,
        "ordinary_cpu": True,
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
    frames: int,
    batch_size: int,
    channels: int,
    blocks: int,
    quick: bool,
) -> dict:
    fit = ProductPairs(workspace, mechanism, "fit", train_samples, frames, SEED + 1)
    calibration = ProductPairs(
        workspace, mechanism, "calibration", calibration_samples, frames, SEED + 2
    )
    development = ProductPairs(
        workspace, mechanism, "development", development_samples, frames, SEED + 3
    )
    model = Expert(mechanism, channels=channels, blocks=blocks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    loader = DataLoader(
        fit,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        collate_fn=_collate,
    )
    history = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        totals = {"loss": 0.0, "waveform": 0.0, "transient": 0.0, "envelope": 0.0, "long_shape": 0.0}
        examples = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            restored = model(batch["wet"])
            loss, parts = restoration_loss(restored, batch["clean"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            count = len(batch["wet"])
            totals["loss"] += float(loss.detach()) * count
            for name, value in parts.items():
                totals[name] += value * count
            examples += count
        train_metrics = {name: value / max(examples, 1) for name, value in totals.items()}
        calibration_loss = _mean_loss(model, calibration, batch_size)
        history.append({"epoch": epoch + 1, "train": train_metrics, "calibration_loss": calibration_loss})
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_state = deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    quality = _quality(model, development)
    runtime = _runtime(model, frames)
    accepted = bool(not quick and quality["accepted"] and runtime["realtime_factor"] <= 0.5)

    target = output / mechanism
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "model.pt"
    payload = {
        "schema": 1,
        "architecture": model.manifest(),
        "state_dict": model.state_dict(),
        "sample_rate": 48_000,
    }
    torch.save(payload, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 1,
        "status": "accepted" if accepted else "diagnostic-not-promoted",
        "accepted": accepted,
        "quick": quick,
        "mechanism": mechanism,
        "model": {**model.manifest(), "checkpoint": str(checkpoint), "sha256": digest},
        "provenance": {
            "product_sources_only": True,
            "source_ids": list(fit.authorization["sources"]),
            "required_attribution": list(fit.authorization["required_attribution"]),
            "research_source_ids": [],
            "research_data_isolated": True,
            "research_data_used_for_gradients": False,
            "research_data_used_for_calibration_or_selection": False,
        },
        "training": {
            "epochs": epochs,
            "fit_samples_per_epoch": train_samples,
            "calibration_samples": calibration_samples,
            "development_samples": development_samples,
            "frames": frames,
            "batch_size": batch_size,
            "history": history,
            "selected_calibration_loss": best_loss,
        },
        "development": quality,
        "runtime": runtime,
        "license": fit.authorization,
        "quality": {
            "source_audio_modified": False,
            "generated_audio_written": False,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
            "locked_final_audio_opened": False,
            "research_data_used_for_gradients": False,
            "research_data_used_for_calibration_or_selection": False,
        },
    }
    (target / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/product1"))
    parser.add_argument("--mechanisms", default=",".join(MECHANISMS))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=768)
    parser.add_argument("--calibration-samples", type=int, default=96)
    parser.add_argument("--development-samples", type=int, default=96)
    parser.add_argument("--frames", type=int, default=16_384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--blocks", type=int, default=7)
    args = parser.parse_args()
    mechanisms = tuple(name.strip() for name in args.mechanisms.split(",") if name.strip())
    if not mechanisms or any(name not in MECHANISMS for name in mechanisms):
        raise ValueError(f"mechanisms must be selected from {MECHANISMS}")
    if args.quick:
        args.epochs = 2
        args.train_samples = 48
        args.calibration_samples = 12
        args.development_samples = 12
        args.frames = 4096
        args.batch_size = 4
        args.channels = 16
        args.blocks = 4

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_audit = audit_product_data(workspace)
    (output / "data.json").write_text(json.dumps(data_audit, indent=2, sort_keys=True) + "\n")
    reports = []
    for mechanism in mechanisms:
        report = train_one(
            workspace,
            output,
            mechanism,
            epochs=args.epochs,
            train_samples=args.train_samples,
            calibration_samples=args.calibration_samples,
            development_samples=args.development_samples,
            frames=args.frames,
            batch_size=args.batch_size,
            channels=args.channels,
            blocks=args.blocks,
            quick=args.quick,
        )
        reports.append(report)
        print(json.dumps({"mechanism": mechanism, "status": report["status"], "development": report["development"], "runtime": report["runtime"]}, sort_keys=True))
    parameters = {report["model"]["parameters"] for report in reports}
    if len(parameters) != 1:
        raise RuntimeError(f"independent baselines do not have equal budgets: {parameters}")
    summary = {
        "schema": 1,
        "status": "accepted" if all(report["accepted"] for report in reports) else "diagnostic-not-promoted",
        "mechanisms": [report["mechanism"] for report in reports],
        "equal_parameter_budget": parameters.pop(),
        "reports": {report["mechanism"]: f"{report['mechanism']}/metrics.json" for report in reports},
        "product_sources_only": True,
        "research_data_isolated": True,
        "physical_audio_devices_used": False,
        "locked_final_audio_opened": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
