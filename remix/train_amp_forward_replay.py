#!/usr/bin/env python3
"""Fit and audit a small forward replay clone for licensed P1/P2 Amp data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .amp_forward_model import AmpForwardReplay
from .guitar_techs_amp_data import PROFILES, RATE, SOURCE_ID
from .train_amp_cab import _pairs
from clone.evaluation.evaluate import one_example, summarize


SEED = 20260924


def _collate(rows: list[dict]) -> dict:
    starts = {int(row["target_start"]) for row in rows}
    if len(starts) != 1:
        raise ValueError("one forward replay batch must share target geometry")
    return {
        "direct": torch.stack([row["clean"] for row in rows]),
        "wet": torch.stack([row["wet"] for row in rows]),
        "target_start": starts.pop(),
    }


def _batch_to(batch: dict, device: torch.device) -> dict:
    return {
        **batch,
        "direct": batch["direct"].to(device),
        "wet": batch["wet"].to(device),
    }


def _loss(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    replay = model(batch["direct"])
    start = batch["target_start"]
    replay = replay[:, start:]
    wet = batch["wet"][:, start:]
    waveform = torch.mean((replay - wet) ** 2)
    replay_difference = replay[:, 1:] - replay[:, :-1]
    wet_difference = wet[:, 1:] - wet[:, :-1]
    transient = torch.mean(torch.abs(replay_difference - wet_difference))
    return waveform + 0.25 * transient


def _mean_loss(
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total = 0.0
    examples = 0
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            batch = _batch_to(raw, device)
            value = _loss(model, batch)
            count = len(batch["direct"])
            total += float(value.detach().cpu()) * count
            examples += count
    return total / max(examples, 1)


def _quality(model: torch.nn.Module, dataset) -> dict:
    replay_metrics = []
    identity_metrics = []
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            replay = model(row["clean"].unsqueeze(0))[0].cpu().numpy()
            start = int(row["target_start"])
            direct = row["clean"][start:].numpy()
            candidate = replay[start:].astype(np.float32)
            wet = row["wet"][start:].numpy()
            replay_metrics.append(one_example(candidate, wet))
            identity_metrics.append(one_example(direct, wet))
    return {
        "replay": summarize(replay_metrics),
        "direct_identity_baseline": summarize(identity_metrics),
    }


def _train_profile(args: argparse.Namespace, profile_index: int) -> dict:
    profile = PROFILES[profile_index]
    fit = _pairs(
        args.workspace, "fit", args.train_samples, args.target_frames,
        SEED + 1 + profile_index, profile_index,
    )
    calibration = _pairs(
        args.workspace, "calibration", args.calibration_samples, args.target_frames,
        SEED + 3 + profile_index, profile_index,
    )
    development = _pairs(
        args.workspace, "development", args.development_samples, args.target_frames,
        SEED + 5 + profile_index, profile_index,
    )
    device = torch.device(args.device)
    model = AmpForwardReplay(args.pre_taps, args.post_taps).to(device)
    loader = DataLoader(
        fit,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED + profile_index),
        collate_fn=_collate,
    )
    initial_loss = _mean_loss(model, calibration, args.batch_size, device)
    best_loss = initial_loss
    best_epoch = 0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    history = [{"epoch": 0, "train_loss": None, "calibration_loss": initial_loss}]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5)
    print(json.dumps({
        "profile": profile.id,
        "stage": "forward_replay",
        "epoch": 0,
        "calibration_loss": initial_loss,
    }), flush=True)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        examples = 0
        for raw in loader:
            batch = _batch_to(raw, device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            count = len(batch["direct"])
            total += float(loss.detach().cpu()) * count
            examples += count
        calibration_loss = _mean_loss(model, calibration, args.batch_size, device)
        train_loss = total / max(examples, 1)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "calibration_loss": calibration_loss,
        })
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(json.dumps({
            "profile": profile.id,
            "stage": "forward_replay",
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "calibration_loss": calibration_loss,
        }), flush=True)
    model = model.cpu()
    model.load_state_dict(best_state)
    checkpoint = args.output / f"forward-{profile.id}.pt"
    torch.save({
        "schema": 1,
        "sample_rate": RATE,
        "profile_id": profile.id,
        "architecture": model.manifest(),
        "state_dict": model.state_dict(),
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {
        "profile": profile.id,
        "hardware": profile.hardware,
        "architecture": model.manifest(),
        "checkpoint": str(checkpoint),
        "sha256": digest,
        "training": {
            "accelerator": device.type,
            "epochs": args.epochs,
            "selected_epoch": best_epoch,
            "initial_calibration_loss": initial_loss,
            "selected_calibration_loss": best_loss,
            "train_samples": args.train_samples,
            "calibration_samples": args.calibration_samples,
            "development_samples": args.development_samples,
            "target_frames": args.target_frames,
            "history_frames": fit.total_frames - args.target_frames,
            "history": history,
        },
        "development": _quality(model, development),
        "data": {
            "source_id": SOURCE_ID,
            "product_weights_authorized": True,
            "p3_input_or_profile": False,
        },
    }


def run(args: argparse.Namespace) -> dict:
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace forward replay output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    profiles = [_train_profile(args, profile_index) for profile_index in range(len(PROFILES))]
    report = {
        "schema": 1,
        "status": "diagnostic-forward-replay",
        "mechanism": "amp",
        "device_scope": "Guitar-TECHS P1/P2 fixed Amp+cab+mic profiles",
        "profiles": profiles,
        "contract": {
            "direction": "direct-input to Amp+cab+mic",
            "topology_input": False,
            "neighbor_effect_input": False,
            "analytic_inverse_used": False,
            "physical_audio_devices_used": False,
            "p3_used": False,
            "calibration_selects_epoch": True,
            "development_read_once_after_selection": True,
        },
        "data": {
            "source_id": SOURCE_ID,
            "license": "CC BY 4.0; product-weight authorization enforced by loader",
            "audit": str((workspace / "runs/foundation/product3-amp-guitar-techs/corrected-pair-audit.json").resolve()),
            "profiles": [profile.id for profile in PROFILES],
        },
        "quality": {
            "generated_audio_written": False,
            "demo_generated": False,
            "source_audio_modified": False,
        },
        "limitations": [
            "forward replay quality is a prerequisite diagnostic, not an inverse-model acceptance result",
            "two fixed named profiles only; no generic Amp or knob interpolation claim",
            "the causal structure may underrepresent residual alignment or cabinet phase behavior",
        ],
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    registry_path = workspace / "experiments.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    registry.append({
        "id": output.name,
        "git_commit": None,
        "data_manifest_sha256": hashlib.sha256(
            (workspace / "runs/foundation/product3-amp-guitar-techs/corrected-pair-audit.json").read_bytes()
        ).hexdigest(),
        "algorithm": "causal Wiener-Hammerstein Amp forward replay",
        "fit_split": "fit",
        "calibration_split": "calibration",
        "validation_split": "development",
        "locked": False,
        "status": "diagnostic",
        "metrics": str(metrics_path),
    })
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "profiles": [
            {
                "profile": row["profile"],
                "selected_epoch": row["training"]["selected_epoch"],
                "replay_esr": row["development"]["replay"]["absolute_esr"]["mean"],
                "identity_esr": row["development"]["direct_identity_baseline"]["absolute_esr"]["mean"],
            }
            for row in profiles
        ],
    }, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product3-amp-guitar-techs/amp-forward-replay-v1"),
    )
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=180)
    parser.add_argument("--calibration-samples", type=int, default=60)
    parser.add_argument("--development-samples", type=int, default=60)
    parser.add_argument("--target-frames", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pre-taps", type=int, default=65)
    parser.add_argument("--post-taps", type=int, default=257)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    args = parser.parse_args()
    if args.epochs < 1 or args.train_samples < 2 or args.calibration_samples < 2 or args.development_samples < 2:
        raise ValueError("sample counts and epochs must be positive")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))
    run(args)


if __name__ == "__main__":
    main()
