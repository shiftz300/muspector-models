#!/usr/bin/env python3
"""MPS pretraining for the product-eligible rusty-amp gray-box inverse."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .open_riff_box_stage_data import sha256
from .train_open_riff_box_stage_probe import _stage_loss, _stage_loss_parts
from .rusty_amp_stage_model import RustyAmpStagewiseGrayBoxInverse


def _load(root: Path, sample: dict, device: torch.device) -> list[torch.Tensor]:
    values = []
    for relative, digest in zip(sample["stages"], sample["sha256"], strict=True):
        path = root / relative
        if sha256(path) != digest:
            raise ValueError(f"stage hash changed: {path}")
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != 44_100 or audio.shape[1] != 1:
            raise ValueError(f"unexpected stage geometry: {path}")
        values.append(torch.from_numpy(audio[:, 0].copy()).to(device).unsqueeze(0))
    return values


def _crop(stages: list[torch.Tensor], frames: int, seed: int) -> list[torch.Tensor]:
    if stages[0].shape[1] < frames:
        raise ValueError("stage render is shorter than training crop")
    rng = random.Random(seed)
    start = rng.randint(0, stages[0].shape[1] - frames)
    return [value[:, start:start + frames] for value in stages]


def _mean_loss(model, root: Path, samples: list[dict], device: torch.device, frames: int) -> float:
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for row in samples:
            stages = _crop(_load(root, row, device), frames, 9000 + row["sample"])
            total += float(_stage_loss(model.stage_outputs(stages[-1]), stages))
    return total / len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--target-frames", type=int, default=16_384)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to replace run: {args.output}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device(args.device)
    random.seed(20260905)
    np.random.seed(20260905)
    torch.manual_seed(20260905)
    root = args.data.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["product_weight_eligible"] or manifest["cabinet_assets_used"]:
        raise PermissionError("rusty-amp stage data is not product eligible and asset-free")
    fit = [row for row in manifest["samples"] if row["split"] == "fit"]
    calibration = [row for row in manifest["samples"] if row["split"] == "calibration"]
    model = RustyAmpStagewiseGrayBoxInverse().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    initial = _mean_loss(model, root, calibration, device, args.target_frames)
    best, best_epoch = initial, 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history = [{"epoch": 0, "calibration_stage_loss": initial}]
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(fit)
        random.Random(20260905 + epoch).shuffle(order)
        model.train()
        total = 0.0
        for row in order:
            stages = _crop(
                _load(root, row, device), args.target_frames,
                20260905 + epoch * 1009 + row["sample"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss = _stage_loss(model.stage_outputs(stages[-1]), stages)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        calibration_loss = _mean_loss(model, root, calibration, device, args.target_frames)
        history.append({
            "epoch": epoch,
            "train_stage_loss": total / len(order),
            "calibration_stage_loss": calibration_loss,
        })
        if calibration_loss < best:
            best, best_epoch = calibration_loss, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(json.dumps(history[-1]), flush=True)
    model = model.cpu()
    model.load_state_dict(best_state)
    per_stage = np.zeros(8, dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for row in calibration:
            stages = _crop(_load(root, row, torch.device("cpu")), args.target_frames, 9000 + row["sample"])
            for index, loss in enumerate(_stage_loss_parts(model.stage_outputs(stages[-1]), stages)):
                per_stage[index] += float(loss)
    improvement = (initial - best) / max(initial, 1e-8)
    accepted = best_epoch > 0 and improvement >= 0.20
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "model.pt"
    torch.save({
        "schema": 1, "sample_rate": 44_100, "architecture": model.manifest(),
        "state_dict": model.state_dict(), "source_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
    }, checkpoint)
    report = {
        "schema": 1,
        "status": "accepted-product-pretraining" if accepted else "rejected-pretraining",
        "accepted": accepted,
        "device": str(device),
        "model": {**model.manifest(), "checkpoint": str(checkpoint), "sha256": sha256(checkpoint)},
        "training": {
            "fit_samples": len(fit), "calibration_samples": len(calibration),
            "epochs": args.epochs, "target_frames": args.target_frames,
            "initial_stage_loss": initial, "selected_stage_loss": best,
            "selected_epoch": best_epoch, "improvement_fraction": improvement,
            "history": history,
            "calibration_loss_by_inverse_stage": {
                name: float(value / len(calibration))
                for name, value in zip(model.manifest()["internal_reverse_stages"], per_stage, strict=True)
            },
        },
        "provenance": {
            "source_ids": manifest["source_ids"],
            "required_attribution": manifest["authorization"]["required_attribution"],
            "renderer_commit": manifest["source_commit"],
            "renderer_binary_sha256": manifest["renderer_binary_sha256"],
            "cabinet_assets_used": False,
            "external_plugins_used": False,
            "physical_validation": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": "fine-tune on authorized measured/product Amp audio; synthetic pretraining cannot promote",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "improvement_fraction": improvement, "checkpoint": str(checkpoint)}, indent=2))


if __name__ == "__main__":
    main()
