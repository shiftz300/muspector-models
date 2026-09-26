#!/usr/bin/env python3
"""Train a disposable MPS stage-supervision probe and retain metrics only."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from .open_riff_box_stage_model import OpenRiffBoxStagewiseGrayBoxInverse
from .quality2 import summarize


def _load(root: Path, sample: dict, device: torch.device) -> list[torch.Tensor]:
    values = []
    for relative in sample["stages"]:
        audio, rate = sf.read(root / relative, dtype="float32", always_2d=True)
        if rate != 44_100:
            raise ValueError(f"unexpected stage rate: {rate}")
        values.append(torch.from_numpy(audio.mean(1)).to(device).unsqueeze(0))
    return values


def _stage_loss_parts(
    predicted: list[torch.Tensor], targets: list[torch.Tensor]
) -> list[torch.Tensor]:
    losses = []
    for output, target in zip(predicted, reversed(targets[:-1]), strict=True):
        scale = target.square().mean(1).add(1.0e-8).sqrt().clamp_min(1.0e-3)
        waveform = ((output - target).abs().mean(1) / scale).clamp_max(20.0).mean()
        output_diff = output[:, 1:] - output[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        diff_scale = target_diff.abs().mean(1).clamp_min(1.0e-4)
        transient = ((output_diff - target_diff).abs().mean(1) / diff_scale).clamp_max(20.0).mean()
        output_envelope = output.unfold(1, 256, 128).square().mean(-1).add(1.0e-8).sqrt()
        target_envelope = target.unfold(1, 256, 128).square().mean(-1).add(1.0e-8).sqrt()
        envelope = F.l1_loss(
            torch.log(output_envelope + 1.0e-4), torch.log(target_envelope + 1.0e-4)
        )
        output_frames = output.unfold(1, 1024, 256)
        target_frames = target.unfold(1, 1024, 256)
        output_crest = output_frames.abs().amax(-1) / output_frames.square().mean(-1).add(1.0e-8).sqrt()
        target_crest = target_frames.abs().amax(-1) / target_frames.square().mean(-1).add(1.0e-8).sqrt()
        crest = F.l1_loss(torch.log(output_crest + 1.0e-4), torch.log(target_crest + 1.0e-4))
        output_attack_rms = output.unfold(1, 240, 120).square().mean(-1).add(1.0e-8).sqrt()
        target_attack_rms = target.unfold(1, 240, 120).square().mean(-1).add(1.0e-8).sqrt()
        output_attack = torch.relu(torch.diff(torch.log(output_attack_rms + 1.0e-6), dim=1))
        target_attack = torch.relu(torch.diff(torch.log(target_attack_rms + 1.0e-6), dim=1))
        attack = F.l1_loss(output_attack, target_attack)
        losses.append(
            waveform + 0.20 * transient + 0.25 * envelope + 0.50 * crest + 1.0 * attack
        )
    return losses


def _stage_loss(predicted: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
    values = torch.stack(_stage_loss_parts(predicted, targets))
    weights = torch.linspace(1.0, 2.0, len(values), device=values.device)
    weights[-1] = 3.0
    return (values * weights).sum() / weights.sum()


def _mean_loss(model, root: Path, samples: list[dict], device: torch.device) -> float:
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for sample in samples:
            stages = _load(root, sample, device)
            total += float(_stage_loss(model.stage_outputs(stages[-1]), stages))
    return total / max(len(samples), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device(args.device)
    torch.manual_seed(20260905)
    random.seed(20260905)
    np.random.seed(20260905)

    manifest = json.loads((args.data / "manifest.json").read_text())
    if manifest["product_weight_eligible"] or manifest["cabinet_assets_used"]:
        raise PermissionError("stage probe data violated its research-only no-cabinet contract")
    fit = [row for row in manifest["samples"] if row["split"] == "fit"]
    calibration = [row for row in manifest["samples"] if row["split"] == "calibration"]
    model = OpenRiffBoxStagewiseGrayBoxInverse(research_probe=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=1.0e-5)
    initial = _mean_loss(model, args.data, calibration, device)
    history = [{"epoch": 0, "calibration_stage_loss": initial}]
    best_loss = initial
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(fit)
        model.train()
        train_loss = 0.0
        for sample in fit:
            stages = _load(args.data, sample, device)
            optimizer.zero_grad(set_to_none=True)
            loss = _stage_loss(model.stage_outputs(stages[-1]), stages)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss += float(loss.detach())
        calibration_loss = _mean_loss(model, args.data, calibration, device)
        history.append({
            "epoch": epoch,
            "train_stage_loss": train_loss / len(fit),
            "calibration_stage_loss": calibration_loss,
        })
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(best_state)

    wet_values, restored_values, clean_values = [], [], []
    per_stage_losses = np.zeros(9, dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for sample in calibration:
            stages = _load(args.data, sample, device)
            restored, _, _ = model(stages[-1])
            for index, loss in enumerate(_stage_loss_parts(model.stage_outputs(stages[-1]), stages)):
                per_stage_losses[index] += float(loss)
            wet_values.append(stages[-1][0].cpu().numpy())
            restored_values.append(restored[0].cpu().numpy())
            clean_values.append(stages[0][0].cpu().numpy())
    quality = summarize("amp", wet_values, restored_values, clean_values)
    report = {
        "schema": 1,
        "status": "research-only-architecture-probe-not-promotable",
        "accepted": False,
        "product_weight_eligible": False,
        "weights_retained": False,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "model": model.manifest(),
        "training": {
            "fit_samples": len(fit),
            "calibration_samples": len(calibration),
            "epochs": args.epochs,
            "initial_stage_loss": initial,
            "best_stage_loss": best_loss,
            "best_epoch": best_epoch,
            "improvement_fraction": (initial - best_loss) / max(initial, 1.0e-8),
            "calibration_loss_by_inverse_stage": {
                name: float(value / len(calibration))
                for name, value in zip(
                    reversed(manifest["stage_names"][:-1]), per_stage_losses, strict=True
                )
            },
            "loss_policy": "deep waveform, derivative, log-envelope, crest and frozen-gate-matched positive attack supervision; later inverse stages and final Clean receive higher weight",
            "history": history,
        },
        "quality": quality,
        "decision": "reinitialize for product-only EGDB training only if the stage-loss probe is material",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
