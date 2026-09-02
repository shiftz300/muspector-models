#!/usr/bin/env python3
"""Refine learned RAT controls through differentiable stable-model reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .asrnn_data import rat_files, read_rat_pair
from .drive_model import DriveControlEstimator
from .forward_drive import pre_emphasis
from .stable_rat import load_stable_rat
from .train import device
from .train_asrnn_rat_inverse import (
    closed_loop,
    control_metrics,
    feature_matrix,
    predict,
)
from .train_asrnn_rat_adapter import _partition


def _identifiable_metrics(prediction, targets, audible):
    result = control_metrics(prediction, targets)
    masked = {}
    for index, name in enumerate(("distortion", "tone")):
        error = (prediction[audible, index] - targets[audible, index]).abs().numpy()
        masked[name] = {
            "examples": int(audible.sum()),
            "mae_normalized": float(np.mean(error)),
            "p95_normalized": float(np.quantile(error, 0.95)),
            "mae_knob_points_0_100": float(np.mean(error) * 100.0),
            "p95_knob_points_0_100": float(np.quantile(error, 0.95) * 100.0),
        }
    result["audible_distortion_tone"] = masked
    result["unidentifiable_quiet_examples"] = int((~audible).sum())
    return result


def _audible_mask(paths):
    values = []
    for path in paths:
        _, wet, _ = read_rat_pair(path)
        values.append(float(np.max(np.abs(wet))) >= 1.0e-3)
    return torch.tensor(values, dtype=torch.bool)


def refine(paths, initial, renderer_path, *, steps, learning_rate, frames, batch_size, pair_reader=read_rat_pair, compute=None):
    target_device = device() if compute is None else torch.device(compute)
    renderer = load_stable_rat(renderer_path).to(target_device)
    for parameter in renderer.parameters():
        parameter.requires_grad_(False)
    refined = []
    for offset in range(0, len(paths), batch_size):
        dry, wet = [], []
        for path in paths[offset : offset + batch_size]:
            clean, affected, _ = pair_reader(path)
            dry.append(torch.from_numpy(clean[:frames]))
            wet.append(torch.from_numpy(affected[:frames]))
        dry = torch.stack(dry).to(target_device)
        wet = torch.stack(wet).to(target_device)
        seed = initial[offset : offset + len(dry)].to(target_device).clamp(1.0e-4, 1.0 - 1.0e-4)
        logits = torch.logit(seed).detach().requires_grad_(True)
        optimizer = torch.optim.Adam((logits,), lr=learning_rate)
        target_wet = wet[:, 1_024:]
        target_scale = target_wet.abs().mean(1).clamp_min(1.0e-4)
        emphasized_target = pre_emphasis(target_wet)
        emphasized_scale = emphasized_target.abs().mean(1).clamp_min(1.0e-4)
        for _ in range(steps):
            controls = torch.sigmoid(logits)
            rendered, _ = renderer(dry, controls)
            rendered = rendered[:, 1_024:]
            waveform = (rendered - target_wet).abs().mean(1) / target_scale
            emphasized = (
                (pre_emphasis(rendered) - emphasized_target).abs().mean(1)
                / emphasized_scale
            )
            prior = (controls - seed).square().mean(1)
            loss = (waveform + 0.25 * emphasized + 1.0e-3 * prior).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        refined.append(torch.sigmoid(logits.detach()).cpu())
        print(json.dumps({"refined": min(offset + batch_size, len(paths)), "total": len(paths)}))
    return torch.cat(refined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--inverse", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--frames", type=int, default=4_096)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"RAT inverse refinement output already exists: {args.output}")
    payload = torch.load(args.inverse, map_location="cpu", weights_only=True)
    model = DriveControlEstimator()
    model.load_state_dict(payload["state_dict"], strict=True)
    target_device = device()
    model = model.to(target_device).eval()
    _, calibration_paths = _partition(rat_files(args.corpus, "train"))
    eval_paths = rat_files(args.corpus, "eval")

    calibration_features = feature_matrix(calibration_paths)
    calibration_initial = predict(model, calibration_features, target_device)
    calibration_refined = refine(
        calibration_paths,
        calibration_initial,
        args.renderer,
        steps=args.steps,
        learning_rate=args.learning_rate,
        frames=args.frames,
        batch_size=args.batch_size,
    )
    calibration_audible = _audible_mask(calibration_paths)
    calibration_metrics = _identifiable_metrics(
        calibration_refined, calibration_features[2], calibration_audible
    )

    eval_features = feature_matrix(eval_paths)
    eval_initial = predict(model, eval_features, target_device)
    eval_refined = refine(
        eval_paths,
        eval_initial,
        args.renderer,
        steps=args.steps,
        learning_rate=args.learning_rate,
        frames=args.frames,
        batch_size=args.batch_size,
    )
    eval_audible = _audible_mask(eval_paths)
    eval_metrics = _identifiable_metrics(eval_refined, eval_features[2], eval_audible)
    reconstruction = closed_loop(eval_paths, eval_refined, args.renderer, target_device)
    accepted = bool(
        eval_metrics["per_control"]["volume"]["mae_normalized"] <= 0.06
        and eval_metrics["per_control"]["volume"]["p95_normalized"] <= 0.18
        and all(
            values["mae_normalized"] <= 0.08 and values["p95_normalized"] <= 0.22
            for values in eval_metrics["audible_distortion_tone"].values()
        )
        and reconstruction["recovered_vs_bypass_esr_improvement"] >= 0.90
        and reconstruction["recovered_vs_bypass_mae_improvement"] >= 0.80
    )
    report = {
        "schema": 1,
        "status": "accepted-internal-noncommercial-pilot" if accepted else "rejected",
        "accepted": accepted,
        "algorithm": {
            "initializer": str(args.inverse),
            "stable_forward": str(args.renderer),
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "optimization_frames": args.frames,
            "quiet_identifiability_peak_threshold": 1.0e-3,
        },
        "calibration": calibration_metrics,
        "official_eval": eval_metrics,
        "closed_loop": reconstruction,
        "quality_policy": {
            "source_files_read_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
        },
        "limitation": "quiet Wet cannot identify upstream Distortion or Filter settings",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
