#!/usr/bin/env python3
"""Train RAT control recovery from explicit Dry/Wet spectral-transfer features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .asrnn_data import LICENSE, RECORD_URL, rat_files, read_rat_pair
from .rat_inverse import RatSpectralControlEstimator, rat_spectral_features
from .train import device
from .train_asrnn_rat_adapter import _partition
from .train_asrnn_rat_inverse import closed_loop, control_metrics


@torch.inference_mode()
def feature_matrix(paths, batch_size=32):
    features, targets = [], []
    for offset in range(0, len(paths), batch_size):
        dry, wet, controls = [], [], []
        for path in paths[offset : offset + batch_size]:
            clean, affected, setting = read_rat_pair(path)
            dry.append(torch.from_numpy(clean))
            wet.append(torch.from_numpy(affected))
            controls.append(torch.from_numpy(setting))
        features.append(rat_spectral_features(torch.stack(dry), torch.stack(wet)).cpu())
        targets.append(torch.stack(controls).cpu())
    return torch.cat(features), torch.cat(targets)


@torch.inference_mode()
def predict(model, features, target_device):
    rows = []
    for batch in DataLoader(TensorDataset(features), batch_size=128):
        rows.append(torch.sigmoid(model(batch[0].to(target_device))).cpu())
    return torch.cat(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    if (args.output / "rat-spectral-inverse.pt").exists() or (args.output / "metrics.json").exists():
        raise ValueError(f"RAT spectral inverse output already exists: {args.output}")
    fit_paths, calibrate_paths = _partition(rat_files(args.corpus, "train"))
    eval_paths = rat_files(args.corpus, "eval")
    fit = feature_matrix(fit_paths)
    calibrate = feature_matrix(calibrate_paths)
    evaluation = feature_matrix(eval_paths)
    torch.manual_seed(20260830)
    np.random.seed(20260830)
    target_device = device()
    model = RatSpectralControlEstimator().to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    loader = DataLoader(TensorDataset(*fit), batch_size=64, shuffle=True)
    best_score, best_state = float("inf"), None
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for features, targets in loader:
            prediction = torch.sigmoid(model(features.to(target_device)))
            loss = torch.nn.functional.smooth_l1_loss(
                prediction, targets.to(target_device), beta=0.03
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        calibration_prediction = predict(model, calibrate[0], target_device)
        metrics = control_metrics(calibration_prediction, calibrate[1])
        # Tone is the target of this route; retain macro pressure to avoid regressions.
        score = (
            metrics["per_control"]["tone"]["mae_normalized"]
            + 0.25 * metrics["macro_mae_normalized"]
        )
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "calibrate_score": score}
        history.append(row)
        if score < best_score:
            best_score = score
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        if epoch == 0 or (epoch + 1) % 25 == 0:
            print(json.dumps(row, sort_keys=True))
    if best_state is None:
        raise RuntimeError("RAT spectral inverse training produced no checkpoint")
    model.load_state_dict(best_state)
    calibration_prediction = predict(model, calibrate[0], target_device)
    eval_prediction = predict(model, evaluation[0], target_device)
    calibration_metrics = control_metrics(calibration_prediction, calibrate[1])
    eval_metrics = control_metrics(eval_prediction, evaluation[1])
    reconstruction = closed_loop(eval_paths, eval_prediction, args.renderer, target_device)
    accepted = bool(
        eval_metrics["macro_mae_normalized"] <= 0.08
        and eval_metrics["per_control"]["tone"]["mae_normalized"] <= 0.08
        and eval_metrics["per_control"]["tone"]["p95_normalized"] <= 0.22
        and reconstruction["recovered_vs_bypass_mae_improvement"] >= 0.75
    )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "rat-spectral-inverse.pt"
    torch.save(
        {
            "schema": 1,
            "sample_rate": 48_000,
            "state_dict": best_state,
            "dataset_record": RECORD_URL,
            "dataset_license": LICENSE,
        },
        checkpoint,
    )
    report = {
        "schema": 1,
        "status": "accepted-internal-noncommercial-pilot" if accepted else "rejected",
        "accepted": accepted,
        "partitions": {"fit": len(fit_paths), "calibrate": len(calibrate_paths), "official_eval": len(eval_paths)},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "calibration": calibration_metrics,
        "official_eval": eval_metrics,
        "closed_loop": reconstruction,
        "history": history,
        "quality_policy": {"source_files_read_only": True, "automatic_normalization": False, "automatic_limiting": False, "lossy_reencoding": False, "physical_audio_devices_used": False},
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": accepted, "official_eval": eval_metrics, "closed_loop": reconstruction}, sort_keys=True))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
