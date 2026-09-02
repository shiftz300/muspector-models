#!/usr/bin/env python3
"""Train and close-loop evaluate RAT knob recovery on the ASRNN hardware corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .asrnn_data import LICENSE, RECORD_URL, rat_files, read_rat_pair
from .drive_model import DriveControlEstimator, drive_features
from .stable_rat import load_stable_rat
from .train import device
from .train_asrnn_rat_adapter import _partition


CONTROL_NAMES = ("distortion", "tone", "volume")


@torch.inference_mode()
def feature_matrix(paths: list[Path], batch_size: int = 32):
    images, statistics, targets = [], [], []
    for offset in range(0, len(paths), batch_size):
        dry, wet, controls = [], [], []
        for path in paths[offset : offset + batch_size]:
            clean, affected, setting = read_rat_pair(path)
            dry.append(torch.from_numpy(clean))
            wet.append(torch.from_numpy(affected))
            controls.append(torch.from_numpy(setting))
        image, stats = drive_features(torch.stack(dry), torch.stack(wet))
        images.append(image.cpu())
        statistics.append(stats.cpu())
        targets.append(torch.stack(controls).cpu())
    return torch.cat(images), torch.cat(statistics), torch.cat(targets)


@torch.inference_mode()
def predict(model, features, target_device):
    image, statistics, _ = features
    rows = []
    for image_batch, statistics_batch in DataLoader(
        TensorDataset(image, statistics), batch_size=128
    ):
        rows.append(
            torch.sigmoid(
                model(image_batch.to(target_device), statistics_batch.to(target_device))
            ).cpu()
        )
    return torch.cat(rows)


def control_metrics(prediction: torch.Tensor, targets: torch.Tensor) -> dict:
    absolute = (prediction - targets).abs().numpy()
    per_control = {}
    for index, name in enumerate(CONTROL_NAMES):
        values = absolute[:, index]
        per_control[name] = {
            "mae_normalized": float(np.mean(values)),
            "median_normalized": float(np.median(values)),
            "p95_normalized": float(np.quantile(values, 0.95)),
            "mae_knob_points_0_100": float(np.mean(values) * 100.0),
            "p95_knob_points_0_100": float(np.quantile(values, 0.95) * 100.0),
        }
    return {
        "examples": len(prediction),
        "per_control": per_control,
        "macro_mae_normalized": float(np.mean(absolute)),
        "macro_p95_normalized": float(np.mean(np.quantile(absolute, 0.95, axis=0))),
    }


@torch.inference_mode()
def closed_loop(paths, recovered, renderer_path, target_device, frames=2_048):
    renderer = load_stable_rat(renderer_path).to(target_device)
    recovered_errors, oracle_errors, bypass_errors = [], [], []
    recovered_absolute = oracle_absolute = bypass_absolute = target_absolute = 0.0
    total_samples = 0
    for offset in range(0, len(paths), 32):
        dry, wet, truth = [], [], []
        for path in paths[offset : offset + 32]:
            clean, affected, controls = read_rat_pair(path)
            dry.append(torch.from_numpy(clean))
            wet.append(torch.from_numpy(affected))
            truth.append(torch.from_numpy(controls))
        dry = torch.stack(dry).to(target_device)
        wet = torch.stack(wet).to(target_device)
        truth = torch.stack(truth).to(target_device)
        estimate = recovered[offset : offset + len(dry)].to(target_device)

        def render(controls):
            state = None
            chunks = []
            for start in range(0, dry.shape[1], frames):
                value, state = renderer(dry[:, start : start + frames], controls, state)
                chunks.append(value)
            return torch.cat(chunks, dim=1)[:, 1_024:]

        predicted = render(estimate)
        oracle = render(truth)
        clean = dry[:, 1_024:]
        target_wet = wet[:, 1_024:]
        energy = target_wet.square().mean(1).clamp_min(1.0e-8)
        recovered_errors.extend(((predicted - target_wet).square().mean(1) / energy).cpu().tolist())
        oracle_errors.extend(((oracle - target_wet).square().mean(1) / energy).cpu().tolist())
        bypass_errors.extend(((clean - target_wet).square().mean(1) / energy).cpu().tolist())
        recovered_absolute += float((predicted - target_wet).abs().sum())
        oracle_absolute += float((oracle - target_wet).abs().sum())
        bypass_absolute += float((clean - target_wet).abs().sum())
        target_absolute += float(target_wet.abs().sum())
        total_samples += target_wet.numel()
    recovered_esr = float(np.mean(recovered_errors))
    oracle_esr = float(np.mean(oracle_errors))
    bypass_esr = float(np.mean(bypass_errors))
    return {
        "examples": len(paths),
        "mean_recovered_control_esr": recovered_esr,
        "mean_oracle_control_esr": oracle_esr,
        "mean_bypass_esr": bypass_esr,
        "recovered_vs_bypass_esr_improvement": 1.0 - recovered_esr / max(bypass_esr, 1.0e-12),
        "recovered_global_mae": recovered_absolute / total_samples,
        "oracle_global_mae": oracle_absolute / total_samples,
        "bypass_global_mae": bypass_absolute / total_samples,
        "recovered_vs_bypass_mae_improvement": 1.0 - recovered_absolute / max(
            bypass_absolute, 1.0e-12
        ),
        "target_mean_absolute_amplitude": target_absolute / total_samples,
    }


def train(corpus, renderer_path, output, epochs=100, batch_size=64):
    if (output / "rat-inverse.pt").exists() or (output / "metrics.json").exists():
        raise ValueError(f"RAT inverse output already exists: {output}")
    fit_paths, calibrate_paths = _partition(rat_files(corpus, "train"))
    eval_paths = rat_files(corpus, "eval")
    fit = feature_matrix(fit_paths)
    calibrate = feature_matrix(calibrate_paths)
    evaluation = feature_matrix(eval_paths)
    torch.manual_seed(20260830)
    np.random.seed(20260830)
    target_device = device()
    model = DriveControlEstimator().to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4.0e-4, weight_decay=1.0e-4)
    loader = DataLoader(TensorDataset(*fit), batch_size=batch_size, shuffle=True)
    best_score = float("inf")
    best_state = None
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for image, statistics, targets in loader:
            prediction = torch.sigmoid(
                model(image.to(target_device), statistics.to(target_device))
            )
            loss = torch.nn.functional.smooth_l1_loss(
                prediction, targets.to(target_device), beta=0.03
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        calibration_prediction = predict(model, calibrate, target_device)
        calibration_metrics = control_metrics(calibration_prediction, calibrate[2])
        score = calibration_metrics["macro_mae_normalized"]
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "calibrate_mae": score}
        history.append(row)
        if score < best_score:
            best_score = score
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(json.dumps(row, sort_keys=True))
    if best_state is None:
        raise RuntimeError("RAT inverse training produced no checkpoint")
    model.load_state_dict(best_state)
    calibration_prediction = predict(model, calibrate, target_device)
    eval_prediction = predict(model, evaluation, target_device)
    calibration_metrics = control_metrics(calibration_prediction, calibrate[2])
    eval_metrics = control_metrics(eval_prediction, evaluation[2])
    reconstruction = closed_loop(
        eval_paths, eval_prediction, renderer_path, target_device
    )
    accepted = bool(
        eval_metrics["macro_mae_normalized"] <= 0.05
        and eval_metrics["macro_p95_normalized"] <= 0.15
        and all(
            values["mae_normalized"] <= 0.06
            and values["p95_normalized"] <= 0.18
            for values in eval_metrics["per_control"].values()
        )
        and reconstruction["recovered_vs_bypass_esr_improvement"] >= 0.75
        and reconstruction["recovered_vs_bypass_mae_improvement"] >= 0.70
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "rat-inverse.pt"
    torch.save(
        {
            "schema": 1,
            "sample_rate": 48_000,
            "state_dict": best_state,
            "dataset_record": RECORD_URL,
            "dataset_license": LICENSE,
            "forward_checkpoint_sha256": hashlib.sha256(renderer_path.read_bytes()).hexdigest(),
        },
        checkpoint,
    )
    report = {
        "schema": 1,
        "status": "accepted-internal-noncommercial-pilot" if accepted else "rejected",
        "accepted": accepted,
        "dataset_record": RECORD_URL,
        "dataset_license": LICENSE,
        "partitions": {
            "fit_files": len(fit_paths),
            "calibrate_files": len(calibrate_paths),
            "official_eval_files": len(eval_paths),
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "calibration": calibration_metrics,
        "official_eval": eval_metrics,
        "closed_loop": reconstruction,
        "history": history,
        "quality_policy": {
            "source_files_read_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
        },
        "release_limitations": [
            "CC-BY-NC-4.0 internal research only",
            "official eval is development evidence, not locked-final",
            "no UI or product promotion",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    report = train(
        args.corpus, args.renderer, args.output, epochs=args.epochs, batch_size=args.batch_size
    )
    print(json.dumps({"accepted": report["accepted"], "official_eval": report["official_eval"], "closed_loop": report["closed_loop"]}, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
