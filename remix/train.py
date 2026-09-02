#!/usr/bin/env python3
"""Train or smoke-test the paired order/control estimator."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from .data import DAFxOrderDataset, SyntheticControlDataset, dry_sources, order_manifest
from .delay_model import DelayControlEstimator, apply_delay_residual
from .differentiable import audio_loss
from .drive_model import DriveControlEstimator, apply_drive_estimate
from .model import PairedEstimator, masked_loss
from .physics import apply_delay_hints, deconvolution_context
from .reverb_model import ReverbControlEstimator, apply_reverb_estimate, reverb_features
from .spec import CONTROL_NAMES


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/corpus/guitar-effects-chains"
RUN = ROOT / "remix/runs/order-control-paired-public"


def device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def limited(dataset: Dataset, count: int) -> Dataset:
    return Subset(dataset, range(min(count, len(dataset)))) if count else dataset


@torch.no_grad()
def evaluate(
    model: PairedEstimator,
    loader: DataLoader,
    target: torch.device,
    drive_model: DriveControlEstimator | None = None,
    delay_model: DelayControlEstimator | None = None,
    reverb_model: ReverbControlEstimator | None = None,
) -> dict:
    model.eval()
    if drive_model is not None:
        drive_model.eval()
    if delay_model is not None:
        delay_model.eval()
    if reverb_model is not None:
        reverb_model.eval()
    totals = {
        "loss": 0.0,
        "order_loss": 0.0,
        "control_loss": 0.0,
        "batches": 0,
        "order_correct": 0.0,
        "order_count": 0.0,
        "order_present": 0.0,
        "order_exact": 0.0,
        "order_examples": 0.0,
        "control_absolute": 0.0,
        "control_count": 0.0,
    }
    normalized_absolute = np.zeros(len(CONTROL_NAMES), dtype=np.float64)
    physical_absolute = np.zeros(len(CONTROL_NAMES), dtype=np.float64)
    control_counts = np.zeros(len(CONTROL_NAMES), dtype=np.float64)
    for batch in loader:
        batch = {name: value.to(target) for name, value in batch.items()}
        raw_estimate = model(batch["dry"], batch["wet"])
        context = deconvolution_context(batch["dry"], batch["wet"])
        estimate = apply_delay_hints(
            raw_estimate, batch["dry"], batch["wet"], context
        )
        impulse_features = (
            reverb_features(batch["dry"], batch["wet"], context)
            if delay_model is not None or reverb_model is not None
            else None
        )
        if delay_model is not None:
            estimate = apply_delay_residual(
                estimate,
                batch["dry"],
                batch["wet"],
                delay_model,
                impulse_features,
                context,
            )
        if drive_model is not None:
            estimate = apply_drive_estimate(
                estimate, batch["dry"], batch["wet"], drive_model
            )
        if reverb_model is not None:
            estimate = apply_reverb_estimate(
                estimate,
                batch["dry"],
                batch["wet"],
                reverb_model,
                impulse_features,
            )
        loss, parts = masked_loss(raw_estimate, batch)
        prediction = estimate.order_logits >= 0.0
        truth = batch["order"] >= 0.5
        active_order = batch["order_mask"] >= 0.5
        totals["order_present"] += float(batch["order_present"].sum())
        totals["order_correct"] += float(((prediction == truth) & active_order).sum())
        totals["order_count"] += float(active_order.sum())
        eligible = active_order.any(dim=1)
        exact = ((prediction == truth) | ~active_order).all(dim=1) & eligible
        totals["order_exact"] += float(exact.sum())
        totals["order_examples"] += float(eligible.sum())
        control_prediction = torch.sigmoid(estimate.control_logits)
        normalized_error = torch.abs(control_prediction - batch["controls"]) * batch["control_mask"]
        totals["control_absolute"] += float(normalized_error.sum())
        totals["control_count"] += float(batch["control_mask"].sum())
        normalized_absolute += normalized_error.sum(dim=0).cpu().numpy()
        control_counts += batch["control_mask"].sum(dim=0).cpu().numpy()
        predicted_physical = physical_controls(control_prediction)
        target_physical = physical_controls(batch["controls"])
        physical_absolute += (
            torch.abs(predicted_physical - target_physical) * batch["control_mask"]
        ).sum(dim=0).cpu().numpy()
        totals["loss"] += float(loss)
        totals["order_loss"] += parts["order"]
        totals["control_loss"] += parts["controls"]
        totals["batches"] += 1
    batches = max(totals["batches"], 1.0)
    per_control = {}
    units = ("dB", "%", "dB", "ms", "%", "%", "s", "%", "%")
    for index, (name, unit) in enumerate(zip(CONTROL_NAMES, units)):
        count = max(control_counts[index], 1.0)
        per_control[name] = {
            "normalized_mae": normalized_absolute[index] / count,
            "physical_mae": physical_absolute[index] / count,
            "unit": unit,
            "values": int(control_counts[index]),
        }
    exact_accuracy = totals["order_exact"] / max(totals["order_examples"], 1.0)
    return {
        "loss": totals["loss"] / batches,
        "order_loss": totals["order_loss"] / batches,
        "control_loss": totals["control_loss"] / batches,
        "pairwise_order_accuracy": totals["order_correct"] / max(totals["order_count"], 1.0),
        "exact_order_accuracy": exact_accuracy,
        "equivalence_aware_exact_order_accuracy": exact_accuracy,
        "normalized_control_mae": totals["control_absolute"] / max(totals["control_count"], 1.0),
        "order_relations": int(totals["order_count"]),
        "ambiguous_order_relations": int(totals["order_present"] - totals["order_count"]),
        "control_values": int(totals["control_count"]),
        "controls": per_control,
        "deconvolution_delay_hints": ["time_ms", "feedback", "mix"],
        "paired_delay_residual": delay_model is not None,
        "paired_drive_estimator": drive_model is not None,
        "deconvolution_reverb_estimator": reverb_model is not None,
    }


def physical_controls(value: torch.Tensor) -> torch.Tensor:
    result = value.clone()
    result[:, 0] = value[:, 0] * 30.0
    result[:, 1] = value[:, 1] * 100.0
    result[:, 2] = value[:, 2] * 30.0 - 18.0
    result[:, 3] = 40.0 * torch.pow(25.0, value[:, 3])
    result[:, 4] = value[:, 4] * 90.0
    result[:, 5] = value[:, 5] * 70.0
    result[:, 6] = 0.2 * torch.pow(40.0, value[:, 6])
    result[:, 7] = value[:, 7] * 100.0
    result[:, 8] = value[:, 8] * 70.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--synthetic-samples", type=int, default=12_800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--audio-loss-weight",
        type=float,
        default=0.0,
        help="experimental Drive/Delay audio loss; disabled until local-gradient parity improves",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="one audited development epoch over 64 real and 64 synthetic pairs",
    )
    args = parser.parse_args()

    random.seed(20260830)
    np.random.seed(20260830)
    torch.manual_seed(20260830)

    manifest = order_manifest(args.corpus)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    real = DAFxOrderDataset(args.corpus, "train")
    synthetic_count = 16 if args.smoke else (64 if args.quick else args.synthetic_samples)
    synthetic = SyntheticControlDataset(dry_sources(args.corpus, "train"), synthetic_count)
    dataset = (
        Subset(synthetic, (4, 10))
        if args.smoke
        else ConcatDataset((limited(real, 64 if args.quick else 0), synthetic))
    )
    batch_size = len(dataset) if args.smoke else min(args.batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=not args.smoke)
    target = device()
    model = PairedEstimator().to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    epochs = 1 if args.smoke or args.quick else args.epochs
    for epoch in range(epochs):
        model.train()
        totals = {
            "loss": 0.0,
            "order": 0.0,
            "controls": 0.0,
            "audio": 0.0,
            "batches": 0,
        }
        for batch in loader:
            batch = {name: value.to(target) for name, value in batch.items()}
            estimate = model(batch["dry"], batch["wet"])
            supervised, parts = masked_loss(estimate, batch)
            reconstruction = (
                audio_loss(estimate, batch)
                if args.audio_loss_weight > 0.0
                else supervised.detach() * 0.0
            )
            optimizer.zero_grad(set_to_none=True)
            supervised.backward(retain_graph=args.audio_loss_weight > 0.0)
            if args.audio_loss_weight > 0.0:
                control_parameters = tuple(model.controls.parameters())
                audio_gradients = torch.autograd.grad(
                    reconstruction,
                    control_parameters,
                    allow_unused=True,
                )
                for parameter, gradient in zip(control_parameters, audio_gradients):
                    if gradient is None:
                        continue
                    contribution = gradient * args.audio_loss_weight
                    parameter.grad = (
                        contribution
                        if parameter.grad is None
                        else parameter.grad + contribution
                    )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss = supervised.detach() + args.audio_loss_weight * reconstruction.detach()
            totals["loss"] += float(loss.detach())
            totals["order"] += parts["order"]
            totals["controls"] += parts["controls"]
            totals["audio"] += float(reconstruction.detach())
            totals["batches"] += 1
            if args.smoke:
                break
        count = max(totals.pop("batches"), 1)
        report = {name: value / count for name, value in totals.items()}
        print(json.dumps({"epoch": epoch + 1, **report}, sort_keys=True))
    if not args.smoke:
        valid_real = DAFxOrderDataset(args.corpus, "valid")
        validation_samples = 32 if args.quick else 1_600
        valid_reference = SyntheticControlDataset(
            dry_sources(args.corpus, "valid"),
            validation_samples,
            seed=20260831,
            renderers=("reference",),
        )
        valid_alternate = SyntheticControlDataset(
            dry_sources(args.corpus, "valid"),
            validation_samples,
            seed=20260831,
            renderers=("alternate",),
        )
        valid_real = limited(valid_real, 32 if args.quick else 0)
        real_validation = evaluate(
            model,
            DataLoader(valid_real, batch_size=args.batch_size, shuffle=False),
            target,
        )
        reference_validation = evaluate(
            model,
            DataLoader(valid_reference, batch_size=args.batch_size, shuffle=False),
            target,
        )
        alternate_validation = evaluate(
            model,
            DataLoader(valid_alternate, batch_size=args.batch_size, shuffle=False),
            target,
        )
        validation = {
            "real_order": real_validation,
            "synthetic_reference": reference_validation,
            "synthetic_alternate": alternate_validation,
        }
        print(json.dumps({"validation": validation}, sort_keys=True))
        checkpoint = args.output / "paired-estimator.pt"
        torch.save(model.state_dict(), checkpoint)
        report = {
            "schema": 1,
            "quick": args.quick,
            "epochs": epochs,
            "train_examples": len(dataset),
            "validation_examples": len(valid_real) + len(valid_reference) + len(valid_alternate),
            "training_renderers": ["reference", "alternate"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "audio_loss_weight": args.audio_loss_weight,
            "validation": validation,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
        (args.output / "metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
