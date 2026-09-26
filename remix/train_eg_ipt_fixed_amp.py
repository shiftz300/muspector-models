#!/usr/bin/env python3
"""Train one fixed-profile physical Amp expert on EG-IPT using MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .eg_ipt_amp_data import EgIptAmpPairs, RATE
from .eg_ipt_amp_model import (
    EgIptFixedProfileAmpInverse,
    EgIptFixedProfileAmpTransientInverse,
)
from .quality2 import summarize
from .train_egdb_pg_amp import SEED, _collate, _perceptual_loss


def _loss(model, batch: dict, device: torch.device) -> tuple[torch.Tensor, dict]:
    wet = batch["wet"].to(device)
    clean = batch["clean"].to(device)
    restored, uncertainty, _ = model(wet)
    start, end = batch["crop_start"], batch["crop_end"]
    return _perceptual_loss(
        restored[:, start:end], uncertainty[:, start:end], clean[:, start:end],
        dynamic_emphasis=True,
    )


def _mean_loss(model, dataset, batch_size: int, device: torch.device) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total = examples = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            loss, _ = _loss(model, batch, device)
            count = len(batch["wet"])
            total += float(loss) * count
            examples += count
    return total / examples


def _quality(model, dataset) -> dict:
    aggregate = ([], [], [])
    pickups = defaultdict(lambda: ([], [], []))
    techniques = defaultdict(lambda: ([], [], []))
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            restored, _, _ = model(row["wet"].unsqueeze(0))
            start, end = row["crop_start"], row["crop_end"]
            values = (
                row["wet"][start:end].numpy(),
                restored[0, start:end].numpy().astype(np.float32),
                row["clean"][start:end].numpy(),
            )
            for collection in (aggregate, pickups[row["pickup"]], techniques[row["technique"]]):
                for target, value in zip(collection, values, strict=True):
                    target.append(value)
    report = summarize("amp", *aggregate)
    report["pickups"] = {
        key: summarize("amp", *value) for key, value in sorted(pickups.items())
    }
    report["techniques"] = {
        key: summarize("amp", *value) for key, value in sorted(techniques.items())
    }
    report["all_pickups_accepted"] = all(
        value["accepted"] for value in report["pickups"].values()
    )
    report["all_techniques_accepted"] = all(
        value["accepted"] for value in report["techniques"].values()
    )
    return report


def _runtime(model) -> dict:
    wet = torch.zeros(1, RATE)
    model.eval()
    with torch.inference_mode():
        model(wet)
        started = time.perf_counter()
        for _ in range(3):
            model(wet)
        elapsed = (time.perf_counter() - started) / 3
    return {"frames": RATE, "mean_seconds": elapsed, "realtime_factor": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-pretrain-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--reservation", type=Path)
    parser.add_argument("--transient", action="store_true")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=3420)
    parser.add_argument("--calibration-samples", type=int, default=342)
    parser.add_argument("--target-frames", type=int, default=16384)
    parser.add_argument("--context-frames", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace fixed-profile Amp run: {output}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    workspace = args.workspace.resolve()
    reserved = frozenset()
    reservation_report = None
    if args.reservation is not None:
        reservation_report = json.loads(args.reservation.resolve().read_text())
        if reservation_report.get("status") != "reserved-internal-v2-fit-only-not-product-validation":
            raise PermissionError("EG-IPT v2 reservation is not admitted")
        reserved = frozenset(reservation_report["relatives"])
    fit = EgIptAmpPairs(
        workspace, "fit", args.train_samples, args.target_frames,
        args.context_frames, SEED + 71 if args.transient else SEED + 61,
        exclude_relatives=reserved,
    )
    calibration = EgIptAmpPairs(
        workspace,
        reservation_report.get("source_partition", "fit") if reserved else "internal_calibration",
        args.calibration_samples,
        args.target_frames, args.context_frames, SEED + 72 if args.transient else SEED + 62,
        include_relatives=reserved or None,
    )
    if args.transient:
        if args.base_checkpoint is None or not reserved:
            raise ValueError("transient training requires --base-checkpoint and --reservation")
        model = EgIptFixedProfileAmpTransientInverse().to("cpu")
        model.load_base(args.base_checkpoint.resolve())
    else:
        references = torch.stack([fit[index]["tone_reference"] for index in range(8)])
        model = EgIptFixedProfileAmpInverse().to("cpu")
        model.load_stage_pretrain(args.stage_pretrain_checkpoint.resolve(), references)
    device = torch.device(args.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5)
    loader = DataLoader(
        fit, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED + 60), collate_fn=_collate,
    )
    initial = _mean_loss(model, calibration, args.batch_size, device)
    best, best_epoch = initial, 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history = [{"epoch": 0, "train": None, "calibration_loss": initial}]
    print(json.dumps(history[-1]), flush=True)
    for epoch in range(1, args.epochs + 1):
        totals, examples = defaultdict(float), 0
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = _loss(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            count = len(batch["wet"])
            totals["loss"] += float(loss.detach()) * count
            for name, value in parts.items():
                totals[name] += value * count
            examples += count
        calibration_loss = _mean_loss(model, calibration, args.batch_size, device)
        row = {
            "epoch": epoch,
            "train": {name: value / examples for name, value in sorted(totals.items())},
            "calibration_loss": calibration_loss,
        }
        history.append(row)
        if calibration_loss < best:
            best, best_epoch = calibration_loss, epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        print(json.dumps({"epoch": epoch, "calibration_loss": calibration_loss}), flush=True)
    model = model.cpu()
    model.load_state_dict(best_state)
    quality = _quality(model, calibration)
    runtime = _runtime(model)
    manifest = model.manifest()
    internal_gates = {
        "calibration_improved": best_epoch > 0 and best <= 0.90 * initial,
        "aggregate_quality": quality["accepted"],
        "individual_pass_fraction": quality["pass_fraction"] >= 0.80,
        "all_pickups": quality["all_pickups_accepted"],
        "all_techniques": quality["all_techniques_accepted"],
        "all_19_techniques_represented": len(quality["techniques"]) == 19,
        "cpu_budget": runtime["realtime_factor"] <= 0.35,
        "wet_only_order_independent": all(
            manifest[key] is False for key in (
                "profile_id_input", "graph_order_input", "neighbor_effect_input",
                "clean_or_oracle_input", "content_dependent_profile_estimation",
            )
        ),
    }
    output.mkdir(parents=True)
    checkpoint = output / "model.pt"
    torch.save({
        "schema": 1, "sample_rate": RATE, "architecture": manifest,
        "state_dict": model.state_dict(),
    }, checkpoint)
    report = {
        "schema": 1,
        "status": "physical-fixed-profile-internal-gates-passed-needs-independent-validation" if all(internal_gates.values()) else "physical-fixed-profile-diagnostic-not-promoted",
        "accepted": False,
        "internal_gates_passed": all(internal_gates.values()),
        "product_gate": False,
        "usable_model": None,
        "mechanism": "amp",
        "model": {
            **manifest, "checkpoint": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "training": {
            "accelerator": args.device, "epochs": args.epochs,
            "transient_refiner": args.transient,
            "base_checkpoint": None if args.base_checkpoint is None else str(args.base_checkpoint.resolve()),
            "fit_samples_per_epoch": len(fit), "calibration_samples": len(calibration),
            "selected_epoch": best_epoch, "initial_calibration_loss": initial,
            "selected_calibration_loss": best, "history": history,
        },
        "internal_calibration": quality,
        "runtime": runtime,
        "gates": internal_gates,
        "data": {
            "source_id": "eg-ipt", "role": "fit-only",
            "reservation": None if args.reservation is None else {
                "path": str(args.reservation.resolve()),
                "pairs": reservation_report["pairs"],
                "relatives_sha256": reservation_report["relatives_sha256"],
                "excluded_from_training": True,
                "not_sampled_by_v30": True,
            },
            "independent_validation_source": None,
            "physical_chain": manifest["device_profile"],
            "locked_final_audio_opened": False,
        },
        "limitations": [
            "EG-IPT internal calibration is fit-only evidence and cannot promote product weights",
            "the fixed device profile needs an independently licensed external physical-chain validation set",
            "no Demo is generated before independent validation passes",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "internal_gates_passed": report["internal_gates_passed"],
        "selected_epoch": best_epoch, "pass_fraction": quality["pass_fraction"],
        "improvements": {
            name: row["median_reduction"] for name, row in quality["metrics"].items()
        },
        "gates": internal_gates, "runtime": runtime, "output": str(output),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
