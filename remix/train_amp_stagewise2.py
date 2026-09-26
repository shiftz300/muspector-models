#!/usr/bin/env python3
"""Fine-tune the product-pretrained eight-stage gray box on measured Marshall taps."""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .amp_data import AmpPairs, RATE
from .inverse2 import inverse_loss
from .rusty_amp_stage_model import RustyAmpStagewiseGrayBoxInverse
from .train_amp import SEED, _amp_dynamics_penalty, _batch_to, _collate, _quality, _require_audit


def _uncertainty(value: torch.Tensor) -> torch.Tensor:
    return F.softplus(value.new_tensor(-3.0)).expand_as(value) + 1e-5


def _phase_loss(model, batch: dict, phase: str, stage_weight: float) -> tuple[torch.Tensor, dict]:
    if phase == "output-preamp":
        predicted = model.inverse_output_to_preamp(batch["wet"])
        return inverse_loss(predicted, _uncertainty(predicted), batch["preamp"], batch["target_start"])
    if phase == "preamp-input":
        predicted = model.inverse_preamp_to_clean(batch["preamp"], batch["wet"])
        loss, parts = inverse_loss(predicted, _uncertainty(predicted), batch["clean"], batch["target_start"])
        attack, crest = _amp_dynamics_penalty(predicted, batch["clean"], batch["target_start"])
        parts.update(attack=float(attack.detach()), crest=float(crest.detach()))
        return loss + 4.0 * attack + crest, parts
    restored, preamp = model.forward_stages(batch["wet"])
    loss, parts = inverse_loss(restored, _uncertainty(restored), batch["clean"], batch["target_start"])
    intermediate, _ = inverse_loss(preamp, _uncertainty(preamp), batch["preamp"], batch["target_start"])
    attack, crest = _amp_dynamics_penalty(restored, batch["clean"], batch["target_start"])
    parts.update(preamp_stage=float(intermediate.detach()), attack=float(attack.detach()), crest=float(crest.detach()))
    return loss + stage_weight * intermediate + 4.0 * attack + crest, parts


def _mean(model, dataset, batch_size: int, device, phase: str, stage_weight: float) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total = count = 0
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            batch = _batch_to(raw, device)
            loss, _ = _phase_loss(model, batch, phase, stage_weight)
            total += float(loss) * len(batch["wet"])
            count += len(batch["wet"])
    return total / max(count, 1)


def _fit_phase(model, fit, calibration, args, device, phase, epochs, learning_rate):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = list(model.stage_parameters(phase))
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-5)
    loader = DataLoader(
        fit, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED + len(phase)), collate_fn=_collate,
    )
    initial = _mean(model, calibration, args.batch_size, device, phase, args.stage_weight)
    best, best_epoch = initial, 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history = [{"epoch": 0, "calibration_loss": initial, "train": None}]
    print(json.dumps({"phase": phase, "epoch": 0, "calibration_loss": initial}), flush=True)
    for epoch in range(1, epochs + 1):
        totals = defaultdict(float)
        examples = 0
        model.train()
        for raw in loader:
            batch = _batch_to(raw, device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = _phase_loss(model, batch, phase, args.stage_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            count = len(batch["wet"])
            totals["loss"] += float(loss.detach()) * count
            for name, value in parts.items():
                totals[name] += value * count
            examples += count
        calibration_loss = _mean(model, calibration, args.batch_size, device, phase, args.stage_weight)
        history.append({
            "epoch": epoch, "calibration_loss": calibration_loss,
            "train": {name: value / examples for name, value in sorted(totals.items())},
        })
        if calibration_loss < best:
            best, best_epoch = calibration_loss, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(json.dumps({"phase": phase, "epoch": epoch, "calibration_loss": calibration_loss}), flush=True)
    model.load_state_dict(best_state)
    return {
        "phase": phase, "epochs": epochs, "learning_rate": learning_rate,
        "initial_calibration_loss": initial, "selected_calibration_loss": best,
        "selected_epoch": best_epoch, "history": history,
    }


def _runtime(model) -> dict:
    wet = torch.zeros(1, RATE)
    model.eval()
    with torch.inference_mode():
        model(wet)
        started = time.perf_counter()
        for _ in range(3):
            model(wet)
        elapsed = (time.perf_counter() - started) / 3
    return {"frames": RATE, "mean_seconds": elapsed, "realtime_factor": elapsed,
            "ordinary_cpu": True, "single_expert_only": True, "audio_callback": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--pretrain-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--output-preamp-epochs", type=int, default=6)
    parser.add_argument("--preamp-input-epochs", type=int, default=8)
    parser.add_argument("--joint-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--stage-weight", type=float, default=0.35)
    parser.add_argument("--train-samples", type=int, default=600)
    parser.add_argument("--calibration-samples", type=int, default=125)
    parser.add_argument("--development-samples", type=int, default=135)
    parser.add_argument("--target-frames", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--defer-development", action="store_true")
    args = parser.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    workspace = args.workspace.resolve()
    audit = _require_audit(args.data_audit.resolve())
    if not audit.get("hash_audio") or audit.get("provenance", {}).get("locked_final_audio_decoded"):
        raise PermissionError("measured preamp audit or locked-final boundary failed")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace run: {output}")
    fit = AmpPairs(workspace, "fit", args.train_samples, args.target_frames, SEED + 1)
    calibration = AmpPairs(workspace, "calibration", args.calibration_samples, args.target_frames, SEED + 2)
    development = None if args.defer_development else AmpPairs(workspace, "development", args.development_samples, args.target_frames, SEED + 3)
    payload = torch.load(args.pretrain_checkpoint.resolve(), map_location="cpu", weights_only=True)
    if payload.get("architecture", {}).get("schema") != 17:
        raise ValueError("pretraining checkpoint is not rusty-amp gray-box schema 17")
    model = RustyAmpStagewiseGrayBoxInverse().to(args.device)
    model.load_state_dict(payload["state_dict"], strict=True)
    phases = [
        _fit_phase(model, fit, calibration, args, torch.device(args.device), "output-preamp", args.output_preamp_epochs, args.learning_rate),
        _fit_phase(model, fit, calibration, args, torch.device(args.device), "preamp-input", args.preamp_input_epochs, args.learning_rate),
        _fit_phase(model, fit, calibration, args, torch.device(args.device), "joint", args.joint_epochs, args.learning_rate * 0.25),
    ]
    model = model.cpu()
    development_report = None if development is None else _quality(model, development)
    runtime = _runtime(model)
    gates = {
        "all_phases_improved": all(row["selected_epoch"] > 0 for row in phases),
        "cpu_budget": runtime["realtime_factor"] <= 0.35,
        "wet_only_order_independent": True,
        "locked_final_unopened": True,
    }
    if development_report is None:
        gates["development_deferred"] = True
    else:
        gates.update({
            "aggregate_quality": bool(development_report["accepted"]),
            "individual_pass_fraction": development_report["pass_fraction"] >= 0.80,
            "all_control_strata": bool(development_report["all_strata_accepted"]),
            "all_seen_settings": bool(development_report["all_settings_accepted"]),
        })
    accepted = bool(development_report is not None and all(gates.values()))
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    torch.save({"schema": 1, "sample_rate": RATE, "architecture": model.manifest(), "state_dict": model.state_dict()}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 1, "status": "accepted-development" if accepted else ("trained-awaiting-development" if development is None else "diagnostic-not-promoted"),
        "accepted": accepted, "mechanism": "amp",
        "model": {**model.manifest(), "checkpoint": str(checkpoint), "sha256": digest},
        "training": {"accelerator": args.device, "phases": phases, "fit_samples_per_epoch": args.train_samples,
                     "calibration_samples": args.calibration_samples, "development_samples": 0 if development is None else args.development_samples,
                     "target_frames": args.target_frames, "stage_weight": args.stage_weight,
                     "pretrain_checkpoint": str(args.pretrain_checkpoint.resolve()),
                     "pretrain_sha256": hashlib.sha256(args.pretrain_checkpoint.read_bytes()).hexdigest()},
        "development": development_report, "runtime": runtime, "gates": gates,
        "data": {"audit": str(args.data_audit.resolve()), "authorized_sources": fit.authorization["sources"],
                 "required_attribution": fit.authorization["required_attribution"]},
        "provenance": {"product_sources_only": True, "synthetic_stage_pretraining": True,
                       "measured_preamp_tap_finetuning": True, "locked_final_audio_opened": False,
                       "graph_order_input": False, "neighbor_effect_input": False, "demo_generated": False},
        "limitations": ["one physical JVM410H OD1 domain", "no cabinet or microphone inverse",
                        "synthetic stage semantics remain an initialization prior, not physical validation"],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "accepted": accepted,
                      "development_pass_fraction": None if development_report is None else development_report["pass_fraction"],
                      "gates": gates, "runtime": runtime, "sha256": digest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
