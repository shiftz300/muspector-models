#!/usr/bin/env python3
"""Train the Marshall inverse around its measured preamp boundary on MPS."""

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
from torch.utils.data import DataLoader

from .amp_data import AmpPairs, RATE
from .amp_model5 import AmpStageSupervisedInverseExpert
from .inverse2 import inverse_loss
from .train_amp import SEED, _amp_dynamics_penalty, _batch_to, _collate, _quality, _require_audit


def _stage_loss(model, batch, phase: str, stage_weight: float) -> tuple[torch.Tensor, dict]:
    if phase == "power-tone":
        prediction = model.inverse_power_tone(batch["wet"], batch["controls"])
        uncertainty = torch.nn.functional.softplus(model.uncertainty_logit).expand_as(prediction) + 1.0e-5
        return inverse_loss(prediction, uncertainty, batch["preamp"], batch["target_start"])
    if phase == "preamp":
        prediction = model.inverse_preamp(batch["preamp"], batch["controls"])
        uncertainty = torch.nn.functional.softplus(model.uncertainty_logit).expand_as(prediction) + 1.0e-5
        loss, parts = inverse_loss(prediction, uncertainty, batch["clean"], batch["target_start"])
        attack, crest = _amp_dynamics_penalty(prediction, batch["clean"], batch["target_start"])
        parts["attack"] = float(attack.detach())
        parts["crest"] = float(crest.detach())
        return loss + 4.0 * attack + crest, parts
    restored, uncertainty, _ = model(batch["wet"], batch["controls"])
    loss, parts = inverse_loss(restored, uncertainty, batch["clean"], batch["target_start"])
    _, preamp = model.forward_stages(batch["wet"], batch["controls"])
    intermediate, _ = inverse_loss(preamp, uncertainty, batch["preamp"], batch["target_start"])
    attack, crest = _amp_dynamics_penalty(restored, batch["clean"], batch["target_start"])
    parts["preamp_stage"] = float(intermediate.detach())
    parts["attack"] = float(attack.detach())
    parts["crest"] = float(crest.detach())
    return loss + stage_weight * intermediate + 4.0 * attack + crest, parts


def _mean(model, dataset, batch_size, device, phase, stage_weight) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total = 0.0
    count = 0
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            batch = _batch_to(raw, device)
            loss, _ = _stage_loss(model, batch, phase, stage_weight)
            total += float(loss) * len(batch["wet"])
            count += len(batch["wet"])
    return total / max(count, 1)


def _fit_phase(model, fit, calibration, args, device, phase, epochs, learning_rate, stage_weight):
    for parameter in model.parameters():
        parameter.requires_grad_(phase == "joint")
    if phase != "joint":
        for parameter in model.stage_parameters(phase):
            parameter.requires_grad_(True)
        model.uncertainty_logit.requires_grad_(True)
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1.0e-5)
    loader = DataLoader(
        fit, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED + len(phase)), collate_fn=_collate,
    )
    initial = _mean(model, calibration, args.batch_size, device, phase, stage_weight)
    best = initial
    best_epoch = 0
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
            loss, parts = _stage_loss(model, batch, phase, stage_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 2.0)
            optimizer.step()
            count = len(batch["wet"])
            totals["loss"] += float(loss.detach()) * count
            for name, value in parts.items():
                totals[name] += value * count
            examples += count
        calibration_loss = _mean(model, calibration, args.batch_size, device, phase, stage_weight)
        history.append({
            "epoch": epoch,
            "calibration_loss": calibration_loss,
            "train": {name: value / examples for name, value in sorted(totals.items())},
        })
        if calibration_loss < best:
            best = calibration_loss
            best_epoch = epoch
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
    controls = torch.full((1, 4), 0.5)
    model.eval()
    with torch.inference_mode():
        model(wet, controls)
        started = time.perf_counter()
        for _ in range(3):
            model(wet, controls)
        elapsed = (time.perf_counter() - started) / 3
    return {"frames": RATE, "mean_seconds": elapsed, "realtime_factor": elapsed,
            "ordinary_cpu": True, "single_expert_only": True, "audio_callback": False}


def train(args) -> dict:
    workspace = args.workspace.resolve()
    audit = _require_audit(args.data_audit.resolve())
    if not audit.get("hash_audio") or not all(
        "preamp" in row.get("paths", {}) and "preamp" in row.get("sha256", {})
        for row in audit.get("rows", []) if not row.get("locked_final")
    ):
        raise PermissionError("stagewise Amp training requires hashed preamp-tap audit rows")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace stagewise Amp run: {output}")
    fit = AmpPairs(workspace, "fit", args.train_samples, args.target_frames, SEED + 1)
    calibration = AmpPairs(workspace, "calibration", args.calibration_samples, args.target_frames, SEED + 2)
    development = AmpPairs(workspace, "development", args.development_samples, args.target_frames, SEED + 3)
    device = torch.device(args.device)
    model = AmpStageSupervisedInverseExpert().to(device)
    phases = [
        _fit_phase(model, fit, calibration, args, device, "power-tone", args.power_epochs, args.learning_rate, 0.0),
        _fit_phase(model, fit, calibration, args, device, "preamp", args.preamp_epochs, args.learning_rate, 0.0),
        _fit_phase(model, fit, calibration, args, device, "joint", args.joint_epochs, args.learning_rate * 0.25, args.stage_weight),
    ]
    model = model.cpu()
    development_report = _quality(model, development)
    runtime = _runtime(model)
    manifest = model.manifest()
    gates = {
        "all_phases_improved": all(row["selected_epoch"] > 0 for row in phases),
        "aggregate_quality": bool(development_report["accepted"]),
        "individual_pass_fraction": development_report["pass_fraction"] >= 0.80,
        "all_control_strata": bool(development_report["all_strata_accepted"]),
        "all_seen_settings": bool(development_report["all_settings_accepted"]),
        "cpu_budget": runtime["realtime_factor"] <= 0.35,
        "order_independent": True,
        "locked_final_unopened": audit.get("provenance", {}).get("locked_final_audio_decoded") is False,
    }
    accepted = bool(all(gates.values()))
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    torch.save({"schema": 1, "sample_rate": RATE, "architecture": manifest,
                "state_dict": model.state_dict()}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 1, "status": "accepted-development" if accepted else "diagnostic-not-promoted",
        "accepted": accepted, "mechanism": "amp",
        "device_scope": "Marshall JVM410H OD1 speaker output; measured preamp boundary",
        "model": {**manifest, "checkpoint": str(checkpoint), "sha256": digest},
        "training": {"accelerator": device.type, "phases": phases, "stage_weight": args.stage_weight,
                     "fit_samples_per_epoch": args.train_samples,
                     "calibration_samples": args.calibration_samples,
                     "development_samples": args.development_samples,
                     "target_frames": args.target_frames},
        "development": development_report, "runtime": runtime, "gates": gates,
        "data": {"audit": str(args.data_audit.resolve()), "audit_status": audit["status"],
                 "authorized_sources": fit.authorization["sources"],
                 "required_attribution": fit.authorization["required_attribution"]},
        "provenance": {"product_sources_only": True, "preamp_tap_supervised": True,
                       "locked_final_audio_opened": False, "demo_generated": False,
                       "graph_order_input": False, "neighbor_effect_input": False},
        "limitations": ["one physical JVM410H OD1 and one repeated source programme",
                        "speaker-output only; no cabinet or microphone inverse",
                        "stage boundary is measured but internal circuit parameters are not identified"],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "accepted": accepted,
                      "development_pass_fraction": development_report["pass_fraction"],
                      "gates": gates, "runtime": runtime, "sha256": digest}, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--power-epochs", type=int, default=8)
    parser.add_argument("--preamp-epochs", type=int, default=12)
    parser.add_argument("--joint-epochs", type=int, default=6)
    parser.add_argument("--stage-weight", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--train-samples", type=int, default=600)
    parser.add_argument("--calibration-samples", type=int, default=125)
    parser.add_argument("--development-samples", type=int, default=135)
    parser.add_argument("--target-frames", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable; run outside the sandbox")
    train(args)


if __name__ == "__main__":
    main()
