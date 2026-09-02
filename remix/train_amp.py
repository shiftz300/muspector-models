#!/usr/bin/env python3
"""Train the product-eligible order-independent Marshall Amp inverse on MPS."""

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

from .amp_data import AmpPairs, EXPECTED_DEVELOPMENT_SETTINGS, RATE, SOURCE_ID
from .amp_model import AmpInverseExpert
from .amp_model2 import AmpDynamicsInverseExpert
from .amp_model3 import AmpFrameDynamicsInverseExpert
from .inverse2 import inverse_loss
from .quality2 import summarize


SEED = 20260921


def _collate(rows: list[dict]) -> dict:
    starts = {int(row["target_start"]) for row in rows}
    if len(starts) != 1:
        raise ValueError("one Amp batch must share target geometry")
    return {
        "wet": torch.stack([row["wet"] for row in rows]),
        "clean": torch.stack([row["clean"] for row in rows]),
        "controls": torch.stack([row["controls"] for row in rows]),
        "target_start": starts.pop(),
    }


def _batch_to(batch: dict, device: torch.device) -> dict:
    return {
        **batch,
        "wet": batch["wet"].to(device),
        "clean": batch["clean"].to(device),
        "controls": batch["controls"].to(device),
    }


def _amp_dynamics_penalty(
    restored: torch.Tensor,
    clean: torch.Tensor,
    target_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    restored = restored[:, target_start:]
    clean = clean[:, target_start:]
    clean_attack_frames = clean.unfold(1, 240, 120)
    restored_attack_frames = restored.unfold(1, 240, 120)
    clean_rms = clean_attack_frames.square().mean(dim=-1).add(1.0e-10).sqrt()
    restored_rms = restored_attack_frames.square().mean(dim=-1).add(1.0e-10).sqrt()
    clean_attack = torch.relu(torch.diff(torch.log(clean_rms + 1.0e-6), dim=1))
    restored_attack = torch.relu(torch.diff(torch.log(restored_rms + 1.0e-6), dim=1))
    attack = torch.nn.functional.l1_loss(restored_attack, clean_attack) / clean_attack.abs().mean().clamp_min(1.0e-5)
    clean_crest_frames = clean.unfold(1, 1024, 256)
    restored_crest_frames = restored.unfold(1, 1024, 256)
    clean_crest = clean_crest_frames.abs().amax(dim=-1) / clean_crest_frames.square().mean(dim=-1).add(1.0e-10).sqrt()
    restored_crest = restored_crest_frames.abs().amax(dim=-1) / restored_crest_frames.square().mean(dim=-1).add(1.0e-10).sqrt()
    crest = torch.nn.functional.l1_loss(restored_crest, clean_crest) / clean_crest.abs().mean().clamp_min(1.0e-5)
    return attack, crest


def _mean_loss(
    model: torch.nn.Module,
    dataset: AmpPairs,
    batch_size: int,
    device: torch.device,
    attack_extra_weight: float,
    crest_weight: float,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total = 0.0
    examples = 0
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            batch = _batch_to(raw, device)
            restored, uncertainty, _ = model(batch["wet"], batch["controls"])
            loss, _ = inverse_loss(restored, uncertainty, batch["clean"], batch["target_start"])
            attack, crest = _amp_dynamics_penalty(restored, batch["clean"], batch["target_start"])
            loss = loss + attack_extra_weight * attack + crest_weight * crest
            count = len(batch["wet"])
            total += float(loss) * count
            examples += count
    return total / max(examples, 1)


def _stratum(values: dict) -> str:
    names = ("bass", "mid", "treble", "gain")
    controls = np.asarray([values[name] for name in names], dtype=np.float64)
    distance = np.abs(controls - 5.0)
    if float(np.max(distance)) <= 0.5:
        return "center"
    return names[int(np.argmax(distance))]


def _quality(model: torch.nn.Module, dataset: AmpPairs) -> dict:
    aggregate = ([], [], [])
    strata = defaultdict(lambda: ([], [], []))
    settings = defaultdict(lambda: ([], [], []))
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            restored, _, _ = model(row["wet"].unsqueeze(0), row["controls"].unsqueeze(0))
            start = int(row["target_start"])
            wet = row["wet"][start:].numpy()
            candidate = restored[0, start:].numpy().astype(np.float32)
            clean = row["clean"][start:].numpy()
            for collection in (
                aggregate,
                strata[_stratum(row["control_values"])],
                settings[row["setting"]],
            ):
                collection[0].append(wet)
                collection[1].append(candidate)
                collection[2].append(clean)
    report = summarize("amp", *aggregate)
    report["strata"] = {name: summarize("amp", *values) for name, values in sorted(strata.items())}
    report["settings"] = {name: summarize("amp", *values) for name, values in sorted(settings.items())}
    report["all_strata_accepted"] = bool(
        report["strata"] and all(row["accepted"] for row in report["strata"].values())
    )
    report["all_settings_accepted"] = bool(
        len(report["settings"]) == EXPECTED_DEVELOPMENT_SETTINGS
        and all(row["accepted"] for row in report["settings"].values())
    )
    return report


def _runtime(model: torch.nn.Module) -> dict:
    wet = torch.zeros(1, RATE)
    controls = torch.full((1, 4), 0.5)
    model.eval()
    with torch.inference_mode():
        model(wet, controls)
        started = time.perf_counter()
        repeats = 3
        for _ in range(repeats):
            model(wet, controls)
        elapsed = (time.perf_counter() - started) / repeats
    return {
        "frames": RATE,
        "mean_seconds": elapsed,
        "realtime_factor": elapsed,
        "ordinary_cpu": True,
        "single_expert_only": True,
        "audio_callback": False,
    }


def _require_audit(path: Path) -> dict:
    report = json.loads(path.read_text())
    if report.get("source_id") != SOURCE_ID or report.get("status") != "audited-product-eligible-not-trained":
        raise PermissionError("Amp training requires the decoded seen-setting data audit")
    alignment = report.get("seen_alignment") or {}
    signal_quality = report.get("seen_signal_quality") or {}
    if (
        not alignment.get("passed")
        or not signal_quality.get("passed")
        or report.get("provenance", {}).get("locked_final_audio_decoded")
    ):
        raise PermissionError("Amp alignment or locked-final boundary failed")
    return report


def train(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    data_audit = _require_audit(args.data_audit.resolve())
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace Amp run: {output}")
    fit = AmpPairs(workspace, "fit", args.train_samples, args.target_frames, SEED + 1)
    calibration = AmpPairs(workspace, "calibration", args.calibration_samples, args.target_frames, SEED + 2)
    development = AmpPairs(workspace, "development", args.development_samples, args.target_frames, SEED + 3)
    device = torch.device(args.device)
    model_class = {
        1: AmpInverseExpert,
        2: AmpDynamicsInverseExpert,
        3: AmpFrameDynamicsInverseExpert,
    }[args.model_version]
    model = model_class(args.hidden_size, args.depth).to(device)
    warm_start = None
    if args.warm_start_profile is not None:
        payload = torch.load(args.warm_start_profile.resolve(), map_location="cpu", weights_only=True)
        state = payload.get("state_dict", {})
        profile_state = {
            name.removeprefix("profile."): value
            for name, value in state.items()
            if name.startswith("profile.")
        }
        if not profile_state:
            raise ValueError("Amp warm start has no profile FIR state")
        model.profile.load_state_dict(profile_state, strict=True)
        warm_start = {
            "checkpoint": str(args.warm_start_profile.resolve()),
            "sha256": hashlib.sha256(args.warm_start_profile.read_bytes()).hexdigest(),
            "loaded_scope": "profile-fir-only",
        }
    if not 0 <= args.freeze_profile_epochs < args.epochs:
        raise ValueError("freeze-profile-epochs must be in 0..epochs-1")
    if args.freeze_profile_epochs:
        for parameter in model.profile.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5)
    loader = DataLoader(
        fit,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        collate_fn=_collate,
    )
    initial_loss = _mean_loss(
        model, calibration, args.batch_size, device,
        args.attack_extra_weight, args.crest_weight,
    )
    history = [{"epoch": 0, "train": None, "calibration_loss": initial_loss}]
    best_loss = initial_loss
    best_epoch = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    print(json.dumps({"mechanism": "amp", "epoch": 0, "calibration_loss": initial_loss}), flush=True)
    for epoch in range(args.epochs):
        if epoch == args.freeze_profile_epochs and args.freeze_profile_epochs:
            for parameter in model.profile.parameters():
                parameter.requires_grad_(True)
        model.train()
        totals = defaultdict(float)
        examples = 0
        for raw in loader:
            batch = _batch_to(raw, device)
            optimizer.zero_grad(set_to_none=True)
            restored, uncertainty, _ = model(batch["wet"], batch["controls"])
            loss, parts = inverse_loss(
                restored, uncertainty, batch["clean"], batch["target_start"]
            )
            attack, crest = _amp_dynamics_penalty(restored, batch["clean"], batch["target_start"])
            loss = loss + args.attack_extra_weight * attack + args.crest_weight * crest
            parts["amp_attack_extra"] = float(attack.detach())
            parts["amp_crest"] = float(crest.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            count = len(batch["wet"])
            totals["loss"] += float(loss.detach()) * count
            for name, value in parts.items():
                totals[name] += value * count
            examples += count
        calibration_loss = _mean_loss(
            model, calibration, args.batch_size, device,
            args.attack_extra_weight, args.crest_weight,
        )
        history.append({
            "epoch": epoch + 1,
            "train": {name: value / max(examples, 1) for name, value in sorted(totals.items())},
            "calibration_loss": calibration_loss,
        })
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        print(json.dumps({"mechanism": "amp", "epoch": epoch + 1, "calibration_loss": calibration_loss}), flush=True)
    model = model.cpu()
    model.load_state_dict(best_state)
    development_report = _quality(model, development)
    runtime = _runtime(model)
    manifest = model.manifest()
    order_independent = all(
        manifest.get(name) is False
        for name in ("graph_order_input", "neighbor_effect_input", "recurrent_state_input")
    )
    locked_final_unopened = bool(
        data_audit.get("pairs", {}).get("locked_final") == 9
        and data_audit.get("provenance", {}).get("locked_final_audio_decoded") is False
    )
    gates = {
        "calibration_improved": best_epoch > 0 and best_loss <= initial_loss * 0.90,
        "aggregate_quality": bool(development_report["accepted"]),
        "individual_pass_fraction": development_report["pass_fraction"] >= 0.80,
        "all_control_strata": bool(development_report["all_strata_accepted"]),
        "all_seen_settings": bool(development_report["all_settings_accepted"]),
        "cpu_budget": runtime["realtime_factor"] <= 0.35,
        "order_independent": order_independent,
        "locked_final_unopened": locked_final_unopened,
    }
    accepted = bool(not args.quick and all(gates.values()))
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    torch.save({
        "schema": 1,
        "sample_rate": RATE,
        "architecture": model.manifest(),
        "state_dict": model.state_dict(),
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 1,
        "status": "accepted-development" if accepted else "diagnostic-not-promoted",
        "accepted": accepted,
        "quick": args.quick,
        "mechanism": "amp",
        "device_scope": "Marshall JVM410H OD1; B/M/T and Gain variable",
        "model": {**manifest, "checkpoint": str(checkpoint), "sha256": digest},
        "training": {
            "accelerator": device.type,
            "model_version": args.model_version,
            "warm_start": warm_start,
            "profile_frozen_epochs": args.freeze_profile_epochs,
            "attack_extra_weight": args.attack_extra_weight,
            "crest_weight": args.crest_weight,
            "epochs": args.epochs,
            "selected_epoch": best_epoch,
            "initial_calibration_loss": initial_loss,
            "selected_calibration_loss": best_loss,
            "fit_samples_per_epoch": args.train_samples,
            "calibration_samples": args.calibration_samples,
            "development_samples": args.development_samples,
            "history_frames": fit.history_frames,
            "target_frames": args.target_frames,
            "history": history,
        },
        "development": development_report,
        "runtime": runtime,
        "gates": gates,
        "data": {
            "audit": str(args.data_audit.resolve()),
            "audit_status": data_audit["status"],
            "authorized_sources": fit.authorization["sources"],
            "required_attribution": fit.authorization["required_attribution"],
            "fit_setting_counts": fit.realized_setting_counts(),
            "calibration_setting_counts": calibration.realized_setting_counts(),
            "development_setting_counts": development.realized_setting_counts(),
        },
        "quality": {
            "metric_schema": 2,
            "generated_audio_written": False,
            "demo_generated": False,
            "source_audio_modified": False,
            "physical_audio_devices_used": False,
            "locked_final_audio_opened": not locked_final_unopened,
        },
        "limitations": [
            "one physical Marshall JVM410H OD1 device and one repeated source program",
            "Gain=1 and Gain=8 are development-only interpolation settings",
            "Gain=6 is excluded because its directory and member filenames contradict",
            "speaker-output only; no cabinet or microphone inverse",
            "locked-final unseen controls remain unopened",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "accepted": accepted,
        "selected_epoch": best_epoch,
        "development_pass_fraction": development_report["pass_fraction"],
        "runtime": runtime,
        "gates": gates,
        "sha256": digest,
    }, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--data-audit", type=Path,
        default=Path("runs/foundation/product3-amp-marshall/data-audit.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product3-amp-marshall/amp"),
    )
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--model-version", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--warm-start-profile", type=Path)
    parser.add_argument("--freeze-profile-epochs", type=int, default=0)
    parser.add_argument("--attack-extra-weight", type=float, default=0.0)
    parser.add_argument("--crest-weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train-samples", type=int, default=600)
    parser.add_argument("--calibration-samples", type=int, default=125)
    parser.add_argument("--development-samples", type=int, default=135)
    parser.add_argument("--target-frames", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.epochs = 2
        args.train_samples = 75
        args.calibration_samples = 50
        args.development_samples = 54
        args.target_frames = 8192
        if args.model_version == 3:
            args.hidden_size = 24
            args.depth = 1
        else:
            args.hidden_size = 12
            args.depth = 5
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable; run outside the sandbox")
    train(args)


if __name__ == "__main__":
    main()
