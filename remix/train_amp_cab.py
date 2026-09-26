#!/usr/bin/env python3
"""Train the product-eligible Guitar-TECHS P1/P2 Amp+cab+mic inverse on MPS."""

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

from .amp_cab_model import AmpCabInverseExpert
from .amp_cab_model2 import AmpCabInverseExpert2
from .amp_cab_model3 import AmpCabInverseExpert3
from .amp_cab_spectral import estimate_profile_firs
from .guitar_techs_amp_data import AmpCabPairs, PROFILES, RATE, SOURCE_ID
from .inverse2 import inverse_loss
from .quality2 import summarize


SEED = 20260922


class _SingleProfilePairs(torch.utils.data.Dataset):
    def __init__(self, base: AmpCabPairs, profile_index: int, samples: int) -> None:
        self.base = base
        self.profile_index = profile_index
        self.samples = samples
        self.authorization = base.authorization
        self.total_frames = base.total_frames
        self.expected_profile_count = 1

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.samples:
            raise IndexError(index)
        return self.base[index * len(PROFILES) + self.profile_index]


def _pairs(
    workspace: Path, split: str, samples: int, target_frames: int, seed: int,
    profile_index: int | None,
) -> AmpCabPairs | _SingleProfilePairs:
    if profile_index is None:
        return AmpCabPairs(workspace, split, samples, target_frames, seed)
    base = AmpCabPairs(
        workspace, split, samples * len(PROFILES), target_frames, seed
    )
    return _SingleProfilePairs(base, profile_index, samples)


def _collate(rows: list[dict]) -> dict:
    starts = {int(row["target_start"]) for row in rows}
    if len(starts) != 1:
        raise ValueError("one Amp-cab batch must share target geometry")
    return {
        "wet": torch.stack([row["wet"] for row in rows]),
        "clean": torch.stack([row["clean"] for row in rows]),
        "profile": torch.stack([row["profile"] for row in rows]),
        "profile_id": [row["profile_id"] for row in rows],
        "target_start": starts.pop(),
    }


def _batch_to(batch: dict, device: torch.device) -> dict:
    return {
        **batch,
        "wet": batch["wet"].to(device),
        "clean": batch["clean"].to(device),
        "profile": batch["profile"].to(device),
    }


def _dynamics_penalty(
    restored: torch.Tensor, clean: torch.Tensor, target_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    restored = restored[:, target_start:]
    clean = clean[:, target_start:]
    clean_frames = clean.unfold(1, 240, 120)
    restored_frames = restored.unfold(1, 240, 120)
    clean_rms = clean_frames.square().mean(-1).add(1.0e-10).sqrt()
    restored_rms = restored_frames.square().mean(-1).add(1.0e-10).sqrt()
    clean_attack = torch.relu(torch.diff(torch.log(clean_rms + 1.0e-6), dim=1))
    restored_attack = torch.relu(torch.diff(torch.log(restored_rms + 1.0e-6), dim=1))
    attack = torch.nn.functional.l1_loss(restored_attack, clean_attack) / clean_attack.abs().mean().clamp_min(1.0e-5)
    clean_crest = clean_frames.abs().amax(-1) / clean_rms.clamp_min(1.0e-5)
    restored_crest = restored_frames.abs().amax(-1) / restored_rms.clamp_min(1.0e-5)
    crest = torch.nn.functional.l1_loss(restored_crest, clean_crest) / clean_crest.abs().mean().clamp_min(1.0e-5)
    return attack, crest


def _mean_loss(
    model: torch.nn.Module,
    dataset: AmpCabPairs,
    batch_size: int,
    device: torch.device,
    attack_weight: float,
    crest_weight: float,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total = 0.0
    examples = 0
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            batch = _batch_to(raw, device)
            restored, uncertainty, _ = model(batch["wet"], batch["profile"])
            loss, _ = inverse_loss(restored, uncertainty, batch["clean"], batch["target_start"])
            attack, crest = _dynamics_penalty(restored, batch["clean"], batch["target_start"])
            loss = loss + attack_weight * attack + crest_weight * crest
            count = len(batch["wet"])
            total += float(loss) * count
            examples += count
    return total / max(examples, 1)


def _quality(model: torch.nn.Module, dataset: AmpCabPairs) -> dict:
    aggregate = ([], [], [])
    profiles = defaultdict(lambda: ([], [], []))
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            restored, _, _ = model(row["wet"].unsqueeze(0), row["profile"].unsqueeze(0))
            start = int(row["target_start"])
            wet = row["wet"][start:].numpy()
            candidate = restored[0, start:].numpy().astype(np.float32)
            clean = row["clean"][start:].numpy()
            for collection in (aggregate, profiles[row["profile_id"]]):
                collection[0].append(wet)
                collection[1].append(candidate)
                collection[2].append(clean)
    report = summarize("amp", *aggregate)
    report["profiles"] = {
        name: summarize("amp", *values) for name, values in sorted(profiles.items())
    }
    expected_profile_count = getattr(dataset, "expected_profile_count", len(PROFILES))
    report["all_profiles_accepted"] = bool(
        len(report["profiles"]) == expected_profile_count
        and all(row["accepted"] for row in report["profiles"].values())
    )
    return report


def _runtime(model: torch.nn.Module) -> dict:
    wet = torch.zeros(1, RATE)
    profile = torch.tensor([[1.0, 0.0]])
    model.eval()
    with torch.inference_mode():
        model(wet, profile)
        started = time.perf_counter()
        repeats = 3
        for _ in range(repeats):
            model(wet, profile)
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
        raise PermissionError("Amp-cab training requires the corrected P1/P2 audit")
    if not report.get("alignment") or not all(row.get("passed") for row in report["alignment"]):
        raise PermissionError("Amp-cab alignment audit failed")
    p3 = report.get("p3") or {}
    if p3.get("models_trained_on_p3") is not False or "contaminated" not in p3.get("status", ""):
        raise PermissionError("P3 contamination boundary is missing")
    return report


def train(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    data_audit = _require_audit(args.data_audit.resolve())
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace Amp-cab run: {output}")
    fit = _pairs(
        workspace, "fit", args.train_samples, args.target_frames, SEED + 1,
        args.profile_index,
    )
    calibration = _pairs(
        workspace, "calibration", args.calibration_samples, args.target_frames, SEED + 2,
        args.profile_index,
    )
    development = _pairs(
        workspace, "development", args.development_samples, args.target_frames, SEED + 3,
        args.profile_index,
    )
    device = torch.device(args.device)
    spectral_initialization = None
    if args.model_version in (2, 3):
        model_class = AmpCabInverseExpert2 if args.model_version == 2 else AmpCabInverseExpert3
        model = model_class(args.hidden_size, args.depth)
        profile_firs, spectral_initialization = estimate_profile_firs(
            workspace, args.spectral_fit_samples
        )
        model.profile_fir.initialize_profiles(profile_firs)
    else:
        model = AmpCabInverseExpert(args.hidden_size, args.depth)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5)
    loader = DataLoader(
        fit,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        collate_fn=_collate,
    )
    initial_loss = _mean_loss(
        model, calibration, args.batch_size, device, args.attack_weight, args.crest_weight
    )
    history = [{"epoch": 0, "train": None, "calibration_loss": initial_loss}]
    best_loss = initial_loss
    best_epoch = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    print(json.dumps({"mechanism": "amp", "profile_scope": "Amp+cab+mic", "epoch": 0, "calibration_loss": initial_loss}), flush=True)
    for epoch in range(args.epochs):
        model.train()
        totals = defaultdict(float)
        examples = 0
        for raw in loader:
            batch = _batch_to(raw, device)
            optimizer.zero_grad(set_to_none=True)
            restored, uncertainty, _ = model(batch["wet"], batch["profile"])
            loss, parts = inverse_loss(restored, uncertainty, batch["clean"], batch["target_start"])
            attack, crest = _dynamics_penalty(restored, batch["clean"], batch["target_start"])
            loss = loss + args.attack_weight * attack + args.crest_weight * crest
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
            model, calibration, args.batch_size, device, args.attack_weight, args.crest_weight
        )
        history.append({
            "epoch": epoch + 1,
            "train": {name: value / max(examples, 1) for name, value in sorted(totals.items())},
            "calibration_loss": calibration_loss,
        })
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_epoch = epoch + 1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
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
    p3_excluded = bool(
        manifest.get("p3_input_or_profile") is False
        and data_audit["p3"]["models_trained_on_p3"] is False
    )
    gates = {
        "calibration_improved": best_epoch > 0 and best_loss <= initial_loss * 0.90,
        "aggregate_quality": bool(development_report["accepted"]),
        "individual_pass_fraction": development_report["pass_fraction"] >= 0.80,
        "all_profiles": bool(development_report["all_profiles_accepted"]),
        "cpu_budget": runtime["realtime_factor"] <= 0.35,
        "order_independent": order_independent,
        "p3_excluded": p3_excluded,
    }
    accepted = bool(not args.quick and all(gates.values()))
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    torch.save({
        "schema": 1,
        "sample_rate": RATE,
        "architecture": manifest,
        "state_dict": model.state_dict(),
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 1,
        "status": "accepted-development" if accepted else "diagnostic-not-promoted",
        "accepted": accepted,
        "quick": args.quick,
        "mechanism": "amp",
        "device_scope": (
            "Guitar-TECHS P1/P2 fixed Amp+cab+mic profiles"
            if args.profile_index is None
            else f"Guitar-TECHS {PROFILES[args.profile_index].id} fixed Amp+cab+mic profile"
        ),
        "model": {**manifest, "checkpoint": str(checkpoint), "sha256": digest},
        "training": {
            "accelerator": device.type,
            "model_version": args.model_version,
            "spectral_initialization": spectral_initialization,
            "attack_weight": args.attack_weight,
            "crest_weight": args.crest_weight,
            "epochs": args.epochs,
            "selected_epoch": best_epoch,
            "initial_calibration_loss": initial_loss,
            "selected_calibration_loss": best_loss,
            "fit_samples_per_epoch": args.train_samples,
            "calibration_samples": args.calibration_samples,
            "development_samples": args.development_samples,
            "history_frames": fit.total_frames - args.target_frames,
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
            "profiles": (
                [profile.id for profile in PROFILES]
                if args.profile_index is None
                else [PROFILES[args.profile_index].id]
            ),
            "p3": data_audit["p3"],
        },
        "quality": {
            "metric_schema": 2,
            "generated_audio_written": False,
            "demo_generated": False,
            "source_audio_modified": False,
            "physical_audio_devices_used": False,
            "p3_audio_opened_by_training": False,
        },
        "limitations": [
            "two fixed named Amp+cab+mic profiles only",
            "no knob interpolation",
            "P3 is permanently contaminated and excluded",
            "development acceptance does not establish generic Amp coverage or listening acceptance",
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
        default=Path("runs/foundation/product3-amp-guitar-techs/corrected-pair-audit.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product3-amp-guitar-techs/amp-cab-v1"),
    )
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--model-version", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--spectral-fit-samples", type=int, default=400)
    parser.add_argument("--profile-index", type=int, choices=(0, 1))
    parser.add_argument("--attack-weight", type=float, default=0.5)
    parser.add_argument("--crest-weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--train-samples", type=int, default=600)
    parser.add_argument("--calibration-samples", type=int, default=100)
    parser.add_argument("--development-samples", type=int, default=100)
    parser.add_argument("--target-frames", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.epochs = 2
        args.train_samples = 60
        args.calibration_samples = 40
        args.development_samples = 40
        args.target_frames = 8192
        args.hidden_size = 16
        args.depth = 4 if args.model_version == 3 else 1
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable; run outside the sandbox")
    train(args)


if __name__ == "__main__":
    main()
