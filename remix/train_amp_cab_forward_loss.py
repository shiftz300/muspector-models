#!/usr/bin/env python3
"""Train one P1 Amp inverse with a frozen, fit-only forward replay loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from clone.evaluation.evaluate import one_example, summarize as clone_summarize

from .amp_cab_model2 import AmpCabInverseExpert2
from .amp_cab_spectral import estimate_profile_firs
from .amp_forward_model import AmpForwardReplay
from .guitar_techs_amp_data import PROFILES, RATE, SOURCE_ID
from .inverse2 import inverse_loss
from .quality2 import summarize as quality_summarize
from .train_amp_cab import _batch_to, _collate, _dynamics_penalty, _pairs, _quality


SEED = 20260925
PROFILE_INDEX = 0


def _git_revision(workspace: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_forward(path: Path) -> tuple[AmpForwardReplay, str]:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("profile_id") != PROFILES[PROFILE_INDEX].id:
        raise ValueError("forward replay checkpoint is not the P1 profile")
    architecture = checkpoint.get("architecture") or {}
    model = AmpForwardReplay(
        int(architecture.get("pre_taps", 65)),
        int(architecture.get("post_taps", 257)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, hashlib.sha256(path.read_bytes()).hexdigest()


def _combined_loss(
    model: torch.nn.Module,
    forward_model: torch.nn.Module,
    batch: dict,
    replay_weight: float,
    attack_weight: float,
    crest_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    restored, uncertainty, _ = model(batch["wet"], batch["profile"])
    loss, parts = inverse_loss(
        restored, uncertainty, batch["clean"], batch["target_start"],
    )
    attack, crest = _dynamics_penalty(
        restored, batch["clean"], batch["target_start"],
    )
    start = batch["target_start"]
    replay = forward_model(restored)
    replay_target = batch["wet"][:, start:]
    replay_prediction = replay[:, start:]
    replay_scale = replay_target.square().mean().clamp_min(1.0e-8)
    replay_error = (replay_prediction - replay_target).square().mean() / replay_scale
    loss = loss + attack_weight * attack + crest_weight * crest + replay_weight * replay_error
    parts["amp_attack_extra"] = float(attack.detach())
    parts["amp_crest"] = float(crest.detach())
    parts["forward_replay"] = float(replay_error.detach())
    return loss, parts


def _mean_loss(
    model: torch.nn.Module,
    forward_model: torch.nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
    replay_weight: float,
    attack_weight: float,
    crest_weight: float,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total = 0.0
    examples = 0
    model.eval()
    forward_model.eval()
    with torch.inference_mode():
        for raw in loader:
            batch = _batch_to(raw, device)
            value, _ = _combined_loss(
                model, forward_model, batch,
                replay_weight, attack_weight, crest_weight,
            )
            count = len(batch["wet"])
            total += float(value.detach().cpu()) * count
            examples += count
    return total / max(examples, 1)


def _replay_quality(
    model: torch.nn.Module,
    forward_model: torch.nn.Module,
    dataset,
) -> dict:
    replay_rows = []
    identity_rows = []
    model.eval()
    forward_model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            restored, _, _ = model(
                row["wet"].unsqueeze(0), row["profile"].unsqueeze(0),
            )
            replay = forward_model(restored)[0].numpy()
            direct_replay = forward_model(row["clean"].unsqueeze(0))[0].numpy()
            start = int(row["target_start"])
            target = row["wet"][start:].numpy()
            replay_rows.append(one_example(replay[start:].astype(np.float32), target))
            identity_rows.append(one_example(direct_replay[start:].astype(np.float32), target))
    return {
        "candidate_replayed_wet": clone_summarize(replay_rows),
        "clean_replayed_wet_floor": clone_summarize(identity_rows),
    }


def train(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace Amp forward-loss output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    forward_path = args.forward_checkpoint.resolve()
    if not forward_path.is_file():
        raise FileNotFoundError(forward_path)
    forward_model, forward_sha = _load_forward(forward_path)
    fit = _pairs(
        workspace, "fit", args.train_samples, args.target_frames,
        SEED + 1, PROFILE_INDEX,
    )
    calibration = _pairs(
        workspace, "calibration", args.calibration_samples, args.target_frames,
        SEED + 2, PROFILE_INDEX,
    )
    development = _pairs(
        workspace, "development", args.development_samples, args.target_frames,
        SEED + 3, PROFILE_INDEX,
    )
    device = torch.device(args.device)
    model = AmpCabInverseExpert2(args.hidden_size, args.depth)
    profile_firs, spectral_initialization = estimate_profile_firs(
        workspace, args.spectral_fit_samples,
    )
    model.profile_fir.initialize_profiles(profile_firs)
    model = model.to(device)
    forward_model = forward_model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5,
    )
    loader = DataLoader(
        fit,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        collate_fn=_collate,
    )
    initial_loss = _mean_loss(
        model, forward_model, calibration, args.batch_size, device,
        args.replay_weight, args.attack_weight, args.crest_weight,
    )
    best_loss = initial_loss
    best_epoch = 0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    history = [{"epoch": 0, "train": None, "calibration_loss": initial_loss}]
    print(json.dumps({
        "profile": PROFILES[PROFILE_INDEX].id,
        "stage": "forward_loss_inverse",
        "epoch": 0,
        "calibration_loss": initial_loss,
    }), flush=True)
    for epoch in range(args.epochs):
        model.train()
        forward_model.eval()
        totals = defaultdict(float)
        examples = 0
        for raw in loader:
            batch = _batch_to(raw, device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = _combined_loss(
                model, forward_model, batch,
                args.replay_weight, args.attack_weight, args.crest_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            count = len(batch["wet"])
            totals["loss"] += float(loss.detach().cpu()) * count
            for name, value in parts.items():
                totals[name] += value * count
            examples += count
        calibration_loss = _mean_loss(
            model, forward_model, calibration, args.batch_size, device,
            args.replay_weight, args.attack_weight, args.crest_weight,
        )
        train_report = {
            name: value / max(examples, 1)
            for name, value in sorted(totals.items())
        }
        history.append({
            "epoch": epoch + 1,
            "train": train_report,
            "calibration_loss": calibration_loss,
        })
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(json.dumps({
            "profile": PROFILES[PROFILE_INDEX].id,
            "stage": "forward_loss_inverse",
            "epoch": epoch + 1,
            "train_loss": train_report["loss"],
            "calibration_loss": calibration_loss,
            "forward_replay": train_report["forward_replay"],
        }), flush=True)
    model = model.cpu()
    model.load_state_dict(best_state)
    forward_model = forward_model.cpu()
    development_report = _quality(model, development)
    replay_report = _replay_quality(model, forward_model, development)
    checkpoint = output / "model.pt"
    manifest = model.manifest()
    torch.save({
        "schema": 1,
        "sample_rate": RATE,
        "profile_id": PROFILES[PROFILE_INDEX].id,
        "architecture": manifest,
        "state_dict": model.state_dict(),
        "forward_replay_checkpoint": str(forward_path),
        "forward_replay_sha256": forward_sha,
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 1,
        "status": "diagnostic-not-promoted",
        "accepted": False,
        "mechanism": "amp",
        "device_scope": f"Guitar-TECHS {PROFILES[PROFILE_INDEX].id} fixed Amp+cab+mic profile",
        "model": {**manifest, "checkpoint": str(checkpoint), "sha256": digest},
        "forward_replay": {
            "checkpoint": str(forward_path),
            "sha256": forward_sha,
            "loss_weight": args.replay_weight,
            "development": replay_report,
        },
        "training": {
            "accelerator": device.type,
            "model_version": 2,
            "spectral_initialization": spectral_initialization,
            "replay_weight": args.replay_weight,
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
        "data": {
            "source_id": SOURCE_ID,
            "audit": str((workspace / "runs/foundation/product3-amp-guitar-techs/corrected-pair-audit.json").resolve()),
            "license": "CC BY 4.0; product-weight authorization enforced by loader",
            "profiles": [PROFILES[PROFILE_INDEX].id],
            "p3_input_or_profile": False,
        },
        "quality": {
            "generated_audio_written": False,
            "demo_generated": False,
            "source_audio_modified": False,
            "physical_audio_devices_used": False,
            "p3_audio_opened_by_training": False,
        },
        "limitations": [
            "one fixed P1 profile only",
            "the forward replay is an imperfect diagnostic clone, not a physical model",
            "no P2 or P3 gradients, no knob interpolation, no generic Amp claim",
            "automatic development quality does not establish listening, physical veto or locked-final",
        ],
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    registry_path = workspace / "experiments.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    registry.append({
        "id": output.name,
        "git_commit": _git_revision(workspace),
        "data_manifest_sha256": hashlib.sha256(
            (workspace / "runs/foundation/product3-amp-guitar-techs/corrected-pair-audit.json").read_bytes()
        ).hexdigest(),
        "algorithm": "P1 Amp inverse with frozen causal forward-replay consistency loss",
        "fit_split": "fit",
        "calibration_split": "calibration",
        "validation_split": "development",
        "locked": False,
        "status": "diagnostic",
        "metrics": str(metrics_path),
    })
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "profile": PROFILES[PROFILE_INDEX].id,
        "selected_epoch": best_epoch,
        "development_pass_fraction": development_report["pass_fraction"],
        "replay_esr": replay_report["candidate_replayed_wet"]["absolute_esr"]["mean"],
        "clean_replay_floor_esr": replay_report["clean_replayed_wet_floor"]["absolute_esr"]["mean"],
        "checkpoint_sha256": digest,
    }, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--forward-checkpoint", type=Path,
        default=Path("runs/foundation/product3-amp-guitar-techs/amp-forward-replay-v1/forward-P1-orange-cr60-sm57.pt"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product3-amp-guitar-techs/amp-cab-forward-loss-p1-v1"),
    )
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=600)
    parser.add_argument("--calibration-samples", type=int, default=120)
    parser.add_argument("--development-samples", type=int, default=120)
    parser.add_argument("--target-frames", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--spectral-fit-samples", type=int, default=400)
    parser.add_argument("--replay-weight", type=float, default=0.25)
    parser.add_argument("--attack-weight", type=float, default=0.5)
    parser.add_argument("--crest-weight", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    args = parser.parse_args()
    if args.epochs < 1 or args.train_samples < 2 or args.calibration_samples < 2 or args.development_samples < 2:
        raise ValueError("sample counts and epochs must be positive")
    if args.replay_weight < 0.0:
        raise ValueError("replay weight must be non-negative")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    train(args)


if __name__ == "__main__":
    main()
