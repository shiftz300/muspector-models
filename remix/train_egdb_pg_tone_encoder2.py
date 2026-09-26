#!/usr/bin/env python3
"""Train a Wet-only tone embedding with profile and gain-structure objectives."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .egdb_pg_amp_data import CATEGORIES, RATE
from .egdb_pg_tone_encoder import WetToneEncoder
from .train_egdb_pg_tone_encoder import (
    CONTEXT_FRAMES,
    OBSERVATION_FRAMES,
    TonePairs,
    _contrastive,
    _retrieval,
)


SEED = 20260908
CATEGORY_INDEX = {name: index for index, name in enumerate(CATEGORIES)}


def _collate(rows: list[dict]) -> dict:
    profile_names = sorted({row["profile_id"] for row in rows})
    profile_index = {name: value for value, name in enumerate(profile_names)}
    profile_labels = torch.tensor(
        [profile_index[row["profile_id"]] for row in rows], dtype=torch.long
    )
    category_labels = torch.tensor(
        [CATEGORY_INDEX[row["category"]] for row in rows], dtype=torch.long
    )
    return {
        "wet": torch.cat((
            torch.stack([row["left"] for row in rows]),
            torch.stack([row["right"] for row in rows]),
        )),
        "profile_labels": torch.cat((profile_labels, profile_labels)),
        "category_labels": torch.cat((category_labels, category_labels)),
        "profile_id": [row["profile_id"] for row in rows] * 2,
    }


def _loss(model, head, batch, device, temperature, margin, profile_weight):
    embedding = model(batch["wet"].to(device))
    profile = _contrastive(
        embedding, batch["profile_labels"].to(device), temperature, margin
    )
    category = torch.nn.functional.cross_entropy(
        head(embedding), batch["category_labels"].to(device)
    )
    return profile_weight * profile + category, profile, category


def _evaluate(model, head, dataset, batch_size, device, temperature, margin, profile_weight):
    total, profile, category, correct, views = [], [], [], 0, 0
    model.eval()
    head.eval()
    with torch.inference_mode():
        for batch in DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate
        ):
            loss, profile_loss, category_loss = _loss(
                model, head, batch, device, temperature, margin, profile_weight
            )
            labels = batch["category_labels"].to(device)
            prediction = head(model(batch["wet"].to(device))).argmax(1)
            total.append(float(loss))
            profile.append(float(profile_loss))
            category.append(float(category_loss))
            correct += int((prediction == labels).sum())
            views += len(labels)
    return {
        "loss": float(np.mean(total)),
        "profile_loss": float(np.mean(profile)),
        "category_loss": float(np.mean(category)),
        "category_accuracy": correct / max(views, 1),
        "views": views,
    }


def train(args) -> dict:
    workspace, output = args.workspace.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace tone run: {output}")
    fit_primary = TonePairs(
        workspace, "fit", args.fit_pairs_per_profile, SEED + 1
    )
    fit_consumed = TonePairs(
        workspace, "fresh_validation", args.fit_pairs_per_profile, SEED + 2
    )
    fit = ConcatDataset((fit_primary, fit_consumed))
    calibration = TonePairs(
        workspace, "calibration", args.calibration_pairs_per_profile, SEED + 3
    )
    development = TonePairs(
        workspace, "development", args.development_pairs_per_profile, SEED + 4
    )
    device = torch.device(args.device)
    model = WetToneEncoder(args.embedding_size).to(device)
    head = torch.nn.Linear(args.embedding_size, len(CATEGORIES)).to(device)
    optimizer = torch.optim.AdamW(
        (*model.parameters(), *head.parameters()),
        lr=args.learning_rate,
        weight_decay=1.0e-4,
    )
    initial = _evaluate(
        model, head, calibration, args.batch_size, device,
        args.temperature, args.margin, args.profile_weight,
    )
    best, best_epoch = initial["loss"], 0
    best_model = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    best_head = {
        name: value.detach().cpu().clone() for name, value in head.state_dict().items()
    }
    history = [{"epoch": 0, "calibration": initial}]
    print(json.dumps(history[-1]), flush=True)
    loader = DataLoader(
        fit, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED), collate_fn=_collate,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        train_losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = _loss(
                model, head, batch, device,
                args.temperature, args.margin, args.profile_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (*model.parameters(), *head.parameters()), 1.0
            )
            optimizer.step()
            train_losses.append(float(loss.detach()))
        calibration_report = _evaluate(
            model, head, calibration, args.batch_size, device,
            args.temperature, args.margin, args.profile_weight,
        )
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "calibration": calibration_report,
        })
        if calibration_report["loss"] < best:
            best, best_epoch = calibration_report["loss"], epoch
            best_model = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_head = {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
        print(json.dumps(history[-1]), flush=True)
    model = model.cpu()
    head = head.cpu()
    model.load_state_dict(best_model)
    head.load_state_dict(best_head)
    calibration_report = _evaluate(
        model, head, calibration, args.batch_size, torch.device("cpu"),
        args.temperature, args.margin, args.profile_weight,
    )
    development_report = _evaluate(
        model, head, development, args.batch_size, torch.device("cpu"),
        args.temperature, args.margin, args.profile_weight,
    )
    retrieval = _retrieval(model, calibration, args.batch_size)
    sample = torch.zeros(1, OBSERVATION_FRAMES)
    with torch.inference_mode():
        model(sample)
        started = time.perf_counter()
        for _ in range(3):
            model(sample)
        elapsed = (time.perf_counter() - started) / 3
    runtime = {
        "observation_seconds": 3.0,
        "mean_seconds": elapsed,
        "realtime_factor": elapsed / 3.0,
    }
    gates = {
        "calibration_improved": best_epoch > 0 and best <= initial["loss"] * 0.75,
        "development_category_accuracy": development_report["category_accuracy"] >= 0.80,
        "content_invariant_retrieval": retrieval["leave_one_view_out_top1"] >= 0.60,
        "cpu_budget": runtime["realtime_factor"] <= 0.10,
        "wet_only": model.manifest()["profile_id_input"] is False,
        "locked_final_unopened": calibration.base.audit["locked_final_downloaded"] is False,
        "fresh_validation_v2_unopened": not calibration.base.audit.get(
            "fresh_validation_v2_downloaded", False
        ),
    }
    accepted = all(gates.values())
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    architecture = {
        **model.manifest(),
        "training_objective": "content-invariant profile contrast plus gain-category supervision",
        "category_label_required_at_inference": False,
    }
    torch.save({
        "schema": 1,
        "architecture": architecture,
        "state_dict": model.state_dict(),
        "category_head_state_dict": head.state_dict(),
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 1,
        "status": "accepted-category-aware-tone-representation" if accepted else "diagnostic-category-aware-tone-representation",
        "accepted": accepted,
        "not_a_restoration_model": True,
        "model": {**architecture, "checkpoint": str(checkpoint), "sha256": digest},
        "training": {
            "accelerator": device.type,
            "epochs": args.epochs,
            "selected_epoch": best_epoch,
            "profile_weight": args.profile_weight,
            "history": history,
            "fit_profiles": fit_primary.profile_count + fit_consumed.profile_count,
            "consumed_fresh_v1_used_for_fit": True,
        },
        "calibration": calibration_report,
        "development": development_report,
        "calibration_retrieval": retrieval,
        "runtime": runtime,
        "gates": gates,
        "data": {
            "authorized_sources": fit_primary.authorization["sources"],
            "required_attribution": fit_primary.authorization["required_attribution"],
            "development_audio_opened": True,
            "fresh_validation_v2_audio_opened": False,
            "locked_final_audio_opened": False,
        },
        "limitations": [
            "representation acceptance does not accept an inverse model",
            "gain category is an auxiliary training label and is not an inference input",
            "fresh validation v2 remains sealed for the downstream restoration model",
        ],
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": report["status"],
        "accepted": accepted,
        "selected_epoch": best_epoch,
        "development_category_accuracy": development_report["category_accuracy"],
        "retrieval": retrieval,
        "gates": gates,
        "sha256": digest,
    }, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--fit-pairs-per-profile", type=int, default=8)
    parser.add_argument("--calibration-pairs-per-profile", type=int, default=4)
    parser.add_argument("--development-pairs-per-profile", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--profile-weight", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
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
