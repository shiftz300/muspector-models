#!/usr/bin/env python3
"""Clean-assisted profile oracle for a frozen EGDB-PG gray-box processor.

This is a diagnostic upper bound only. It fits one latent condition on support
tracks from the already revealed development profiles, evaluates disjoint query
tracks, discards the conditions, and can never promote a product model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .egdb_pg_amp_data import CATEGORIES, EgdbPgAmpPairs, RATE
from .egdb_pg_amp_model10 import WetTonePhaseGrayBoxAmpCabInverse
from .quality2 import summarize
from .train_egdb_pg_amp import SEED, _perceptual_loss


def _batch(rows: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    wet = torch.stack([row["wet"] for row in rows]).to(device)
    clean = torch.stack([row["clean"] for row in rows]).to(device)
    return wet, clean


def _initial_condition(
    model: WetTonePhaseGrayBoxAmpCabInverse,
    rows: list[dict],
    device: torch.device,
) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for row in rows:
            reference = row["tone_reference"].unsqueeze(0).to(device)
            values.append(model.condition(model.tone_encoder(reference))[0])
    mean = torch.stack(values).mean(0).clamp(-0.999, 0.999)
    return torch.atanh(mean).detach()


def _fit_condition(
    model: WetTonePhaseGrayBoxAmpCabInverse,
    rows: list[dict],
    device: torch.device,
    steps: int,
    learning_rate: float,
    batch_size: int,
) -> tuple[torch.Tensor, dict]:
    raw = torch.nn.Parameter(_initial_condition(model, rows, device))
    optimizer = torch.optim.Adam((raw,), lr=learning_rate)
    best_loss = float("inf")
    best_raw = raw.detach().clone()
    history = []
    for step in range(steps + 1):
        total = 0.0
        examples = 0
        if step:
            optimizer.zero_grad(set_to_none=True)
        for offset in range(0, len(rows), batch_size):
            selected = rows[offset:offset + batch_size]
            wet, clean = _batch(selected, device)
            condition = torch.tanh(raw).unsqueeze(0).expand(len(selected), -1)
            restored = model.forward_with_condition(wet, condition)
            uncertainty = F.softplus(model.uncertainty_logit).expand_as(restored) + 1.0e-5
            start, end = selected[0]["crop_start"], selected[0]["crop_end"]
            loss, _ = _perceptual_loss(
                restored[:, start:end],
                uncertainty[:, start:end],
                clean[:, start:end],
                dynamic_emphasis=True,
            )
            if step:
                (loss * len(selected) / len(rows)).backward()
            total += float(loss.detach()) * len(selected)
            examples += len(selected)
        mean = total / examples
        history.append({"step": step, "support_loss": mean})
        if mean < best_loss:
            best_loss = mean
            best_raw = raw.detach().clone()
        if step:
            torch.nn.utils.clip_grad_norm_((raw,), 1.0)
            optimizer.step()
    return torch.tanh(best_raw).detach(), {
        "initial_support_loss": history[0]["support_loss"],
        "best_support_loss": best_loss,
        "best_step": min(history, key=lambda row: row["support_loss"])["step"],
        "history": history,
    }


def _quality(
    model: WetTonePhaseGrayBoxAmpCabInverse,
    rows: list[dict],
    conditions: dict[str, torch.Tensor] | None,
    device: torch.device,
) -> dict:
    aggregate = ([], [], [])
    categories = defaultdict(lambda: ([], [], []))
    profiles = defaultdict(lambda: ([], [], []))
    model.eval()
    with torch.inference_mode():
        for row in rows:
            wet = row["wet"].unsqueeze(0).to(device)
            if conditions is None:
                restored, _, _ = model(
                    wet,
                    tone_reference=row["tone_reference"].unsqueeze(0).to(device),
                )
            else:
                restored = model.forward_with_condition(
                    wet, conditions[row["profile_id"]].unsqueeze(0)
                )
            start, end = row["crop_start"], row["crop_end"]
            values = (
                row["wet"][start:end].numpy(),
                restored[0, start:end].cpu().numpy().astype(np.float32),
                row["clean"][start:end].numpy(),
            )
            for collection in (
                aggregate,
                categories[row["category"]],
                profiles[row["profile_id"]],
            ):
                for target, value in zip(collection, values, strict=True):
                    target.append(value)
    report = summarize("amp", *aggregate)
    report["categories"] = {
        name: summarize("amp", *values) for name, values in sorted(categories.items())
    }
    report["profiles"] = {
        name: summarize("amp", *values) for name, values in sorted(profiles.items())
    }
    return report


def _fit_example_conditions(
    model: WetTonePhaseGrayBoxAmpCabInverse,
    rows: list[dict],
    device: torch.device,
    steps: int,
    learning_rate: float,
    batch_size: int,
) -> tuple[torch.Tensor, dict]:
    initial = []
    with torch.no_grad():
        for row in rows:
            reference = row["tone_reference"].unsqueeze(0).to(device)
            condition = model.condition(model.tone_encoder(reference))[0]
            initial.append(torch.atanh(condition.clamp(-0.999, 0.999)))
    raw = torch.nn.Parameter(torch.stack(initial))
    optimizer = torch.optim.Adam((raw,), lr=learning_rate)

    def mean_loss(backward: bool) -> float:
        total = 0.0
        for offset in range(0, len(rows), batch_size):
            selected = rows[offset:offset + batch_size]
            wet, clean = _batch(selected, device)
            condition = torch.tanh(raw[offset:offset + len(selected)])
            restored = model.forward_with_condition(wet, condition)
            uncertainty = F.softplus(model.uncertainty_logit).expand_as(restored) + 1.0e-5
            start, end = selected[0]["crop_start"], selected[0]["crop_end"]
            loss, _ = _perceptual_loss(
                restored[:, start:end],
                uncertainty[:, start:end],
                clean[:, start:end],
                dynamic_emphasis=True,
            )
            if backward:
                (loss * len(selected) / len(rows)).backward()
            total += float(loss.detach()) * len(selected)
        return total / len(rows)

    initial_loss = mean_loss(False)
    best_loss = initial_loss
    best_raw = raw.detach().clone()
    best_step = 0
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        mean_loss(True)
        torch.nn.utils.clip_grad_norm_((raw,), 1.0)
        optimizer.step()
        current = mean_loss(False)
        if current < best_loss:
            best_loss = current
            best_raw = raw.detach().clone()
            best_step = step
    return torch.tanh(best_raw).detach(), {
        "initial_self_loss": initial_loss,
        "best_self_loss": best_loss,
        "best_step": best_step,
    }


def _quality_example_conditions(
    model: WetTonePhaseGrayBoxAmpCabInverse,
    rows: list[dict],
    conditions: torch.Tensor,
    device: torch.device,
) -> dict:
    aggregate = ([], [], [])
    categories = defaultdict(lambda: ([], [], []))
    profiles = defaultdict(lambda: ([], [], []))
    model.eval()
    with torch.inference_mode():
        for index, row in enumerate(rows):
            wet = row["wet"].unsqueeze(0).to(device)
            restored = model.forward_with_condition(wet, conditions[index:index + 1])
            start, end = row["crop_start"], row["crop_end"]
            values = (
                row["wet"][start:end].numpy(),
                restored[0, start:end].cpu().numpy().astype(np.float32),
                row["clean"][start:end].numpy(),
            )
            for collection in (
                aggregate,
                categories[row["category"]],
                profiles[row["profile_id"]],
            ):
                for target, value in zip(collection, values, strict=True):
                    target.append(value)
    report = summarize("amp", *aggregate)
    report["categories"] = {
        name: summarize("amp", *values) for name, values in sorted(categories.items())
    }
    report["profiles"] = {
        name: summarize("amp", *values) for name, values in sorted(profiles.items())
    }
    return report


def audit(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    run = args.run.resolve()
    checkpoint = run / "model.pt"
    training = json.loads((run / "metrics.json").read_text())
    digest_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest_before != training["model"]["sha256"]:
        raise PermissionError("checkpoint hash differs from frozen training report")
    if training["model"].get("schema") != 12:
        raise ValueError("profile oracle requires phase gray-box schema 12")
    if training["data"].get("development_audio_opened") is not True:
        raise PermissionError("profile oracle is restricted to already revealed development")
    if training["quality"].get("locked_final_audio_opened") is not False:
        raise PermissionError("training report did not preserve locked-final")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    architecture = payload["architecture"]
    model = WetTonePhaseGrayBoxAmpCabInverse(
        architecture["channels"], architecture["depth"], architecture["condition_size"]
    )
    model.load_state_dict(payload["state_dict"])
    model.tone_encoder_sha256 = architecture["tone_encoder_sha256"]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = torch.device(args.device)
    model = model.to(device).eval()

    contract = json.loads((workspace / "remix/egdb_pg_subset_v1.json").read_text())
    tracks = list(contract["tracks"]["development"])
    if not 2 <= args.support_tracks < len(tracks):
        raise ValueError("support track count must leave a nonempty query set")
    support_ids = set(tracks[:args.support_tracks])
    query_ids = set(tracks[args.support_tracks:])
    if support_ids & query_ids:
        raise PermissionError("oracle support/query tracks overlap")
    dataset = EgdbPgAmpPairs(
        workspace,
        "development",
        len(CATEGORIES) * len(tracks),
        args.target_frames,
        args.context_frames,
        SEED + 3,
        round(args.reference_seconds * RATE),
    )
    rows = [dataset[index] for index in range(len(dataset))]
    support = [row for row in rows if row["track"] in support_ids]
    query = [row for row in rows if row["track"] in query_ids]
    by_profile = defaultdict(list)
    for row in support:
        by_profile[row["profile_id"]].append(row)
    if set(by_profile) != {profile for _, profile in dataset.profile_rows}:
        raise PermissionError("oracle support does not cover every development profile")

    conditions = {}
    optimization = {}
    for profile, selected in sorted(by_profile.items()):
        conditions[profile], optimization[profile] = _fit_condition(
            model,
            selected,
            device,
            args.steps,
            args.learning_rate,
            args.batch_size,
        )
    baseline = _quality(model, query, None, device)
    oracle = _quality(model, query, conditions, device)
    example_conditions = None
    example_optimization = None
    example_oracle = None
    if args.example_self_steps:
        example_conditions, example_optimization = _fit_example_conditions(
            model,
            query,
            device,
            args.example_self_steps,
            args.example_self_learning_rate,
            args.example_self_batch_size,
        )
        example_oracle = _quality_example_conditions(
            model, query, example_conditions, device
        )
    digest_after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 1,
        "status": "clean-assisted-development-oracle-not-promotable",
        "accepted": False,
        "product_candidate": False,
        "usable_model": None,
        "checkpoint_sha256": digest_before,
        "checkpoint_unchanged": digest_after == digest_before,
        "device": device.type,
        "split": "revealed-development-only",
        "support_tracks": sorted(support_ids),
        "query_tracks": sorted(query_ids),
        "support_query_disjoint": not bool(support_ids & query_ids),
        "support_examples": len(support),
        "query_examples": len(query),
        "clean_oracle_support_used": True,
        "condition_vectors_saved": False,
        "profile_oracle_allowed_at_runtime": False,
        "fresh_validation_opened": False,
        "fresh_validation_v2_opened": False,
        "locked_final_opened": False,
        "baseline_wet_estimator": baseline,
        "clean_assisted_profile_oracle": oracle,
        "clean_assisted_example_self_oracle": example_oracle,
        "example_self_oracle_optimization": example_optimization,
        "example_self_oracle_is_same_audio_contaminated": example_oracle is not None,
        "optimization": optimization,
        "interpretation": (
            "If the oracle materially exceeds the Wet estimator on disjoint query tracks, "
            "parameter identifiability is limiting. Otherwise the frozen gray-box processor "
            "family is limiting. This report can never promote or select product weights."
        ),
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace profile oracle report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "support_examples": len(support),
        "query_examples": len(query),
        "baseline_pass_fraction": baseline["pass_fraction"],
        "oracle_pass_fraction": oracle["pass_fraction"],
        "baseline_metrics": baseline["metrics"],
        "oracle_metrics": oracle["metrics"],
        "example_self_oracle_pass_fraction": (
            None if example_oracle is None else example_oracle["pass_fraction"]
        ),
        "example_self_oracle_metrics": (
            None if example_oracle is None else example_oracle["metrics"]
        ),
        "checkpoint_unchanged": report["checkpoint_unchanged"],
    }, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--support-tracks", type=int, default=12)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-frames", type=int, default=16384)
    parser.add_argument("--context-frames", type=int, default=4096)
    parser.add_argument("--reference-seconds", type=float, default=3.0)
    parser.add_argument("--example-self-steps", type=int, default=0)
    parser.add_argument("--example-self-learning-rate", type=float, default=0.02)
    parser.add_argument("--example-self-batch-size", type=int, default=12)
    args = parser.parse_args()
    if (
        args.steps < 1 or args.batch_size < 1 or args.example_self_steps < 0
        or args.example_self_batch_size < 1
    ):
        raise ValueError("oracle optimization geometry is invalid")
    torch.manual_seed(SEED)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    audit(args)


if __name__ == "__main__":
    main()
