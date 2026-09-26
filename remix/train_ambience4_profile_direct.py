#!/usr/bin/env python3
"""Fit/calibration-only MPS capacity screen for Product4 Reverb graybox v4."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .ambience5_profile_direct import AmbiencePairsV4MixedProfileDirect
from .ambience_model5 import AmbienceProfileDirectExpert
from .reverb_quality2 import measure, summarize


SEED = 20260909


def _collate(rows: list[dict]) -> dict:
    starts = {int(row["target_start"]) for row in rows}
    if len(starts) != 1:
        raise ValueError("one profile-direct batch must share target geometry")
    maximum_transfer = max(len(row["transfer"]) for row in rows)
    transfers = torch.zeros(len(rows), maximum_transfer)
    for index, row in enumerate(rows):
        transfers[index, : len(row["transfer"])] = row["transfer"]
    return {
        "wet": torch.stack([row["wet"] for row in rows]),
        "clean": torch.stack([row["clean"] for row in rows]),
        "controls": torch.stack([row["controls"] for row in rows]),
        "profile_features": torch.stack([row["profile_features"] for row in rows]),
        "transfer": transfers,
        "analytic_base": torch.stack([row["analytic_base"] for row in rows]),
        "exact_base": torch.stack([row["exact_base"] for row in rows]),
        "exact_available": torch.tensor([row["exact_available"] for row in rows]),
        "target_start": starts.pop(),
    }


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _quality(model: AmbienceProfileDirectExpert, dataset, device: torch.device) -> dict:
    aggregate = []
    groups = defaultdict(list)
    rows = []
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            restored, _ = model(
                row["wet"][None].to(device),
                row["controls"][None].to(device),
                row["profile_features"][None].to(device),
                row["analytic_base"][None].to(device),
                row["exact_base"][None].to(device),
                torch.tensor([row["exact_available"]], device=device),
            )
            start = int(row["target_start"])
            wet = row["wet"][start:].numpy()
            clean = row["clean"][start:].numpy()
            candidate = restored[0, start:].cpu().numpy().astype(np.float32)
            result = measure(wet, candidate, clean)
            raw_decision = result["decision"]
            if raw_decision != "effective-restoration":
                result = measure(wet, wet.copy(), clean)
            aggregate.append(result)
            decay = dataset.decay_stratum(float(row["control_values"]["decay_p999_seconds"]))
            room = dataset.room_group(row)
            groups[f"source:{row['source_id']}"].append(result)
            groups[f"room:{room}"].append(result)
            groups[f"decay:{decay}"].append(result)
            rows.append({
                "index": index,
                "source_id": row["source_id"],
                "rir_source_id": row["rir_source_id"],
                "room_group": room,
                "decay_stratum": decay,
                "exact_available": bool(row["exact_available"]),
                "analytic_mode": row["analytic_mode"],
                "raw_decision": raw_decision,
                "oracle_decision": result["decision"],
            })
    report = summarize(aggregate)
    group_reports = {name: summarize(values) for name, values in sorted(groups.items())}
    return {
        "accepted": bool(report["accepted"] and all(row["accepted"] for row in group_reports.values())),
        "aggregate": report,
        "groups": group_reports,
        "rows": rows,
    }


def _rank(report: dict) -> tuple[float, ...]:
    groups = list(report["groups"].values())
    return (
        float(report["accepted"]),
        float(report["aggregate"]["accepted"]),
        min((row["effective_eligible_coverage"] for row in groups), default=-1.0),
        report["aggregate"]["effective_eligible_coverage"],
        -report["aggregate"]["hard_bypass_examples"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/foundation/product4-reverb-profile-direct-v4-capacity-smoke-v1"),
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=16)
    parser.add_argument("--calibration-samples", type=int, default=24)
    parser.add_argument("--target-frames", type=int, default=65_536)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--physical-weight", type=float, default=0.25)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace profile-direct run: {output}")
    if args.calibration_samples < 24:
        raise ValueError("profile-direct calibration screen needs at least 24 examples")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))
    workspace = args.workspace.resolve()
    fit = AmbiencePairsV4MixedProfileDirect(
        workspace, "fit", args.train_samples, args.target_frames, SEED + 1
    )
    calibration = AmbiencePairsV4MixedProfileDirect(
        workspace, "calibration", args.calibration_samples, args.target_frames, SEED + 2
    )
    device = torch.device(args.device)
    model = AmbienceProfileDirectExpert(args.channels, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5)
    loader = DataLoader(
        fit,
        batch_size=1,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        collate_fn=_collate,
    )
    candidates = []
    history = []
    for epoch in range(args.epochs + 1):
        if epoch:
            model.train()
            totals = defaultdict(float)
            examples = 0
            for raw in loader:
                batch = _to_device(raw, device)
                optimizer.zero_grad(set_to_none=True)
                loss, parts = model.training_loss(
                    batch["wet"],
                    batch["controls"],
                    batch["profile_features"],
                    batch["analytic_base"],
                    batch["exact_base"],
                    batch["exact_available"],
                    batch["transfer"],
                    batch["clean"],
                    batch["target_start"],
                    physical_weight=args.physical_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                totals["loss"] += float(loss.detach())
                for name, value in parts.items():
                    totals[name] += value
                examples += 1
            history.append({
                "epoch": epoch,
                "train": {name: value / examples for name, value in sorted(totals.items())},
            })
        quality = _quality(model, calibration, device)
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        candidates.append(( _rank(quality), epoch, state, quality))
        print(json.dumps({
            "epoch": epoch,
            "calibration_accepted": quality["accepted"],
            "calibration": quality["aggregate"],
        }, sort_keys=True), flush=True)
    rank, selected_epoch, state, quality = max(candidates, key=lambda row: (row[0], row[1]))
    model = model.cpu()
    model.load_state_dict(state)
    output.mkdir(parents=True)
    checkpoint = output / "model.pt"
    torch.save({
        "schema": 1,
        "sample_rate": 48_000,
        "architecture": model.manifest(),
        "state_dict": model.state_dict(),
    }, checkpoint)
    report = {
        "schema": 1,
        "status": (
            "calibration-capacity-passed-gate-not-deployable"
            if quality["accepted"] else "calibration-capacity-rejected"
        ),
        "accepted": False,
        "capacity_gate_passed": quality["accepted"],
        "partition": "fit-and-calibration-only",
        "development_opened": False,
        "listening_rows_used": False,
        "fresh_rochester_opened": False,
        "locked_final_accessed": False,
        "deployable": False,
        "calibration_oracle": "Clean may retain an effective v4 output; otherwise exact Wet hard bypass",
        "model": {
            **model.manifest(),
            "checkpoint": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "training": {
            "accelerator": args.device,
            "seed": SEED,
            "epochs": args.epochs,
            "fit_samples_per_epoch": args.train_samples,
            "calibration_samples": args.calibration_samples,
            "target_frames": args.target_frames,
            "learning_rate": args.learning_rate,
            "physical_replay_weight": args.physical_weight,
            "selected_epoch": selected_epoch,
            "selection_rank": list(rank),
            "history": history,
        },
        "calibration": quality,
        "provenance": {
            "product_sources_only": True,
            "authorized_source_ids": fit.authorization["sources"],
            "required_attribution": fit.authorization["required_attribution"],
            "realized_source_counts": {
                "fit": fit.realized_source_counts(),
                "calibration": calibration.realized_source_counts(),
            },
            "realized_rir_counts": {
                "fit": fit.realized_rir_counts(),
                "calibration": calibration.realized_rir_counts(),
            },
            "research_source_ids": [],
            "generated_audio_written": False,
            "physical_audio_devices_used": False,
        },
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "selected_epoch": selected_epoch,
        "calibration": quality["aggregate"],
        "metrics": str(output / "metrics.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
