#!/usr/bin/env python3
"""Train the independent long-tail product ambience expert."""

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

from .ambience2 import AmbiencePairsV2
from .ambience_model2 import AmbienceExpert, ambience_loss
from .quality2 import summarize as summarize_universal
from .reverb_quality import summarize as summarize_tail


SEED = 20260907


def _collate(rows: list[dict]) -> dict:
    starts = {int(row["target_start"]) for row in rows}
    if len(starts) != 1:
        raise ValueError("one ambience2 batch must share target geometry")
    return {
        "wet": torch.stack([row["wet"] for row in rows]),
        "clean": torch.stack([row["clean"] for row in rows]),
        "controls": torch.stack([row["controls"] for row in rows]),
        "late_base": torch.stack([row["late_base"] for row in rows]),
        "target_start": starts.pop(),
    }


def _device_batch(batch: dict, device: torch.device) -> dict:
    return {
        **batch,
        "wet": batch["wet"].to(device),
        "clean": batch["clean"].to(device),
        "controls": batch["controls"].to(device),
        "late_base": batch["late_base"].to(device),
    }


def _batch_loss(
    model,
    batch: dict,
    uncertainty_weight: float,
    tail_weight: float,
    gate_oracle_weight: float,
) -> tuple[torch.Tensor, dict]:
    training_loss = getattr(model, "training_loss", None)
    if training_loss is not None:
        return training_loss(
            batch["wet"],
            batch["controls"],
            batch["late_base"],
            batch["clean"],
            batch["target_start"],
            uncertainty_weight,
            tail_weight,
            gate_oracle_weight,
        )
    restored, uncertainty = model(batch["wet"], batch["controls"], batch["late_base"])
    return ambience_loss(
        restored,
        uncertainty,
        batch["wet"],
        batch["clean"],
        batch["target_start"],
        uncertainty_weight,
        tail_weight,
    )


def _mean_loss(
    model: AmbienceExpert,
    dataset: AmbiencePairsV2,
    batch_size: int,
    device: torch.device,
    uncertainty_weight: float = 0.05,
    tail_weight: float = 1.0,
    gate_oracle_weight: float = 0.0,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total = 0.0
    examples = 0
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            batch = _device_batch(raw, device)
            loss, _ = _batch_loss(
                model,
                batch,
                uncertainty_weight,
                tail_weight,
                gate_oracle_weight,
            )
            count = len(batch["wet"])
            total += float(loss) * count
            examples += count
    return total / max(examples, 1)


def _decay_stratum(seconds: float) -> str:
    return "short-tail" if seconds < 1.85 else "long-tail"


def _summaries(
    values: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]],
    quality_contract: str | None = None,
) -> dict:
    universal = summarize_universal("ambience", *values)
    try:
        tail = summarize_tail(*values)
    except ValueError:
        tail = {"accepted": False, "reason": "no eligible temporal tails"}
    result = {
        "universal": universal,
        "tail": tail,
        "accepted": bool(universal["accepted"] and tail["accepted"]),
    }
    if quality_contract == "tail-removal-with-global-nonregression":
        nonregression_gates = {
            name: row["nonregression_fraction"] >= 0.90
            for name, row in universal["metrics"].items()
        }
        nonregression_gates["no_new_clipping"] = universal["gates"]["no_new_clipping"]
        result["global_nonregression"] = {
            "gates": nonregression_gates,
            "accepted": all(nonregression_gates.values()),
        }
        result["accepted"] = bool(
            result["global_nonregression"]["accepted"] and tail["accepted"]
        )
    return result


def _quality(model: AmbienceExpert, dataset: AmbiencePairsV2) -> dict:
    aggregate = ([], [], [])
    sources = defaultdict(lambda: ([], [], []))
    decays = defaultdict(lambda: ([], [], []))
    rooms = defaultdict(lambda: ([], [], []))
    decay_stratum = getattr(dataset, "decay_stratum", _decay_stratum)
    room_group = getattr(dataset, "room_group", None)
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            restored, _ = model(
                row["wet"].unsqueeze(0), row["controls"].unsqueeze(0), row["late_base"].unsqueeze(0)
            )
            start = int(row["target_start"])
            wet = row["wet"][start:].numpy()
            candidate = restored[0, start:].numpy().astype(np.float32)
            clean = row["clean"][start:].numpy()
            stratum = decay_stratum(float(row["control_values"]["decay_p999_seconds"]))
            collections = [aggregate, sources[row["source_id"]], decays[stratum]]
            if room_group is not None:
                collections.append(rooms[room_group(row)])
            for collection in collections:
                collection[0].append(wet)
                collection[1].append(candidate)
                collection[2].append(clean)
    quality_contract = getattr(dataset, "quality_contract", None)
    result = _summaries(aggregate, quality_contract)
    result["quality_contract"] = quality_contract or "universal-improvement-and-tail-removal"
    result["sources"] = {
        name: _summaries(values, quality_contract) for name, values in sorted(sources.items())
    }
    result["decay_strata"] = {
        name: _summaries(values, quality_contract) for name, values in sorted(decays.items())
    }
    result["all_sources_accepted"] = bool(
        result["sources"] and all(row["accepted"] for row in result["sources"].values())
    )
    result["all_decay_strata_accepted"] = bool(
        result["decay_strata"]
        and set(result["decay_strata"]) >= set(getattr(dataset, "required_decay_strata", ()))
        and all(row["accepted"] for row in result["decay_strata"].values())
    )
    if room_group is not None:
        result["rooms"] = {
            name: _summaries(values, quality_contract) for name, values in sorted(rooms.items())
        }
        result["all_rooms_accepted"] = bool(
            result["rooms"] and all(row["accepted"] for row in result["rooms"].values())
        )
    return result


def _runtime(model: AmbienceExpert, frames: int) -> dict:
    wet = torch.zeros(1, frames)
    controls = torch.zeros(1, 3)
    if getattr(model, "requires_profile_candidate_bank", False):
        late_base = wet[:, None].repeat(1, model.candidate_count, 1)
    else:
        late_base = wet.clone()
    model.eval()
    with torch.inference_mode():
        model(wet, controls, late_base)
        started = time.perf_counter()
        repeats = 2
        for _ in range(repeats):
            model(wet, controls, late_base)
        elapsed = (time.perf_counter() - started) / repeats
    return {
        "frames": frames,
        "mean_seconds": elapsed,
        "realtime_factor": elapsed / (frames / 48_000.0),
        "ordinary_cpu": True,
        "bounded_window": True,
        "audio_callback": False,
        "profile_candidate_generation_included": False,
    }


def _quality_rank(report: dict) -> tuple[float, ...]:
    """Rank calibration checkpoints by product tail evidence, never development."""
    tail_reports = [report["tail"]]
    for axis in ("sources", "rooms", "decay_strata"):
        tail_reports.extend(row["tail"] for row in report.get(axis, {}).values())
    measurable = [row for row in tail_reports if "pass_fraction" in row]
    if not measurable:
        return (-1.0,) * 9
    group_gates = (
        bool(report["accepted"]),
        bool(report["all_sources_accepted"]),
        bool(report["all_decay_strata_accepted"]),
        bool(report.get("all_rooms_accepted", True)),
    )
    universal_reports = [report["universal"]]
    for axis in ("sources", "rooms", "decay_strata"):
        universal_reports.extend(
            row["universal"] for row in report.get(axis, {}).values()
        )
    universal_metrics = [
        metric
        for universal in universal_reports
        for metric in universal["metrics"].values()
    ]
    return (
        float(all(group_gates)),
        float(sum(group_gates)),
        float(all(row.get("added_reverb_fraction", 1.0) <= 0.05 for row in measurable)),
        min(float(row["pass_fraction"]) for row in measurable),
        float(report["tail"]["pass_fraction"]),
        min(float(row["median_tail_excess_reduction"]) for row in measurable),
        min(float(row["median_tail_envelope_esr_improvement"]) for row in measurable),
        min(float(row["nonregression_fraction"]) for row in universal_metrics),
        min(
            float(row["median_reduction"])
            if row["median_reduction"] is not None else -1.0
            for row in universal_metrics
        ),
    )


def train(
    args: argparse.Namespace,
    dataset_class=AmbiencePairsV2,
    model_class=AmbienceExpert,
) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    seed = getattr(args, "seed", SEED)
    fit = dataset_class(workspace, "fit", args.train_samples, args.target_frames, seed + 1)
    calibration = dataset_class(
        workspace, "calibration", args.calibration_samples, args.target_frames, seed + 2
    )
    development = dataset_class(
        workspace, "development", args.development_samples, args.target_frames, seed + 3
    )
    device = torch.device(args.device)
    uncertainty_weight = getattr(args, "uncertainty_weight", 0.05)
    tail_weight = getattr(args, "tail_weight", 1.0)
    gate_oracle_weight = getattr(args, "gate_oracle_weight", 0.0)
    model = model_class(args.channels, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5)
    loader = DataLoader(
        fit,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_collate,
    )
    initial_loss = _mean_loss(
        model,
        calibration,
        args.batch_size,
        device,
        uncertainty_weight,
        tail_weight,
        gate_oracle_weight,
    )
    history = [{"epoch": 0, "train": None, "calibration_loss": initial_loss}]
    best_loss = initial_loss
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    quality_selection = bool(getattr(args, "select_by_calibration_quality", False))
    selection_interval = int(getattr(args, "quality_selection_interval", 2))
    candidate_states = [(0, initial_loss, best_state)] if quality_selection else []
    print(json.dumps({"mechanism": "ambience", "epoch": 0, "calibration_loss": initial_loss}), flush=True)
    for epoch in range(args.epochs):
        model.train()
        totals = defaultdict(float)
        examples = 0
        for raw in loader:
            batch = _device_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = _batch_loss(
                model,
                batch,
                uncertainty_weight,
                tail_weight,
                gate_oracle_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            count = len(batch["wet"])
            totals["loss"] += float(loss.detach()) * count
            for name, value in parts.items():
                totals[name] += value * count
            examples += count
        calibration_loss = _mean_loss(
            model,
            calibration,
            args.batch_size,
            device,
            uncertainty_weight,
            tail_weight,
            gate_oracle_weight,
        )
        history.append({
            "epoch": epoch + 1,
            "train": {name: value / max(examples, 1) for name, value in sorted(totals.items())},
            "calibration_loss": calibration_loss,
        })
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        if quality_selection and (
            (epoch + 1) % selection_interval == 0 or epoch + 1 == args.epochs
        ):
            candidate_states.append((
                epoch + 1,
                calibration_loss,
                {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            ))
        print(json.dumps({"mechanism": "ambience", "epoch": epoch + 1, "calibration_loss": calibration_loss}), flush=True)
    model = model.cpu()
    calibration_selection = []
    selected_epoch = next(
        row["epoch"] for row in history if row["calibration_loss"] == best_loss
    )
    if quality_selection:
        ranked = []
        for epoch, calibration_loss, state in candidate_states:
            model.load_state_dict(state)
            calibration_quality = _quality(model, calibration)
            rank = _quality_rank(calibration_quality)
            calibration_selection.append({
                "epoch": epoch,
                "calibration_loss": calibration_loss,
                "rank": list(rank),
                "tail": calibration_quality["tail"],
                "all_sources_accepted": calibration_quality["all_sources_accepted"],
                "all_decay_strata_accepted": calibration_quality["all_decay_strata_accepted"],
                "all_rooms_accepted": calibration_quality.get("all_rooms_accepted"),
            })
            ranked.append((rank, -calibration_loss, epoch, state))
            print(json.dumps({
                "mechanism": "ambience",
                "calibration_quality_epoch": epoch,
                "quality_rank": rank,
            }), flush=True)
        _, negated_loss, selected_epoch, best_state = max(ranked, key=lambda row: row[:3])
        best_loss = -negated_loss
    model.load_state_dict(best_state)
    quality = _quality(model, development)
    runtime = _runtime(model, fit.total_frames)
    accepted = bool(
        not args.quick
        and quality["accepted"]
        and quality["all_sources_accepted"]
        and quality["all_decay_strata_accepted"]
        and quality.get("all_rooms_accepted", True)
        and runtime["realtime_factor"] <= 0.5
    )
    target = output / "ambience"
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "model.pt"
    torch.save({
        "schema": 1,
        "sample_rate": 48_000,
        "architecture": model.manifest(),
        "state_dict": model.state_dict(),
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 1,
        "status": "accepted" if accepted else "diagnostic-not-promoted",
        "accepted": accepted,
        "quick": args.quick,
        "mechanism": "ambience",
        "model": {**model.manifest(), "checkpoint": str(checkpoint), "sha256": digest},
        "training": {
            "seed": seed,
            "epochs": args.epochs,
            "uncertainty_weight": uncertainty_weight,
            "tail_weight": tail_weight,
            "gate_oracle_weight": gate_oracle_weight,
            "target_frames": args.target_frames,
            "history_frames": fit.history_frames,
            "fit_samples_per_epoch": args.train_samples,
            "calibration_samples": args.calibration_samples,
            "development_samples": args.development_samples,
            "selected_calibration_loss": best_loss,
            "selected_epoch": selected_epoch,
            "checkpoint_selection": (
                "calibration-product-tail-quality" if quality_selection else "calibration-mean-loss"
            ),
            "calibration_quality_candidates": calibration_selection,
            "history": history,
            "accelerator": device.type,
        },
        "provenance": {
            "product_sources_only": True,
            "realized_source_counts": {
                "fit": fit.realized_source_counts(),
                "calibration": calibration.realized_source_counts(),
                "development": development.realized_source_counts(),
            },
            "realized_rir_counts": {
                "fit": fit.realized_rir_counts(),
                "calibration": calibration.realized_rir_counts(),
                "development": development.realized_rir_counts(),
            },
            "authorized_source_ids": fit.authorization["sources"],
            "required_attribution": fit.authorization["required_attribution"],
            "research_source_ids": [],
            "generated_audio_written": False,
            "physical_audio_devices_used": False,
            "locked_final_audio_opened": False,
        },
        "development": quality,
        "runtime": runtime,
    }
    (target / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/product2"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=120)
    parser.add_argument("--calibration-samples", type=int, default=30)
    parser.add_argument("--development-samples", type=int, default=48)
    parser.add_argument("--target-frames", type=int, default=16_384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    args = parser.parse_args()
    if args.quick:
        args.epochs = 2
        args.train_samples = 24
        args.calibration_samples = 8
        args.development_samples = 12
        args.batch_size = 1
        args.channels = 6
        args.depth = 8
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))
    report = train(args)
    print(json.dumps({
        "mechanism": "ambience",
        "status": report["status"],
        "checkpoint": report["model"]["checkpoint"],
        "metrics": str(args.output.resolve() / "ambience/metrics.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
