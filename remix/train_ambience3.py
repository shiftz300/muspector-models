#!/usr/bin/env python3
"""Train the profile-conditioned independent Ambience v3 expert."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .ambience3 import AmbiencePairsV3, profile_inverse_base, wet_candidate_tail_ratio
from .ambience_model2 import ambience_loss
from .ambience_model3 import AmbienceProfileExpert
from .product_data import _rir
from .quality2 import summarize as summarize_universal
from .reverb_quality import summarize as summarize_tail


SEED = 20260920
THRESHOLD_CANDIDATES = (0.80, 0.90, 1.00, 1.05)
MINIMUM_COVERAGE = 0.55


def _collate(rows: list[dict]) -> dict:
    starts = {int(row["target_start"]) for row in rows}
    if len(starts) != 1:
        raise ValueError("one ambience3 batch must share target geometry")
    return {
        "wet": torch.stack([row["wet"] for row in rows]),
        "clean": torch.stack([row["clean"] for row in rows]),
        "controls": torch.stack([row["controls"] for row in rows]),
        "profile_base": torch.stack([row["profile_base"] for row in rows]),
        "profile_fallback": torch.tensor([row["profile_fallback"] for row in rows], dtype=torch.bool),
        "target_start": starts.pop(),
    }


def _device_batch(batch: dict, device: torch.device) -> dict:
    return {
        **batch,
        "wet": batch["wet"].to(device),
        "clean": batch["clean"].to(device),
        "controls": batch["controls"].to(device),
        "profile_base": batch["profile_base"].to(device),
        "profile_fallback": batch["profile_fallback"].to(device),
    }


def _mean_loss(
    model: AmbienceProfileExpert,
    dataset: AmbiencePairsV3,
    batch_size: int,
    device: torch.device,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total = 0.0
    examples = 0
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            batch = _device_batch(raw, device)
            restored, uncertainty = model(
                batch["wet"], batch["controls"], batch["profile_base"], batch["profile_fallback"]
            )
            loss, _ = ambience_loss(
                restored, uncertainty, batch["wet"], batch["clean"], batch["target_start"]
            )
            count = len(batch["wet"])
            total += float(loss) * count
            examples += count
    return total / max(examples, 1)


def _decay_stratum(seconds: float) -> str:
    return "short-tail" if seconds < 1.85 else "long-tail"


def _empty_rows() -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    return [], [], []


def _summaries(values: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]) -> dict:
    if not values[0]:
        return {"accepted": False, "reason": "no accepted examples"}
    universal = summarize_universal("ambience", *values)
    try:
        tail = summarize_tail(*values)
    except ValueError:
        tail = {"accepted": False, "reason": "no eligible temporal tails"}
    return {
        "universal": universal,
        "tail": tail,
        "accepted": bool(universal["accepted"] and tail["accepted"]),
    }


def _quality(
    model: AmbienceProfileExpert,
    dataset: AmbiencePairsV3,
    threshold: float,
) -> dict:
    aggregate = _empty_rows()
    fallback_rows = _empty_rows()
    sources = defaultdict(_empty_rows)
    decays = defaultdict(_empty_rows)
    attempted_sources = Counter()
    accepted_sources = Counter()
    attempted_decays = Counter()
    accepted_decays = Counter()
    modes = Counter()
    abstentions = []
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            source = row["source_id"]
            decay = _decay_stratum(float(row["control_values"]["decay_p999_seconds"]))
            attempted_sources[source] += 1
            attempted_decays[decay] += 1
            restored, _ = model(
                row["wet"].unsqueeze(0),
                row["controls"].unsqueeze(0),
                row["profile_base"].unsqueeze(0),
                torch.tensor([row["profile_fallback"]]),
            )
            candidate_full = restored[0].numpy().astype(np.float32)
            mode = "fallback" if row["profile_fallback"] else "exact"
            score = 0.0
            if row["profile_fallback"]:
                score = wet_candidate_tail_ratio(
                    row["wet"].numpy(), candidate_full, int(row["target_start"])
                )
                if score > threshold:
                    abstentions.append({
                        "index": index,
                        "source_id": source,
                        "group": row["group"],
                        "rir": row["rir"],
                        "tail_ratio": score,
                        "threshold": threshold,
                    })
                    continue
            modes[mode] += 1
            accepted_sources[source] += 1
            accepted_decays[decay] += 1
            start = int(row["target_start"])
            wet = row["wet"][start:].numpy()
            candidate = candidate_full[start:]
            clean = row["clean"][start:].numpy()
            for collection in (aggregate, sources[source], decays[decay]):
                collection[0].append(wet)
                collection[1].append(candidate)
                collection[2].append(clean)
            if mode == "fallback":
                fallback_rows[0].append(wet)
                fallback_rows[1].append(candidate)
                fallback_rows[2].append(clean)

    result = _summaries(aggregate)
    result["fallback_quality"] = _summaries(fallback_rows)
    result["attempted_examples"] = len(dataset)
    result["accepted_examples"] = len(aggregate[0])
    result["coverage"] = len(aggregate[0]) / len(dataset)
    result["accepted_modes"] = dict(modes)
    result["abstentions"] = abstentions
    result["tail_ratio_threshold"] = threshold
    result["sources"] = {}
    for name in sorted(attempted_sources):
        report = _summaries(sources[name])
        report["attempted_examples"] = attempted_sources[name]
        report["accepted_examples"] = accepted_sources[name]
        report["coverage"] = accepted_sources[name] / attempted_sources[name]
        result["sources"][name] = report
    result["decay_strata"] = {}
    for name in sorted(attempted_decays):
        report = _summaries(decays[name])
        report["attempted_examples"] = attempted_decays[name]
        report["accepted_examples"] = accepted_decays[name]
        report["coverage"] = accepted_decays[name] / attempted_decays[name]
        result["decay_strata"][name] = report
    result["gates"] = {
        "aggregate_quality": bool(result.get("accepted")),
        "fallback_quality": bool(result["fallback_quality"].get("accepted")),
        "coverage": result["coverage"] >= MINIMUM_COVERAGE,
        "each_source_quality": bool(
            result["sources"] and all(row.get("accepted", False) for row in result["sources"].values())
        ),
        "each_decay_quality": bool(
            result["decay_strata"]
            and all(row.get("accepted", False) for row in result["decay_strata"].values())
        ),
    }
    result["accepted"] = all(result["gates"].values())
    return result


def _select_threshold(model: AmbienceProfileExpert, calibration: AmbiencePairsV3) -> tuple[float, dict]:
    reports = {str(value): _quality(model, calibration, value) for value in THRESHOLD_CANDIDATES}
    accepted = [value for value in THRESHOLD_CANDIDATES if reports[str(value)]["accepted"]]
    selected = max(accepted) if accepted else min(THRESHOLD_CANDIDATES)
    return selected, {
        "schema": 1,
        "selected": selected,
        "selection_split": "calibration",
        "candidates": reports,
        "accepted_candidate_exists": bool(accepted),
    }


def _runtime(model: AmbienceProfileExpert, dataset: AmbiencePairsV3) -> dict:
    raw = dataset.base[0]
    impulse = _rir(dataset.rir_paths[raw["rir"]])
    started = time.perf_counter()
    base, profile = profile_inverse_base(raw["wet"].numpy(), impulse, raw["control_values"])
    profile_seconds = time.perf_counter() - started
    wet = raw["wet"].unsqueeze(0)
    controls = raw["controls"].unsqueeze(0)
    profile_base = torch.from_numpy(base).unsqueeze(0)
    fallback = torch.tensor([profile["fallback"]])
    model.eval()
    with torch.inference_mode():
        model(wet, controls, profile_base, fallback)
        started = time.perf_counter()
        repeats = 2
        for _ in range(repeats):
            model(wet, controls, profile_base, fallback)
        model_seconds = (time.perf_counter() - started) / repeats
    audio_seconds = len(raw["wet"]) / 48_000.0
    return {
        "frames": len(raw["wet"]),
        "profile_inverse_seconds": profile_seconds,
        "model_seconds": model_seconds,
        "full_realtime_factor": (profile_seconds + model_seconds) / audio_seconds,
        "ordinary_cpu": True,
        "bounded_window": True,
        "audio_callback": False,
    }


def train(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    fit = AmbiencePairsV3(workspace, "fit", args.train_samples, args.target_frames, SEED + 1)
    calibration = AmbiencePairsV3(
        workspace, "calibration", args.calibration_samples, args.target_frames, SEED + 2
    )
    development = AmbiencePairsV3(
        workspace, "development", args.development_samples, args.target_frames, SEED + 3
    )
    device = torch.device(args.device)
    model = AmbienceProfileExpert(args.channels, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5)
    loader = DataLoader(
        fit,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        collate_fn=_collate,
    )
    initial_loss = _mean_loss(model, calibration, args.batch_size, device)
    history = [{"epoch": 0, "train": None, "calibration_loss": initial_loss}]
    best_loss = initial_loss
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    print(json.dumps({"mechanism": "ambience", "epoch": 0, "calibration_loss": initial_loss}), flush=True)
    for epoch in range(args.epochs):
        model.train()
        totals = defaultdict(float)
        examples = 0
        for raw in loader:
            batch = _device_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            restored, uncertainty = model(
                batch["wet"], batch["controls"], batch["profile_base"], batch["profile_fallback"]
            )
            loss, parts = ambience_loss(
                restored, uncertainty, batch["wet"], batch["clean"], batch["target_start"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            count = len(batch["wet"])
            totals["loss"] += float(loss.detach()) * count
            for name, value in parts.items():
                totals[name] += value * count
            examples += count
        calibration_loss = _mean_loss(model, calibration, args.batch_size, device)
        history.append({
            "epoch": epoch + 1,
            "train": {name: value / max(examples, 1) for name, value in sorted(totals.items())},
            "calibration_loss": calibration_loss,
        })
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(json.dumps({"mechanism": "ambience", "epoch": epoch + 1, "calibration_loss": calibration_loss}), flush=True)

    model = model.cpu()
    model.load_state_dict(best_state)
    threshold, calibration_report = _select_threshold(model, calibration)
    development_report = _quality(model, development, threshold)
    runtime = _runtime(model, development)
    accepted = bool(
        not args.quick
        and calibration_report["accepted_candidate_exists"]
        and development_report["accepted"]
        and runtime["full_realtime_factor"] <= 0.5
    )
    target = output / "ambience"
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "model.pt"
    torch.save({
        "schema": 2,
        "sample_rate": 48_000,
        "architecture": model.manifest(),
        "tail_ratio_threshold": threshold,
        "state_dict": model.state_dict(),
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 2,
        "status": "accepted" if accepted else "diagnostic-not-promoted",
        "accepted": accepted,
        "quick": args.quick,
        "mechanism": "ambience",
        "model": {**model.manifest(), "checkpoint": str(checkpoint), "sha256": digest},
        "training": {
            "seed": SEED,
            "epochs": args.epochs,
            "target_frames": args.target_frames,
            "history_frames": fit.history_frames,
            "fit_samples_per_epoch": args.train_samples,
            "calibration_samples": args.calibration_samples,
            "development_samples": args.development_samples,
            "selected_calibration_loss": best_loss,
            "history": history,
            "accelerator": device.type,
        },
        "calibration": calibration_report,
        "development": development_report,
        "runtime": runtime,
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
    }
    (target / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/product3-ambience"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=120)
    parser.add_argument("--calibration-samples", type=int, default=48)
    parser.add_argument("--development-samples", type=int, default=72)
    parser.add_argument("--target-frames", type=int, default=16_384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    args = parser.parse_args()
    if args.quick:
        args.epochs = 2
        args.train_samples = 24
        args.calibration_samples = 16
        args.development_samples = 24
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
