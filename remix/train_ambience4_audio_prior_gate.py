#!/usr/bin/env python3
"""Fit-only MPS training and one-shot calibration of a Reverb audio-prior gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .ambience5_profile_direct import AmbiencePairsV4MixedProfileDirect
from .ambience_audio_prior_gate import (
    ReverbCandidateAudioPriorGate,
    candidate_audio_features,
)
from .reverb_quality2 import measure, summarize
from .train_ambience4_profile_direct import SEED


def _examples(dataset) -> list[dict]:
    examples = []
    for index in range(len(dataset)):
        row = dataset[index]
        start = int(row["target_start"])
        wet = row["wet"][start:]
        candidate = row["analytic_base"][start:]
        clean = row["clean"][start:]
        result = measure(wet.numpy(), candidate.numpy(), clean.numpy())
        examples.append({
            "features": candidate_audio_features(wet, candidate, row["profile_features"]),
            "controls": row["controls"],
            "exact_mode": torch.tensor([float(row["analytic_mode"] == "exact")]),
            "label": float(result["decision"] == "effective-restoration"),
            "wet": wet.numpy(),
            "candidate": candidate.numpy(),
            "clean": clean.numpy(),
            "source_id": row["source_id"],
            "rir_source_id": row["rir_source_id"],
            "room_group": dataset.room_group(row),
            "decay_stratum": dataset.decay_stratum(
                float(row["control_values"]["decay_p999_seconds"])
            ),
            "analytic_mode": row["analytic_mode"],
            "index": index,
        })
    return examples


def _tensors(rows: list[dict]) -> TensorDataset:
    return TensorDataset(
        torch.stack([row["features"] for row in rows]),
        torch.stack([row["controls"] for row in rows]),
        torch.stack([row["exact_mode"] for row in rows]),
        torch.tensor([row["label"] for row in rows], dtype=torch.float32),
    )


def _scores(model, rows: list[dict], batch_size: int, device: torch.device) -> np.ndarray:
    values = []
    model.eval()
    with torch.inference_mode():
        for features, controls, exact_mode, _ in DataLoader(_tensors(rows), batch_size=batch_size):
            values.extend(model(
                features.to(device), controls.to(device), exact_mode.to(device)
            ).cpu().tolist())
    return np.asarray(values, dtype=np.float64)


def _zero_false_positive_threshold(rows: list[dict], scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    selected = 0
    for rank, index in enumerate(order):
        if not rows[int(index)]["label"]:
            break
        selected = rank + 1
    if not selected:
        return float("inf")
    if selected == len(order):
        return float(scores[order[-1]])
    return 0.5 * float(scores[order[selected - 1]] + scores[order[selected]])


def _deploy(rows: list[dict], scores: np.ndarray, threshold: float) -> dict:
    results = []
    groups = defaultdict(list)
    public = []
    for row, score in zip(rows, scores, strict=True):
        selected = bool(score >= threshold)
        restored = row["candidate"] if selected else row["wet"]
        result = measure(row["wet"], restored, row["clean"])
        results.append(result)
        groups[f"source:{row['source_id']}"].append(result)
        groups[f"room:{row['room_group']}"].append(result)
        groups[f"decay:{row['decay_stratum']}"].append(result)
        public.append({
            "index": row["index"], "score": float(score), "selected": selected,
            "candidate_effective": bool(row["label"]),
            "decision": result["decision"], "source_id": row["source_id"],
            "rir_source_id": row["rir_source_id"], "room_group": row["room_group"],
            "decay_stratum": row["decay_stratum"], "analytic_mode": row["analytic_mode"],
        })
    aggregate = summarize(results)
    group_reports = {name: summarize(values) for name, values in sorted(groups.items())}
    return {
        "accepted": bool(
            aggregate["accepted"] and all(value["accepted"] for value in group_reports.values())
        ),
        "aggregate": aggregate,
        "groups": group_reports,
        "rows": public,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product4-reverb-audio-prior-gate-v5-formal"),
    )
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--fit-samples", type=int, default=960)
    parser.add_argument("--calibration-samples", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace audio-prior gate run: {output}")
    if args.fit_samples < 480 or args.calibration_samples < 96:
        raise ValueError("audio-prior gate needs a substantial fit and calibration audit")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    random.seed(SEED + 50)
    np.random.seed(SEED + 50)
    torch.manual_seed(SEED + 50)
    workspace = args.workspace.resolve()
    fit_dataset = AmbiencePairsV4MixedProfileDirect(
        workspace, "fit", args.fit_samples, 65_536, SEED + 51
    )
    fit_rows = _examples(fit_dataset)
    train_rows = [row for index, row in enumerate(fit_rows) if index % 5]
    threshold_rows = [row for index, row in enumerate(fit_rows) if not index % 5]
    device = torch.device(args.device)
    model = ReverbCandidateAudioPriorGate(args.channels).to(device)
    positives = sum(row["label"] for row in train_rows)
    positive_weight = torch.tensor(
        (len(train_rows) - positives) / max(positives, 1.0), device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    loader = DataLoader(
        _tensors(train_rows), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED + 52),
    )
    history = []
    best = None
    checkpoints = {min(args.epochs, value) for value in (5, 10, 20, 30, args.epochs)}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = examples = 0
        for features, controls, exact_mode, labels in loader:
            features, controls = features.to(device), controls.to(device)
            exact_mode, labels = exact_mode.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, controls, exact_mode)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=positive_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(labels)
            examples += len(labels)
        if epoch in checkpoints:
            threshold_scores = _scores(model, threshold_rows, args.batch_size, device)
            threshold = _zero_false_positive_threshold(threshold_rows, threshold_scores)
            report = _deploy(threshold_rows, threshold_scores, threshold)
            rank = (
                float(report["accepted"]),
                float(report["aggregate"]["safe_fraction"] == 1.0),
                report["aggregate"]["effective_eligible_coverage"],
            )
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            candidate = (rank, -epoch, state, threshold, report)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
            history.append({
                "epoch": epoch, "fit_bce": total / examples,
                "threshold_holdout": report["aggregate"],
            })
            print(json.dumps(history[-1], sort_keys=True), flush=True)
    if best is None:
        raise RuntimeError("audio-prior training produced no selectable checkpoint")
    _, negative_epoch, state, threshold, threshold_report = best
    selected_epoch = -negative_epoch
    model = model.cpu()
    model.load_state_dict(state)
    calibration_dataset = AmbiencePairsV4MixedProfileDirect(
        workspace, "calibration", args.calibration_samples, 65_536, SEED + 2
    )
    calibration_rows = _examples(calibration_dataset)
    calibration_scores = _scores(model, calibration_rows, args.batch_size, torch.device("cpu"))
    calibration_report = _deploy(calibration_rows, calibration_scores, threshold)
    gates = {
        "threshold_fit_holdout_zero_false_positive": (
            threshold_report["aggregate"]["safe_fraction"] == 1.0
        ),
        "calibration_every_example_safe": (
            calibration_report["aggregate"]["safe_fraction"] == 1.0
        ),
        "calibration_effective_coverage": (
            calibration_report["aggregate"]["effective_eligible_coverage"] >= 0.50
        ),
        "all_calibration_groups": all(
            row["accepted"] for row in calibration_report["groups"].values()
        ),
        "development_sealed": True,
        "fresh_rochester_sealed": True,
        "locked_final_sealed": True,
    }
    accepted = all(gates.values())
    output.mkdir(parents=True)
    checkpoint = output / "gate.pt"
    torch.save({
        "schema": 1,
        "architecture": model.manifest(),
        "score_threshold": threshold,
        "state_dict": model.state_dict(),
    }, checkpoint)
    report = {
        "schema": 1,
        "status": "calibration-audio-prior-gate-passed-development-sealed" if accepted else "calibration-audio-prior-gate-rejected",
        "accepted": accepted,
        "usable_model": None,
        "mechanism": "ambience",
        "model": {
            **model.manifest(), "checkpoint": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "score_threshold": threshold,
        },
        "training": {
            "accelerator": args.device,
            "fit_samples": args.fit_samples,
            "train_samples": len(train_rows),
            "threshold_holdout_samples": len(threshold_rows),
            "calibration_samples": args.calibration_samples,
            "epochs": args.epochs,
            "selected_epoch": selected_epoch,
            "history": history,
            "selection_rule": "fit-holdout zero false positives, then maximize effective coverage",
            "calibration_used_for_selection": False,
        },
        "threshold_holdout": threshold_report,
        "calibration": calibration_report,
        "gates": gates,
        "development_opened": False,
        "listening_rows_used": False,
        "fresh_rochester_opened": False,
        "locked_final_accessed": False,
        "quality": {"generated_audio_written": False, "demo_generated": False},
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "accepted": accepted,
        "selected_epoch": selected_epoch, "threshold": threshold,
        "holdout": threshold_report["aggregate"],
        "calibration": calibration_report["aggregate"],
        "gates": gates, "output": str(output),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
