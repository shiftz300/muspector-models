#!/usr/bin/env python3
"""Contrastively train the Wet-only EGDB-PG tone encoder on MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .egdb_pg_amp_data import EgdbPgAmpPairs, RATE
from .egdb_pg_tone_encoder import WetToneEncoder


SEED = 20260906
OBSERVATION_FRAMES = 3 * RATE
CONTEXT_FRAMES = 1024


class TonePairs(Dataset):
    def __init__(self, workspace: Path, split: str, pairs: int, seed: int) -> None:
        probe = EgdbPgAmpPairs(
            workspace, split, 1, OBSERVATION_FRAMES, CONTEXT_FRAMES, seed
        )
        self.profile_count = len(probe.profile_rows)
        self.base = EgdbPgAmpPairs(
            workspace, split, max(pairs * 2, pairs + 7) * self.profile_count,
            OBSERVATION_FRAMES, CONTEXT_FRAMES, seed,
        )
        self.pairs = pairs * self.profile_count
        self.authorization = self.base.authorization

    def __len__(self) -> int:
        return self.pairs

    def __getitem__(self, index: int) -> dict:
        profile = index % self.profile_count
        cycle = index // self.profile_count
        left = self.base[cycle * self.profile_count + profile]
        right = self.base[(cycle + 7) * self.profile_count + profile]
        if left["profile_id"] != right["profile_id"] or left["track"] == right["track"]:
            raise ValueError("tone positive pair must share profile but not content")
        start, end = left["crop_start"], left["crop_end"]
        right_start, right_end = right["crop_start"], right["crop_end"]
        return {
            "left": left["wet"][start:end],
            "right": right["wet"][right_start:right_end],
            "profile_id": left["profile_id"],
            "category": left["category"],
        }


def _collate(rows: list[dict]) -> dict:
    profile_names = sorted({row["profile_id"] for row in rows})
    index = {name: value for value, name in enumerate(profile_names)}
    labels = torch.tensor([index[row["profile_id"]] for row in rows], dtype=torch.long)
    return {
        "wet": torch.cat((
            torch.stack([row["left"] for row in rows]),
            torch.stack([row["right"] for row in rows]),
        )),
        "labels": torch.cat((labels, labels)),
        "profile_id": [row["profile_id"] for row in rows] * 2,
    }


def _contrastive(
    embedding: torch.Tensor, labels: torch.Tensor, temperature: float, margin: float,
) -> torch.Tensor:
    cosine = embedding @ embedding.T
    similarity = cosine / temperature
    diagonal = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    if not positives.any(dim=1).all():
        raise ValueError("every tone embedding needs a positive view")
    logits = similarity.masked_fill(diagonal, float("-inf"))
    log_probability = similarity - torch.logsumexp(logits, dim=1, keepdim=True)
    supervised = -(log_probability.masked_fill(~positives, 0.0).sum(1) / positives.sum(1)).mean()
    negatives = ~labels[:, None].eq(labels[None, :]) & ~diagonal
    hardest_positive = cosine.masked_fill(~positives, 2.0).amin(1)
    hardest_negative = cosine.masked_fill(~negatives, -2.0).amax(1)
    hard_margin = torch.relu(hardest_negative - hardest_positive + margin).mean()
    return supervised + hard_margin


def _mean_loss(model, dataset, batch_size: int, device, temperature: float, margin: float) -> float:
    values = []
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate):
            labels = batch["labels"].to(device)
            values.append(float(_contrastive(model(batch["wet"].to(device)), labels, temperature, margin)))
    return float(np.mean(values))


def _retrieval(model, dataset, batch_size: int) -> dict:
    embeddings, names = [], []
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate):
            embeddings.append(model(batch["wet"]).cpu())
            names.extend(batch["profile_id"])
    embedding = torch.cat(embeddings)
    similarity = embedding @ embedding.T
    similarity.fill_diagonal_(-2.0)
    nearest = similarity.argmax(1).tolist()
    accuracy = float(np.mean([names[row] == names[column] for row, column in enumerate(nearest)]))
    positive, negative = [], []
    for row in range(len(names)):
        for column in range(row + 1, len(names)):
            target = positive if names[row] == names[column] else negative
            target.append(float(similarity[row, column]))
    return {
        "views": len(names),
        "profiles": len(set(names)),
        "leave_one_view_out_top1": accuracy,
        "positive_cosine_median": float(np.median(positive)),
        "negative_cosine_p95": float(np.percentile(negative, 95)),
        "cosine_margin": float(np.median(positive) - np.percentile(negative, 95)),
    }


def train(args) -> dict:
    workspace, output = args.workspace.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace tone run: {output}")
    fit = TonePairs(workspace, "fit", args.fit_pairs_per_profile, SEED + 1)
    calibration = TonePairs(workspace, "calibration", args.calibration_pairs_per_profile, SEED + 2)
    device = torch.device(args.device)
    model = WetToneEncoder(args.embedding_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    initial = _mean_loss(model, calibration, args.batch_size, device, args.temperature, args.margin)
    best, best_epoch = initial, 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history = [{"epoch": 0, "calibration_loss": initial}]
    print(json.dumps(history[-1]), flush=True)
    loader = DataLoader(
        fit, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED), collate_fn=_collate,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _contrastive(
                model(batch["wet"].to(device)), batch["labels"].to(device),
                args.temperature, args.margin,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        calibration_loss = _mean_loss(
            model, calibration, args.batch_size, device, args.temperature, args.margin
        )
        history.append({
            "epoch": epoch, "train_loss": float(np.mean(train_losses)),
            "calibration_loss": calibration_loss,
        })
        if calibration_loss < best:
            best, best_epoch = calibration_loss, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(json.dumps(history[-1]), flush=True)
    model = model.cpu()
    model.load_state_dict(best_state)
    retrieval = _retrieval(model, calibration, args.batch_size)
    sample = torch.zeros(1, OBSERVATION_FRAMES)
    with torch.inference_mode():
        model(sample)
        started = time.perf_counter()
        for _ in range(3):
            model(sample)
        elapsed = (time.perf_counter() - started) / 3
    runtime = {"observation_seconds": 3.0, "mean_seconds": elapsed, "realtime_factor": elapsed / 3.0}
    gates = {
        "calibration_improved": best_epoch > 0 and best <= initial * 0.80,
        "content_invariant_retrieval": retrieval["leave_one_view_out_top1"] >= 0.80,
        "cosine_margin": retrieval["cosine_margin"] >= 0.20,
        "cpu_budget": runtime["realtime_factor"] <= 0.10,
        "wet_only": model.manifest()["profile_id_input"] is False,
        "locked_final_unopened": calibration.base.audit["locked_final_downloaded"] is False,
    }
    accepted = all(gates.values())
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    torch.save({"schema": 1, "architecture": model.manifest(), "state_dict": model.state_dict()}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 1,
        "status": "accepted-tone-representation" if accepted else "diagnostic-tone-representation",
        "accepted": accepted,
        "not_a_restoration_model": True,
        "model": {**model.manifest(), "checkpoint": str(checkpoint), "sha256": digest},
        "training": {
            "accelerator": device.type, "epochs": args.epochs,
            "selected_epoch": best_epoch, "initial_calibration_loss": initial,
            "selected_calibration_loss": best, "temperature": args.temperature,
            "hard_cosine_margin": args.margin,
            "fit_pairs_per_profile": args.fit_pairs_per_profile,
            "calibration_pairs_per_profile": args.calibration_pairs_per_profile,
            "history": history,
        },
        "calibration_retrieval": retrieval,
        "runtime": runtime,
        "gates": gates,
        "data": {
            "authorized_sources": fit.authorization["sources"],
            "required_attribution": fit.authorization["required_attribution"],
            "fit_profiles": fit.profile_count,
            "development_audio_opened": False,
            "locked_final_audio_opened": False,
        },
        "limitations": [
            "representation acceptance does not accept an inverse model",
            "calibration contains the same profiles but disjoint performances",
            "unseen-profile utility must be tested only inside the next frozen restoration model",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "accepted": accepted, "retrieval": retrieval, "gates": gates}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--fit-pairs-per-profile", type=int, default=8)
    parser.add_argument("--calibration-pairs-per-profile", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--margin", type=float, default=0.20)
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
