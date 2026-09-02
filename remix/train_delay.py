#!/usr/bin/env python3
"""Train the Delay Feedback/Mix residual companion across DSP domains."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import soundfile
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import RATE, SAMPLES, SyntheticControlDataset, TOPOLOGIES, _waveform, dry_sources
from .delay_model import DelayControlEstimator, delay_features
from .pedalboard_renderer import render_pedalboard_chain
from .render import render_chain
from .spec import ChainSpec, Delay
from .train import CORPUS, RUN, device


@torch.no_grad()
def feature_matrix(
    corpus: Path,
    split: str,
    samples: int,
    seed: int,
    renderers: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = SyntheticControlDataset(
        dry_sources(corpus, split), samples, seed=seed, renderers=renderers
    )
    rows, targets = [], []
    pending_dry, pending_wet, pending_targets = [], [], []

    def flush() -> None:
        if not pending_dry:
            return
        rows.append(delay_features(torch.stack(pending_dry), torch.stack(pending_wet)).cpu())
        targets.append(torch.stack(pending_targets))
        pending_dry.clear()
        pending_wet.clear()
        pending_targets.clear()

    for index in range(len(dataset)):
        if "delay" not in TOPOLOGIES[index % len(TOPOLOGIES)]:
            continue
        item = dataset[index]
        pending_dry.append(item["dry"])
        pending_wet.append(item["wet"])
        pending_targets.append(item["controls"][4:6])
        if len(pending_dry) == 8:
            flush()
    flush()
    return torch.cat(rows), torch.cat(targets)


@torch.no_grad()
def delay_only_feature_matrix(
    corpus: Path,
    split: str,
    samples: int,
    seed: int,
    renderer: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a balanced Delay-only set for the known-family inverse path."""

    sources = dry_sources(corpus, split)
    rows, targets = [], []
    pending_dry, pending_wet, pending_targets = [], [], []

    def flush() -> None:
        if not pending_dry:
            return
        rows.append(delay_features(torch.stack(pending_dry), torch.stack(pending_wet)).cpu())
        targets.append(torch.stack(pending_targets))
        pending_dry.clear()
        pending_wet.clear()
        pending_targets.clear()

    for index in range(samples):
        rng = random.Random(seed + index * 104_729)
        source = sources[index % len(sources)]
        info = soundfile.info(source)
        frames = round(info.frames * RATE / info.samplerate)
        offset = rng.randrange(max(1, frames - SAMPLES + 1))
        dry = _waveform(source, offset)
        peak = max(float(np.max(np.abs(dry))), 1.0e-5)
        dry = np.asarray(dry * (0.22 / peak), dtype=np.float32)
        effect = Delay(
            40.0 * 25.0 ** rng.random(),
            rng.uniform(0.0, 0.85),
            rng.uniform(0.05, 0.7),
        )
        spec = ChainSpec((effect,))
        wet = (
            render_pedalboard_chain(dry, spec, RATE)
            if renderer == "pedalboard"
            else render_chain(dry, spec, RATE, renderer)
        )
        pending_dry.append(torch.from_numpy(dry.copy()))
        pending_wet.append(torch.from_numpy(wet.copy()))
        pending_targets.append(
            torch.tensor((effect.feedback / 0.9, effect.mix / 0.7), dtype=torch.float32)
        )
        if len(pending_dry) == 8:
            flush()
    flush()
    return torch.cat(rows), torch.cat(targets)


@torch.no_grad()
def evaluate(
    model: DelayControlEstimator,
    features: torch.Tensor,
    targets: torch.Tensor,
    target_device: torch.device,
) -> dict[str, float]:
    model.eval()
    prediction = []
    for (batch,) in DataLoader(TensorDataset(features), batch_size=128):
        prediction.append(torch.sigmoid(model(batch.to(target_device))).cpu())
    absolute = torch.abs(torch.cat(prediction) - targets).mean(dim=0)
    return {"feedback": float(absolute[0]), "mix": float(absolute[1])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--train-samples", type=int, default=19_200)
    parser.add_argument("--valid-samples", type=int, default=1_600)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    random.seed(20260830)
    np.random.seed(20260830)
    torch.manual_seed(20260830)
    training_renderers = ("reference", "alternate", "stress")
    train = feature_matrix(
        args.corpus, "train", args.train_samples, 20260830, training_renderers
    )
    calibration = {
        renderer: feature_matrix(
            args.corpus, "calibrate", args.valid_samples, 20260901, (renderer,)
        )
        for renderer in training_renderers
    }
    validation = {
        renderer: feature_matrix(
            args.corpus, "valid", args.valid_samples, 20260831, (renderer,)
        )
        for renderer in training_renderers
    }
    target_device = device()
    model = DelayControlEstimator().to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=1.0e-4)
    loader = DataLoader(TensorDataset(*train), batch_size=64, shuffle=True)
    best_score, best_state = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        for features, targets in loader:
            logits = model(features.to(target_device))
            loss = torch.nn.functional.smooth_l1_loss(
                torch.sigmoid(logits), targets.to(target_device), beta=0.05
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        current = {
            renderer: evaluate(model, *values, target_device)
            for renderer, values in calibration.items()
        }
        score = max(sum(values.values()) / len(values) for values in current.values())
        if score < best_score:
            best_score = score
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(json.dumps({"epoch": epoch + 1, "calibration": current}, sort_keys=True))
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    calibration_metrics = {
        renderer: evaluate(model, *values, target_device)
        for renderer, values in calibration.items()
    }
    validation_metrics = {
        renderer: evaluate(model, *values, target_device)
        for renderer, values in validation.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "delay-estimator.pt"
    torch.save(best_state, checkpoint)
    report = {
        "schema": 1,
        "train_examples": len(train[0]),
        "calibration_examples_per_renderer": len(next(iter(calibration.values()))[0]),
        "validation_examples_per_renderer": len(next(iter(validation.values()))[0]),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "calibration_normalized_mae": calibration_metrics,
        "validation_normalized_mae": validation_metrics,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    (args.output / "delay-metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
