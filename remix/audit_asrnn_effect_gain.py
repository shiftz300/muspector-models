#!/usr/bin/env python3
"""Test whether train-only per-setting gain can repair stable-effect peaks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .asrnn_effects import effect_files
from .evaluate_asrnn_effect import EffectClips
from .stable_effect import load_stable_effect


@torch.inference_mode()
def _render(model, dry: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
    state = None
    chunks = []
    for start in range(0, dry.shape[1], 2_048):
        value, state = model(dry[:, start : start + 2_048], controls, state)
        chunks.append(value)
    return torch.cat(chunks, dim=1)[:, 1_024:]


@torch.inference_mode()
def audit(device: str, checkpoint: Path, corpus: Path) -> dict:
    model, payload = load_stable_effect(checkpoint)
    if payload["device"] != device:
        raise ValueError("gain audit device differs from checkpoint")
    by_setting: dict[tuple[float, ...], list[float]] = {}
    train_loader = DataLoader(
        EffectClips(effect_files(corpus, device, "train"), device), batch_size=32
    )
    for batch in train_loader:
        predicted = _render(model, batch["dry"], batch["controls"])
        wet = batch["wet"][:, 1_024:]
        ratios = wet.abs().amax(1) / predicted.abs().amax(1).clamp_min(1.0e-8)
        for controls, ratio in zip(batch["controls"].tolist(), ratios.tolist()):
            key = tuple(round(value, 4) for value in controls)
            by_setting.setdefault(key, []).append(ratio)
    gains = {key: float(np.median(values)) for key, values in by_setting.items()}

    errors, relative, ratios = [], [], []
    squared_error = target_energy = 0.0
    eval_loader = DataLoader(
        EffectClips(effect_files(corpus, device, "eval"), device), batch_size=32
    )
    for batch in eval_loader:
        predicted = _render(model, batch["dry"], batch["controls"])
        wet = batch["wet"][:, 1_024:]
        gain = torch.tensor(
            [
                gains[tuple(round(value, 4) for value in controls)]
                for controls in batch["controls"].tolist()
            ]
        ).unsqueeze(1)
        predicted = predicted * gain
        target_peak = wet.abs().amax(1)
        predicted_peak = predicted.abs().amax(1)
        absolute = (predicted_peak - target_peak).abs()
        errors.extend(absolute.tolist())
        relative.extend((absolute / target_peak.clamp_min(1.0e-8)).tolist())
        ratios.extend((predicted_peak / target_peak.clamp_min(1.0e-8)).tolist())
        squared_error += float((predicted - wet).square().sum())
        target_energy += float(wet.square().sum())
    return {
        "schema": 1,
        "device": device,
        "method": "train-setting-median-peak-gain-upper-bound",
        "train_setting_peak_gains": {
            str(key): value for key, value in sorted(gains.items())
        },
        "official_eval_global_esr": squared_error / target_energy,
        "official_eval_absolute_peak_error_p95": float(np.quantile(errors, 0.95)),
        "official_eval_relative_peak_error_p95": float(np.quantile(relative, 0.95)),
        "official_eval_peak_ratio_p05_median_p95": np.quantile(
            ratios, (0.05, 0.5, 0.95)
        ).tolist(),
        "passes_frozen_absolute_peak_gate": bool(np.quantile(errors, 0.95) <= 0.02),
        "quality_policy": {
            "source_files_read_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("dfz", "cs3"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.device, args.checkpoint, args.corpus.resolve())
    if args.output.exists():
        raise ValueError(f"gain audit output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
