#!/usr/bin/env python3
"""Audit official unregularized ASRNN RAT checkpoints under Muspector gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .asrnn_data import rat_files
from .train import device
from .train_asrnn_rat_adapter import RatClips


class OfficialConcatenatedLstm(torch.nn.Module):
    def __init__(self, model_args: dict) -> None:
        super().__init__()
        if model_args.get("type") != "LSTM":
            raise ValueError("this audit admits official unregularized LSTM checkpoints only")
        self.input_coef = float(model_args["input_coef"])
        self.layers = int(model_args["num_layers"])
        hidden = int(model_args["hidden_size"])
        condition_size = int(model_args["condition_size"])
        recurrent = [torch.nn.LSTM(1 + condition_size, hidden, batch_first=True)]
        recurrent.extend(
            torch.nn.LSTM(hidden + condition_size, hidden, batch_first=True)
            for _ in range(1, self.layers)
        )
        self.rnn_layers = torch.nn.ModuleList(recurrent)
        self.output_layer = torch.nn.Linear(hidden, 1)

    def forward(self, dry, condition, states=None):
        if states is None:
            states = [None] * self.layers
        repeated = condition.unsqueeze(1).expand(-1, dry.shape[1], -1)
        value = dry.unsqueeze(-1) * self.input_coef
        next_states = []
        for layer, state in zip(self.rnn_layers, states):
            value, next_state = layer(torch.cat((value, repeated), dim=2), state)
            next_states.append(next_state)
        return self.output_layer(value).squeeze(-1), next_states


@torch.inference_mode()
def evaluate(checkpoint: Path, corpus: Path, frames: int = 2_048) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = OfficialConcatenatedLstm(payload["model_args"])
    model.load_state_dict(payload["model_state"], strict=True)
    target = device()
    model = model.to(target).eval()
    loader = DataLoader(RatClips(rat_files(corpus, "eval")), batch_size=32)
    per_file, ratios, quiet_peaks = [], [], []
    total_error = 0.0
    total_target = 0.0
    for batch in loader:
        dry = batch["dry"].to(target)
        wet = batch["wet"].to(target)
        controls = batch["controls"].to(target)
        # Loader exposes Muspector Tone; the official model expects raw RAT Filter.
        physical = controls.clone()
        physical[:, 1] = 1.0 - physical[:, 1]
        condition = physical.mul(2.0).sub(1.0)
        states = None
        chunks = []
        for start in range(0, dry.shape[1], frames):
            stop = min(start + frames, dry.shape[1])
            prediction, states = model(dry[:, start:stop], condition, states)
            chunks.append(prediction)
        prediction = torch.cat(chunks, dim=1)[:, 1_024:]
        wet = wet[:, 1_024:]
        error = (prediction - wet).square()
        energy = wet.square()
        total_error += float(error.sum())
        total_target += float(energy.sum())
        per_file.extend((error.mean(dim=1) / energy.mean(dim=1).clamp_min(1.0e-8)).cpu().tolist())
        target_peak = wet.abs().amax(dim=1)
        prediction_peak = prediction.abs().amax(dim=1)
        audible = target_peak >= 1.0e-3
        ratios.extend((prediction_peak[audible] / target_peak[audible]).cpu().tolist())
        quiet_peaks.extend(prediction_peak[~audible].cpu().tolist())
    silence = torch.zeros(3, 4_096, device=target)
    conditions = torch.tensor(((-1.0, -1.0, -1.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), device=target)
    silent_output, _ = model(silence, conditions)
    return {
        "checkpoint": str(checkpoint),
        "official_global_esr": total_error / max(total_target, 1.0e-12),
        "strict_mean_per_file_esr": float(np.mean(per_file)),
        "strict_median_per_file_esr": float(np.median(per_file)),
        "strict_p95_per_file_esr": float(np.quantile(per_file, 0.95)),
        "peak_ratio_median": float(np.median(ratios)),
        "peak_ratio_p95": float(np.quantile(ratios, 0.95)),
        "quiet_prediction_peak_maximum": max(quiet_peaks, default=0.0),
        "static_silence_max_absolute_output": float(silent_output.abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.checkpoints.glob("LSTM-1-64-GFB-*.pt"))
    if not paths:
        raise ValueError("no official LSTM-1-64-GFB checkpoints found")
    results = [evaluate(path, args.corpus) for path in paths]
    report = {
        "schema": 1,
        "scope": "official unregularized development baseline; not a Muspector release model",
        "checkpoints": results,
        "best_official_global_esr": min(row["official_global_esr"] for row in results),
        "physical_audio_devices_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
