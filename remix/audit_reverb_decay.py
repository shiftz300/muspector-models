#!/usr/bin/env python3
"""Audit signal-derived Reverb Decay estimates on held-out synthetic pairs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import KIND_INDEX, RATE, SyntheticControlDataset, dry_sources


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/corpus/guitar-effects-chains"
OUTPUT = ROOT / "remix/runs/reverb-decay-audit.json"
ANALYSIS_RATE = RATE // 4


def deconvolved_impulse(
    dry: torch.Tensor, wet: torch.Tensor, regularization: float
) -> np.ndarray:
    """Return the first five seconds of a regularized Clean/Wet transfer estimate."""

    dry_small = torch.nn.functional.avg_pool1d(dry[None, None], 4, 4)[0, 0]
    wet_small = torch.nn.functional.avg_pool1d(wet[None, None], 4, 4)[0, 0]
    required = dry_small.numel() * 2
    padded = 1 << math.ceil(math.log2(required))
    dry_spectrum = torch.fft.rfft(dry_small, n=padded)
    wet_spectrum = torch.fft.rfft(wet_small, n=padded)
    power = dry_spectrum.abs().square()
    transfer = wet_spectrum * dry_spectrum.conj() / (
        power + power.mean() * regularization
    )
    return torch.fft.irfft(transfer, n=padded)[: dry_small.numel()].numpy()


def local_decay_hint(
    impulse: np.ndarray,
    *,
    block: int,
    stop_db: float,
) -> float:
    """Fit the robust local-energy slope and return normalized 0.2-8 s decay."""

    usable = impulse[: (len(impulse) // block) * block]
    energy = np.mean(usable.reshape(-1, block) ** 2, axis=1)
    # Median smoothing suppresses sparse Delay taps without flattening the
    # diffuse Reverb envelope.
    padded = np.pad(energy, (2, 2), mode="edge")
    energy = np.median(np.stack([padded[i : i + len(energy)] for i in range(5)]), axis=0)
    time = (np.arange(len(energy)) + 0.5) * block / ANALYSIS_RATE
    active = (time >= 0.025) & (time <= 4.75)
    reference = float(np.max(energy[active]))
    db = 10.0 * np.log10(np.maximum(energy / max(reference, 1.0e-20), 1.0e-12))
    selected = active & (db <= -3.0) & (db >= stop_db)
    if selected.sum() < 6:
        selected = active & (db <= -1.0) & (db >= stop_db - 10.0)
    if selected.sum() < 3:
        return 0.5
    slope, _ = np.polyfit(time[selected], db[selected], 1)
    decay_s = np.clip(-60.0 / min(slope, -1.0e-3), 0.2, 8.0)
    return float(np.clip(math.log(decay_s / 0.2) / math.log(40.0), 0.0, 1.0))


def topology_name(encoded: torch.Tensor) -> str:
    kinds = {index: kind for kind, index in KIND_INDEX.items()}
    return "->".join(kinds[int(value)] for value in encoded if int(value) >= 0)


def stratum(topology: str) -> str:
    parts = set(topology.split("->"))
    if parts == {"reverb"}:
        return "reverb_only"
    if "delay" in parts and "drive" in parts:
        return "with_delay_and_drive"
    if "delay" in parts:
        return "with_delay"
    if "drive" in parts:
        return "with_drive"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--split", choices=("calibrate", "valid"), default="calibrate")
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    candidates = tuple(
        (regularization, block, stop_db)
        for regularization in (
            1.0e-2,
            3.0e-3,
            1.0e-3,
            3.0e-4,
            1.0e-4,
            3.0e-5,
            1.0e-5,
        )
        for block in (64, 128, 256, 512, 1024)
        for stop_db in (-20.0, -30.0, -40.0)
    )
    errors: dict[tuple[float, int, float], dict[str, list[float]]] = {
        candidate: defaultdict(list) for candidate in candidates
    }
    baselines: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)

    for renderer in ("reference", "alternate"):
        dataset = SyntheticControlDataset(
            dry_sources(args.corpus, args.split),
            args.examples,
            seed=args.seed,
            renderers=(renderer,),
        )
        for index in range(len(dataset)):
            item = dataset[index]
            if not bool(item["control_mask"][6]):
                continue
            topology = topology_name(item["topology"])
            group = stratum(topology)
            truth = float(item["controls"][6])
            truth_s = 0.2 * 40.0**truth
            baselines[renderer].append(abs(0.5 - truth))
            counts[f"{renderer}:{group}"] += 1
            impulses = {
                regularization: deconvolved_impulse(
                    item["dry"], item["wet"], regularization
                )
                for regularization in sorted({value[0] for value in candidates})
            }
            for candidate in candidates:
                regularization, block, stop_db = candidate
                prediction = local_decay_hint(
                    impulses[regularization], block=block, stop_db=stop_db
                )
                prediction_s = 0.2 * 40.0**prediction
                for key in (renderer, f"{renderer}:{group}"):
                    errors[candidate][key].append(abs(prediction - truth))
                    errors[candidate][f"{key}:physical"].append(
                        abs(prediction_s - truth_s)
                    )

    ranking = []
    for candidate, groups in errors.items():
        reference = float(np.mean(groups["reference"]))
        alternate = float(np.mean(groups["alternate"]))
        ranking.append(
            {
                "regularization": candidate[0],
                "block": candidate[1],
                "stop_db": candidate[2],
                "normalized_mae": {
                    key: float(np.mean(value))
                    for key, value in groups.items()
                    if not key.endswith(":physical")
                },
                "physical_mae_s": {
                    key.removesuffix(":physical"): float(np.mean(value))
                    for key, value in groups.items()
                    if key.endswith(":physical")
                },
                "worst_renderer_normalized_mae": max(reference, alternate),
            }
        )
    ranking.sort(key=lambda row: row["worst_renderer_normalized_mae"])
    report = {
        "schema": 1,
        "split": args.split,
        "seed": args.seed,
        "requested_examples_per_renderer": args.examples,
        "counts": dict(sorted(counts.items())),
        "constant_midpoint_normalized_mae": {
            key: float(np.mean(value)) for key, value in baselines.items()
        },
        "candidates": ranking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**report, "candidates": ranking[:5]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
