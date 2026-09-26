#!/usr/bin/env python3
"""Audit a bounded WPE candidate bank with a nondeployable Clean oracle."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .ambience2 import blind_late_suppression
from .ambience4 import (
    AmbiencePairsV4,
    DECAY_BANK_VARIANTS,
    TARGET_FRAMES_MINIMUM,
    multiband_decay_suppression,
)
from .train_ambience2 import _summaries


SEED = 20260907
WPE_VARIANTS = ((1, 4), (3, 6), (6, 8), (12, 8))
DECAY_VARIANTS = DECAY_BANK_VARIANTS
N_FFT = 1024
HOP = 256
STRENGTH = 0.375


def _gate_upper(wet: torch.Tensor, spectral_frames: int) -> torch.Tensor:
    envelope = torch.nn.functional.avg_pool1d(
        wet[None, None].square(), 480, 120, padding=240
    ).clamp_min(1.0e-12).sqrt()
    recent_peak = torch.nn.functional.max_pool1d(
        torch.nn.functional.pad(envelope, (80, 0)), 81, 1
    )
    relative_level = envelope / recent_peak.clamp_min(1.0e-6)
    temporal = torch.sigmoid((0.60 - relative_level) / 0.08)
    temporal = temporal * (
        recent_peak >= recent_peak.amax(dim=2, keepdim=True) * 0.03
    )
    temporal = torch.nn.functional.interpolate(
        temporal, size=spectral_frames, mode="linear", align_corners=False
    )[0, 0]
    frequencies = torch.linspace(0.0, 24_000.0, N_FFT // 2 + 1, dtype=wet.dtype)
    spectral = torch.sigmoid((2_000.0 - frequencies) / 400.0)
    return spectral[:, None] * temporal[None, :]


def clean_oracle_candidate_bank(
    wet: torch.Tensor,
    clean: torch.Tensor,
    family: str,
) -> tuple[torch.Tensor, Counter[str]]:
    """Select the lowest-Clean-error bounded candidate independently per TF bin."""
    if wet.ndim != 1 or clean.shape != wet.shape:
        raise ValueError("candidate-bank oracle expects matching mono Wet and Clean")
    window = torch.hann_window(N_FFT, dtype=wet.dtype)
    wet_spectrum = torch.stft(
        wet, N_FFT, HOP, window=window, center=True,
        pad_mode="constant", return_complex=True,
    )
    clean_spectrum = torch.stft(
        clean, N_FFT, HOP, window=window, center=True,
        pad_mode="constant", return_complex=True,
    )
    estimates = [wet_spectrum]
    names = ["wet"]
    rows = []
    if family in {"wpe", "hybrid"}:
        rows.extend([
            (
                f"delay-{delay}-taps-{taps}",
                blind_late_suppression(
                    wet[None], n_fft=N_FFT, hop=HOP, delay=delay,
                    taps=taps, strength=STRENGTH,
                )[0],
                _gate_upper(wet, wet_spectrum.shape[-1]),
            )
            for delay, taps in WPE_VARIANTS
        ])
    if family in {"decay", "hybrid"}:
        rows.extend([
            (
                f"half-{half_life_ms:g}-ratio-{ratio_threshold:g}-strength-{strength:g}",
                multiband_decay_suppression(
                    wet[None], half_life_ms=half_life_ms,
                    ratio_threshold=ratio_threshold, strength=strength,
                    n_fft=N_FFT, hop=HOP,
                )[0],
                torch.ones_like(wet_spectrum.real),
            )
            for half_life_ms, ratio_threshold, strength in DECAY_VARIANTS
        ])
    if not rows:
        raise ValueError(f"unsupported candidate family: {family}")
    for name, candidate, upper in rows:
        candidate_spectrum = torch.stft(
            candidate, N_FFT, HOP, window=window, center=True,
            pad_mode="constant", return_complex=True,
        )
        direction = candidate_spectrum - wet_spectrum
        target = clean_spectrum - wet_spectrum
        projection = (
            target.real * direction.real + target.imag * direction.imag
        ) / direction.abs().square().add(1.0e-8)
        gate = torch.minimum(projection.clamp_min(0.0), upper)
        estimates.append(wet_spectrum + gate * direction)
        names.append(name)
    stack = torch.stack(estimates)
    error = (stack - clean_spectrum[None]).abs().square()
    selected = error.argmin(dim=0)
    restored_spectrum = torch.gather(
        stack, 0, selected[None]
    )[0]
    counts = Counter({
        name: int((selected == index).sum())
        for index, name in enumerate(names)
    })
    restored = torch.istft(
        restored_spectrum, N_FFT, HOP, window=window,
        center=True, length=wet.shape[0],
    )
    return restored, counts


def _add(collection, wet, restored, clean) -> None:
    collection[0].append(wet)
    collection[1].append(restored)
    collection[2].append(clean)


def audit(workspace: Path, samples: int, family: str) -> dict:
    dataset = AmbiencePairsV4(
        workspace, "development", samples, TARGET_FRAMES_MINIMUM, SEED
    )
    aggregate = ([], [], [])
    sources = defaultdict(lambda: ([], [], []))
    rooms = defaultdict(lambda: ([], [], []))
    decays = defaultdict(lambda: ([], [], []))
    selected_bins: Counter[str] = Counter()
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            start = int(row["target_start"])
            restored, counts = clean_oracle_candidate_bank(
                row["wet"], row["clean"], family
            )
            selected_bins.update(counts)
            wet = row["wet"][start:].numpy()
            candidate = restored[start:].numpy().astype(np.float32)
            clean = row["clean"][start:].numpy()
            source = str(row["source_id"])
            room = dataset.room_group(row)
            decay = dataset.decay_stratum(
                float(row["control_values"]["decay_p999_seconds"])
            )
            for collection in (
                aggregate, sources[source], rooms[room], decays[decay]
            ):
                _add(collection, wet, candidate, clean)
    return {
        "schema": 1,
        "status": "diagnostic-clean-oracle-upper-bound-not-deployable",
        "accepted": False,
        "purpose": "candidate-family-viability-only",
        "split": "development",
        "locked_final_opened": False,
        "samples": samples,
        "target_frames": TARGET_FRAMES_MINIMUM,
        "candidate_bank": {
            "family": family,
            "variants": {
                "wpe": (
                    [
                    {"delay": delay, "taps": taps, "strength": STRENGTH}
                    for delay, taps in WPE_VARIANTS
                    ]
                    if family in {"wpe", "hybrid"}
                    else []
                ),
                "decay": (
                    [
                    {
                        "half_life_ms": half_life_ms,
                        "ratio_threshold": ratio_threshold,
                        "strength": strength,
                    }
                    for half_life_ms, ratio_threshold, strength in DECAY_VARIANTS
                    ]
                    if family in {"decay", "hybrid"}
                    else []
                ),
            },
            "wet_identity_reachable": True,
            "candidate_extrapolation": False,
            "clean_input_at_runtime": True,
            "chain_order_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
            "selected_time_frequency_bins": dict(sorted(selected_bins.items())),
        },
        "aggregate": _summaries(aggregate),
        "sources": {name: _summaries(rows) for name, rows in sorted(sources.items())},
        "rooms": {name: _summaries(rows) for name, rows in sorted(rooms.items())},
        "decay_strata": {name: _summaries(rows) for name, rows in sorted(decays.items())},
    }


def _tail_view(report: dict) -> dict:
    def compact(row: dict) -> dict:
        tail = row["tail"]
        return {
            key: tail.get(key)
            for key in (
                "accepted", "eligible", "pass_fraction",
                "median_tail_excess_reduction",
                "median_tail_envelope_esr_improvement",
                "added_reverb_fraction",
            )
        }
    return {
        "aggregate": compact(report["aggregate"]),
        "sources": {name: compact(row) for name, row in report["sources"].items()},
        "rooms": {name: compact(row) for name, row in report["rooms"].items()},
        "decay_strata": {
            name: compact(row) for name, row in report["decay_strata"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--family", choices=("wpe", "decay", "hybrid"), default="wpe")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    output = (
        args.output
        or Path(f"runs/foundation/product4-reverb-{args.family}-bank-oracle-v1/metrics.json")
    ).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace candidate-bank audit: {output}")
    report = audit(args.workspace.resolve(), args.samples, args.family)
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metrics": str(output), **_tail_view(report)}, sort_keys=True))


if __name__ == "__main__":
    main()
