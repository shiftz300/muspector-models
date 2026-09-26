#!/usr/bin/env python3
"""Seal full-path CPU runtime for the frozen Product4 Reverb frequency bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import fftconvolve

from .ambience2 import _controls, _render
from .ambience4_frequency_shaping import (
    DEFAULT_LOOKAHEAD_FRAMES,
    frequency_profile_candidate_kernels,
    partitioned_frequency_profile_candidate_bank,
)
from .ambience_model4 import AmbienceFrequencyProfileBankExpert
from .openslr26_rir_data import (
    DECAY_DOMAIN_SECONDS,
    load_openslr26_rir,
    product_rir_splits,
)


SEED = 20260909


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean_seconds(operation, repeats: int) -> float:
    operation()
    started = time.perf_counter()
    for _ in range(repeats):
        operation()
    return (time.perf_counter() - started) / repeats


def audit(
    workspace: Path,
    checkpoint: Path,
    *,
    frames: int,
    block_frames: int,
    repeats: int,
) -> dict:
    workspace = workspace.resolve()
    checkpoint = checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    architecture = payload["architecture"]
    model = AmbienceFrequencyProfileBankExpert(
        channels=architecture["channels"],
        depth=architecture["depth"],
        n_fft=architecture["n_fft"],
        hop=architecture["hop"],
    ).eval()
    model.load_state_dict(payload["state_dict"])
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))

    source_root = workspace / "data/corpus/openslr26-simulated-rir"
    development_rirs = product_rir_splits(source_root)[0]["development"]
    profile_path = development_rirs[0]
    profile_relative = profile_path.relative_to(source_root.resolve())
    impulse = load_openslr26_rir(profile_path)
    rng = np.random.default_rng(SEED)
    clean = rng.normal(0.0, 0.015, frames).astype(np.float32)
    wet, control_values = _render(clean, impulse, random.Random(SEED))
    controls = torch.from_numpy(
        _controls(control_values, DECAY_DOMAIN_SECONDS)
    )[None]
    wet_tensor = torch.from_numpy(wet)[None]

    started = time.perf_counter()
    kernels, kernel_reports = frequency_profile_candidate_kernels(
        impulse, control_values
    )
    profile_setup_seconds = time.perf_counter() - started
    offline = np.stack([
        fftconvolve(wet, kernel, mode="full")[
            DEFAULT_LOOKAHEAD_FRAMES : DEFAULT_LOOKAHEAD_FRAMES + len(wet)
        ]
        for kernel in kernels
    ]).astype(np.float32)

    def candidate_operation() -> np.ndarray:
        return partitioned_frequency_profile_candidate_bank(
            wet, kernels, block_frames=block_frames
        )

    partitioned = candidate_operation()
    parity_error = float(np.max(np.abs(partitioned - offline)))

    def selector_operation(bank: np.ndarray = partitioned) -> torch.Tensor:
        with torch.inference_mode():
            return model(wet_tensor, controls, torch.from_numpy(bank)[None])[0]

    candidate_seconds = _mean_seconds(candidate_operation, repeats)
    selector_seconds = _mean_seconds(selector_operation, repeats)

    def full_operation() -> torch.Tensor:
        bank = candidate_operation()
        return selector_operation(bank)

    full_seconds = _mean_seconds(full_operation, repeats)
    output = full_operation()
    duration_seconds = frames / 48_000.0
    full_rtf = full_seconds / duration_seconds
    accepted = bool(
        parity_error <= 2.0e-6
        and full_rtf <= 0.5
        and torch.isfinite(output).all()
        and float(output.abs().max()) <= 1.05
    )
    return {
        "schema": 1,
        "status": (
            "runtime-sealed-development-candidate-not-promoted"
            if accepted else "runtime-rejected-development-candidate"
        ),
        "accepted": accepted,
        "checkpoint": {
            "path": str(checkpoint.relative_to(workspace)),
            "sha256": _sha256(checkpoint),
        },
        "profile": {
            "source_id": "openslr26-simulated-rir-external-v1",
            "split": "development",
            "path": str(Path("data/corpus/openslr26-simulated-rir") / profile_relative),
            "locked_final": False,
            "profile_setup_seconds": profile_setup_seconds,
            "setup_cache_scope": "once per current RIR and control state",
            "candidate_count": len(kernel_reports),
        },
        "runtime": {
            "ordinary_cpu": True,
            "audio_callback": False,
            "frames": frames,
            "duration_seconds": duration_seconds,
            "partition_block_frames": block_frames,
            "partition_algorithm": "finite-overlap-add",
            "candidate_generation_seconds": candidate_seconds,
            "selector_seconds": selector_seconds,
            "full_path_seconds": full_seconds,
            "full_path_realtime_factor": full_rtf,
            "budget_realtime_factor": 0.5,
            "profile_candidate_generation_included": True,
            "profile_setup_included_per_clip": False,
        },
        "equivalence": {
            "reference": "scipy fftconvolve full finite kernel",
            "partitioned_max_absolute_error": parity_error,
            "maximum_allowed_error": 2.0e-6,
            "passed": parity_error <= 2.0e-6,
        },
        "output": {
            "finite": bool(torch.isfinite(output).all()),
            "peak": float(output.abs().max()),
        },
        "inference_contract": {
            "inputs": ["Wet", "current RIR/profile", "current mix", "current room gain"],
            "clean_input": False,
            "graph_order_input": False,
            "chain_order_input": False,
            "neighbor_effect_input": False,
        },
        "promotion_blockers": [
            "listening acceptance is incomplete",
            "fresh physical-room validation and locked-final remain unopened",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/foundation/product4-reverb-frequency-profile-v3-mixed-formal/"
            "ambience/model.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/foundation/product4-reverb-frequency-runtime-v1/metrics.json"),
    )
    parser.add_argument("--frames", type=int, default=173_536)
    parser.add_argument("--block-frames", type=int, default=8_191)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.frames < 65_536 or args.block_frames < 256 or args.repeats < 1:
        raise ValueError("runtime audit geometry is too small")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace runtime audit: {output}")
    report = audit(
        args.workspace, args.checkpoint,
        frames=args.frames, block_frames=args.block_frames, repeats=args.repeats,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "accepted": report["accepted"],
        "runtime": report["runtime"],
        "equivalence": report["equivalence"],
        "output": str(output),
    }, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
