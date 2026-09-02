#!/usr/bin/env python3
"""Measure offline PyTorch throughput for the canonical stable RAT pilot."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from .stable_rat import load_stable_rat


@torch.inference_mode()
def benchmark(checkpoint: Path, iterations: int, block_frames: int) -> dict:
    if iterations <= 0 or block_frames <= 0:
        raise ValueError("benchmark iterations and block size must be positive")
    model = load_stable_rat(checkpoint)
    torch.manual_seed(20260830)
    dry = torch.randn(1, 48_000) * 0.04
    controls = torch.tensor(((0.5, 0.5, 0.5),))
    model(dry, controls)
    started = time.perf_counter()
    for _ in range(iterations):
        model(dry, controls)
    full_seconds = (time.perf_counter() - started) / iterations
    started = time.perf_counter()
    for _ in range(iterations):
        state = None
        for offset in range(0, dry.shape[1], block_frames):
            _, state = model(dry[:, offset : offset + block_frames], controls, state)
    streaming_seconds = (time.perf_counter() - started) / iterations
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "schema": 1,
        "device": "cpu",
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "parameters": parameters,
        "parameter_bytes_float32": parameters * 4,
        "state_floats_per_mono_stream": model.layers * model.hidden_size * 2,
        "full_second_render_seconds": full_seconds,
        "full_realtime_factor": full_seconds,
        "streaming_block_frames": block_frames,
        "streaming_second_render_seconds": streaming_seconds,
        "streaming_realtime_factor": streaming_seconds,
        "pytorch_cpu_realtime_passed": full_seconds < 1.0 and streaming_seconds < 1.0,
        "runtime_promotion_ready": False,
        "runtime_note": "offline model accepted; optimized export/runtime is a later phase",
        "physical_audio_devices_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--block-frames", type=int, default=1_024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark(args.checkpoint, args.iterations, args.block_frames)
    if args.output.exists():
        raise ValueError(f"stable RAT benchmark already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
