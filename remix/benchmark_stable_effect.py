#!/usr/bin/env python3
"""Offline CPU throughput and dynamic-control safety for stable effect pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from .stable_effect import load_stable_effect


@torch.inference_mode()
def benchmark(checkpoint: Path, iterations: int = 3, block_frames: int = 1024) -> dict:
    if iterations <= 0 or block_frames <= 0:
        raise ValueError("iterations and block frames must be positive")
    model, payload = load_stable_effect(checkpoint)
    torch.manual_seed(20260903)
    dry = torch.randn(1, 48000) * 0.04
    controls = torch.full((1, model.control_count), 0.5)
    model(dry, controls)
    started = time.perf_counter()
    for _ in range(iterations):
        model(dry, controls)
    full_rtf = (time.perf_counter() - started) / iterations
    started = time.perf_counter()
    for _ in range(iterations):
        state = None
        for offset in range(0, dry.shape[1], block_frames):
            _, state = model(dry[:, offset:offset + block_frames], controls, state)
    stream_rtf = (time.perf_counter() - started) / iterations
    time_axis = torch.arange(8192) / 48000.0
    dynamic = torch.sin(time_axis * 61).mul(0.5).add(0.5)[None, :, None]
    dynamic = dynamic.expand(1, -1, model.control_count)
    signal = torch.sin(time_axis * (997 * 2 * torch.pi))[None] * 0.1
    whole, _ = model(signal, dynamic)
    chunks, state, start = [], None, 0
    for stop in (17, 513, 2121, 8192):
        value, state = model(signal[:, start:stop], dynamic[:, start:stop], state)
        chunks.append(value)
        start = stop
    stream_error = float((whole - torch.cat(chunks, 1)).abs().max())
    quiet, _ = model(signal * 1.0e-5, dynamic)
    silence, _ = model(torch.zeros_like(signal), dynamic)
    safety = bool(stream_error <= 2.0e-6 and float(quiet.abs().max()) <= 1.0e-3 and float(silence.abs().max()) == 0.0)
    return {
        "schema": 1,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "effect": payload["device"],
        "compute_device": "cpu",
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "state_floats_per_mono_stream": model.state_floats_per_mono_stream,
        "full_realtime_factor": full_rtf,
        "streaming_block_frames": block_frames,
        "streaming_realtime_factor": stream_rtf,
        "pytorch_cpu_realtime_passed": full_rtf < 1.0 and stream_rtf < 1.0,
        "dynamic_control_signal_stream_max_error": stream_error,
        "quiet_input_peak": float(quiet.abs().max()),
        "dynamic_control_silence_peak": float(silence.abs().max()),
        "additional_safety_passed": safety,
        "physical_audio_devices_used": False,
        "ui_integration_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1, help="CPU intra-op threads; mono recurrent inference defaults to one")
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("thread count must be positive")
    torch.set_num_threads(args.threads)
    if args.output.exists():
        raise ValueError(f"benchmark output already exists: {args.output}")
    report = benchmark(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["additional_safety_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
