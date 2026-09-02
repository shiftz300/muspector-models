#!/usr/bin/env python3
"""Validate the native partitioned-convolution contract for Reverb profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .forward_reverb import load_reverb_profile, profile_manifest, stream_reverb_block
from .spec import Reverb


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/reverb-forward-phase1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=RUN / "reverb-device-profile.npz")
    parser.add_argument("--output", type=Path, default=RUN / "runtime-contract.json")
    parser.add_argument("--chunk-frames", type=int, default=1_024)
    args = parser.parse_args()
    profile = load_reverb_profile(args.profile)
    rng = np.random.default_rng(20261201)
    dry = (rng.standard_normal(args.chunk_frames * 8) * 0.05).astype(np.float32)
    parity = {}
    for name, effect in {
        "minimum": Reverb(0.2, 0.0, 0.7),
        "typical": Reverb(2.4, 0.55, 0.43),
        "maximum": Reverb(8.0, 1.0, 0.7),
    }.items():
        full = profile.render(dry, effect)
        state = None
        chunks = []
        for start in range(0, len(dry), args.chunk_frames):
            rendered, state = stream_reverb_block(
                profile, dry[start : start + args.chunk_frames], effect, state
            )
            chunks.append(rendered)
        streamed = np.concatenate(chunks)
        difference = streamed.astype(np.float64) - full.astype(np.float64)
        parity[name] = {
            "max_absolute_error": float(np.max(np.abs(difference))),
            "rms_error": float(np.sqrt(np.mean(np.square(difference)))),
        }
    bypass = profile.render(dry, Reverb(3.0, 0.5, 0.0))
    silence = profile.render(np.zeros_like(dry), Reverb(8.0, 1.0, 0.7))
    maximum_error = max(value["max_absolute_error"] for value in parity.values())
    passed = bool(
        maximum_error <= 2.0e-6
        and float(np.max(np.abs(bypass - dry))) == 0.0
        and float(np.max(np.abs(silence))) == 0.0
    )
    report = {
        "schema": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        **profile_manifest(profile, args.profile),
        "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        "runtime": {
            "implementation": "native uniform partitioned causal FIR convolution",
            "python_reference": "history-carrying FFT convolution",
            "onnx_required": False,
            "chunk_frames_validated": args.chunk_frames,
            "controls_fixed_while_state_is_reused": True,
            "tail_frames": len(profile.mean_ir),
        },
        "parity": parity,
        "bypass_max_absolute_error": float(np.max(np.abs(bypass - dry))),
        "silence_max_absolute_output": float(np.max(np.abs(silence))),
        "audio_quality_policy": {
            "source_files_read_only": True,
            "runtime_automatic_normalization": False,
            "runtime_limiting": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
