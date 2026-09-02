#!/usr/bin/env python3
"""Export learned Delay coefficients and verify the stateful DSP contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .forward_delay import DelayForwardRenderer, stream_delay_block
from .forward_drive import FORWARD_RATE


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/delay-forward-pilot"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stream(
    model: DelayForwardRenderer,
    dry: torch.Tensor,
    controls: torch.Tensor,
    chunk_frames: int,
) -> torch.Tensor:
    state = None
    pieces = []
    for start in range(0, dry.shape[1], chunk_frames):
        piece, state = stream_delay_block(
            model,
            dry[:, start : start + chunk_frames],
            controls,
            state,
        )
        pieces.append(piece)
    return torch.cat(pieces, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=RUN / "delay-forward-candidate.pt")
    parser.add_argument("--parameters", type=Path, default=RUN / "delay-forward-parameters.json")
    parser.add_argument("--contract", type=Path, default=RUN / "runtime-contract.json")
    parser.add_argument("--chunk-frames", type=int, default=1_024)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema") != 1 or payload.get("sample_rate") != FORWARD_RATE:
        raise ValueError("Delay forward checkpoint contract is incompatible")
    model = DelayForwardRenderer(fir_taps=int(payload["fir_taps"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    learned = model.learned_parameters()
    parameter_report = {
        "schema": 1,
        "model_family": "delay-forward-hybrid",
        "sample_rate": FORWARD_RATE,
        "controls": ["time_ms", "feedback", "mix"],
        "physical_ranges": {
            "time_ms": [40.0, 1_000.0],
            "feedback": [0.0, 0.9],
            "mix": [0.0, 0.7],
        },
        "learned_repeat_path": learned,
    }
    args.parameters.write_text(json.dumps(parameter_report, indent=2, sort_keys=True) + "\n")

    rng = np.random.default_rng(20261027)
    dry = torch.from_numpy(rng.normal(0.0, 0.12, size=(1, 120_137)).astype(np.float32))
    cases = {
        "minimum_time": torch.tensor(((0.0, 0.0, 0.0),), dtype=torch.float32),
        "typical": torch.tensor(((0.31, 0.63, 0.52),), dtype=torch.float32),
        "maximum_time": torch.tensor(((1.0, 1.0, 1.0),), dtype=torch.float32),
    }
    parity = {}
    with torch.inference_mode():
        for name, controls in cases.items():
            complete = model(dry, controls)
            chunked = _stream(model, dry, controls, args.chunk_frames)
            difference = (chunked - complete).double()
            parity[name] = {
                "max_absolute_error": float(torch.max(torch.abs(difference))),
                "rms_error": float(torch.sqrt(torch.mean(torch.square(difference)))),
            }
    silence = torch.zeros((1, 8_193), dtype=torch.float32)
    with torch.inference_mode():
        silence_output = _stream(model, silence, cases["maximum_time"], args.chunk_frames)
        bypass = _stream(model, dry, cases["minimum_time"], args.chunk_frames)
    passed = bool(
        all(values["max_absolute_error"] <= 2.0e-6 for values in parity.values())
        and float(torch.max(torch.abs(silence_output))) == 0.0
        and float(torch.max(torch.abs(bypass - dry))) == 0.0
    )
    contract = {
        "schema": 1,
        "status": "passed" if passed else "failed",
        "model_family": "delay-forward-hybrid",
        "runtime_kind": "native physical delay plus learned FIR coefficients",
        "onnx_required": False,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "parameters": str(args.parameters),
        "parameters_sha256": _sha256(args.parameters),
        "sample_rate": FORWARD_RATE,
        "sample_format": "float32",
        "maximum_stream_chunk_frames": 1_920,
        "validated_chunk_frames": args.chunk_frames,
        "state": {
            "dry_history_frames": model.fir_taps - 1,
            "delay_line_frames": "round(time_ms * 48)",
            "controls_fixed_during_state_lifetime": True,
        },
        "physical_formulas": {
            "delay_samples": "round(time_ms * 48000 / 1000)",
            "write": "colored_dry + feedback * delayed",
            "output": "dry * (1 - mix) + delayed * mix",
        },
        "parity": parity,
        "bypass_max_absolute_error": float(torch.max(torch.abs(bypass - dry))),
        "silence_max_absolute_output": float(torch.max(torch.abs(silence_output))),
        "audio_quality_policy": {
            "source_audio_immutable": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "automatic_dither": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
        },
    }
    args.contract.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(json.dumps(contract, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
