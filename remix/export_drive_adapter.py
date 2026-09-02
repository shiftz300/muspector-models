#!/usr/bin/env python3
"""Export and validate the stateful per-device Drive adapter ONNX contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

from .drive_adapter import DriveDeviceAdapter, load_drive_adapter
from .forward_drive import FORWARD_RATE


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/drive-adapter-pilot"


class AdapterExport(torch.nn.Module):
    def __init__(self, adapter: DriveDeviceAdapter) -> None:
        super().__init__()
        self.adapter = adapter

    def forward(
        self,
        dry: torch.Tensor,
        base: torch.Tensor,
        controls: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.adapter(dry, base, controls, hidden)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _torch_stream(
    adapter: DriveDeviceAdapter,
    dry: np.ndarray,
    base: np.ndarray,
    controls: np.ndarray,
    frames: int,
) -> np.ndarray:
    state = None
    pieces = []
    with torch.inference_mode():
        for start in range(0, dry.shape[1], frames):
            piece, state = adapter(
                torch.from_numpy(dry[:, start : start + frames]),
                torch.from_numpy(base[:, start : start + frames]),
                torch.from_numpy(controls),
                state,
            )
            pieces.append(piece.numpy())
    return np.concatenate(pieces, axis=1)


def _onnx_stream(
    session: onnxruntime.InferenceSession,
    dry: np.ndarray,
    base: np.ndarray,
    controls: np.ndarray,
    hidden_size: int,
    frames: int,
) -> np.ndarray:
    hidden = np.zeros((1, dry.shape[0], hidden_size), dtype=np.float32)
    pieces = []
    for start in range(0, dry.shape[1], frames):
        piece, hidden = session.run(
            ("rendered", "next_hidden"),
            {
                "dry": dry[:, start : start + frames],
                "base": base[:, start : start + frames],
                "controls": controls,
                "hidden": hidden,
            },
        )
        pieces.append(piece)
    return np.concatenate(pieces, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=RUN / "drive-device-adapter.pt")
    parser.add_argument("--output", type=Path, default=RUN / "drive-device-adapter.onnx")
    parser.add_argument("--contract", type=Path, default=RUN / "runtime-contract.json")
    args = parser.parse_args()
    adapter = load_drive_adapter(args.checkpoint).eval()
    wrapper = AdapterExport(adapter).eval()
    dry = torch.zeros((1, 257), dtype=torch.float32)
    base = torch.zeros_like(dry)
    controls = torch.full((1, 3), 0.5, dtype=torch.float32)
    hidden = torch.zeros((1, 1, adapter.hidden_size), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (dry, base, controls, hidden),
        args.output,
        dynamo=False,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=("dry", "base", "controls", "hidden"),
        output_names=("rendered", "next_hidden"),
        dynamic_axes={
            "dry": {0: "batch", 1: "frames"},
            "base": {0: "batch", 1: "frames"},
            "controls": {0: "batch"},
            "hidden": {1: "batch"},
            "rendered": {0: "batch", 1: "frames"},
            "next_hidden": {1: "batch"},
        },
    )
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph, full_check=True)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(args.output),
        sess_options=options,
        providers=("CPUExecutionProvider",),
    )
    rng = np.random.default_rng(20270212)
    source = (rng.standard_normal((3, 8_207)) * 0.05).astype(np.float32)
    generic = np.tanh(source * np.asarray(((1.4,), (2.1,), (3.2,)), dtype=np.float32))
    normalized = np.asarray(((0.1, 0.8, 0.3), (0.5, 0.5, 0.5), (0.9, 0.2, 0.7)), dtype=np.float32)
    torch_full = _torch_stream(adapter, source, generic, normalized, source.shape[1])
    torch_chunked = _torch_stream(adapter, source, generic, normalized, 509)
    onnx_full = _onnx_stream(session, source, generic, normalized, adapter.hidden_size, source.shape[1])
    onnx_chunked = _onnx_stream(session, source, generic, normalized, adapter.hidden_size, 509)

    def comparison(reference: np.ndarray, candidate: np.ndarray) -> dict:
        difference = candidate.astype(np.float64) - reference.astype(np.float64)
        return {
            "max_absolute_error": float(np.max(np.abs(difference))),
            "rms_error": float(np.sqrt(np.mean(np.square(difference)))),
        }

    silence = np.zeros((1, 1_031), dtype=np.float32)
    silence_controls = np.asarray(((1.0, 1.0, 1.0),), dtype=np.float32)
    silence_output = _onnx_stream(
        session,
        silence,
        silence,
        silence_controls,
        adapter.hidden_size,
        257,
    )
    parity = {
        "onnx_vs_torch_full": comparison(torch_full, onnx_full),
        "onnx_vs_torch_chunked": comparison(torch_chunked, onnx_chunked),
        "torch_chunk_vs_full": comparison(torch_full, torch_chunked),
        "onnx_chunk_vs_full": comparison(onnx_full, onnx_chunked),
        "silence_max_absolute_output": float(np.max(np.abs(silence_output))),
    }
    passed = bool(
        parity["onnx_vs_torch_full"]["max_absolute_error"] <= 2.0e-5
        and parity["onnx_vs_torch_chunked"]["max_absolute_error"] <= 2.0e-5
        and parity["torch_chunk_vs_full"]["max_absolute_error"] <= 2.0e-6
        and parity["onnx_chunk_vs_full"]["max_absolute_error"] <= 2.0e-5
        and parity["silence_max_absolute_output"] == 0.0
    )
    contract = {
        "schema": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "model_family": "drive-device-causal-residual-adapter",
        "sample_rate": FORWARD_RATE,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "onnx": str(args.output),
        "onnx_sha256": _sha256(args.output),
        "onnx_opset": 17,
        "hidden_size": adapter.hidden_size,
        "parity": parity,
        "state_policy": "zero once at stream start; carry next_hidden across chunks",
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
