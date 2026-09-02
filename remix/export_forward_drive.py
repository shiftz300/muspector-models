"""Export and numerically validate the causal Drive forward renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

from .forward_drive import FORWARD_RATE, DriveForwardRenderer


RUN = Path(__file__).parent / "runs" / "drive-forward-pilot"
CONTROL_NAMES = ("gain_db", "tone", "level_db")
CONTROL_RANGES = ((0.0, 30.0), (0.0, 1.0), (-18.0, 12.0))


class StatefulDriveExport(torch.nn.Module):
    """Expose recurrent state as explicit ONNX inputs and outputs."""

    def __init__(self, renderer: DriveForwardRenderer) -> None:
        super().__init__()
        self.renderer = renderer

    def forward(
        self,
        dry: torch.Tensor,
        controls: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rendered, (next_hidden, next_cell) = self.renderer(
            dry,
            controls,
            (hidden, cell),
        )
        return rendered, next_hidden, next_cell


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(checkpoint: Path) -> DriveForwardRenderer:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema") != 1:
        raise ValueError("unsupported forward Drive checkpoint schema")
    if payload.get("sample_rate") != FORWARD_RATE:
        raise ValueError("forward Drive checkpoint has the wrong sample rate")
    if tuple(payload.get("controls", ())) != CONTROL_NAMES:
        raise ValueError("forward Drive checkpoint has an incompatible control contract")
    model = DriveForwardRenderer(hidden_size=int(payload["hidden_size"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def _export(model: DriveForwardRenderer, output: Path, opset: int) -> None:
    wrapper = StatefulDriveExport(model).eval()
    batch = 2
    frames = 257
    dry = torch.zeros(batch, frames, dtype=torch.float32)
    controls = torch.full((batch, 3), 0.5, dtype=torch.float32)
    hidden = torch.zeros(1, batch, model.hidden_size, dtype=torch.float32)
    cell = torch.zeros_like(hidden)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dry, controls, hidden, cell),
        output,
        dynamo=False,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=("dry", "controls", "hidden", "cell"),
        output_names=("rendered", "next_hidden", "next_cell"),
        dynamic_axes={
            "dry": {0: "batch", 1: "frames"},
            "controls": {0: "batch"},
            "hidden": {1: "batch"},
            "cell": {1: "batch"},
            "rendered": {0: "batch", 1: "frames"},
            "next_hidden": {1: "batch"},
            "next_cell": {1: "batch"},
        },
    )
    graph = onnx.load(output)
    onnx.checker.check_model(graph, full_check=True)


def _session(path: Path) -> onnxruntime.InferenceSession:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(
        str(path),
        sess_options=options,
        providers=("CPUExecutionProvider",),
    )


def _torch_render(
    model: DriveForwardRenderer,
    dry: np.ndarray,
    controls: np.ndarray,
    chunk_frames: int,
) -> np.ndarray:
    state = None
    pieces = []
    with torch.inference_mode():
        control_tensor = torch.from_numpy(controls)
        for start in range(0, dry.shape[1], chunk_frames):
            piece, state = model(
                torch.from_numpy(dry[:, start : start + chunk_frames]),
                control_tensor,
                state,
            )
            pieces.append(piece.numpy())
    return np.concatenate(pieces, axis=1)


def _onnx_render(
    session: onnxruntime.InferenceSession,
    dry: np.ndarray,
    controls: np.ndarray,
    hidden_size: int,
    chunk_frames: int,
) -> np.ndarray:
    hidden = np.zeros((1, dry.shape[0], hidden_size), dtype=np.float32)
    cell = np.zeros_like(hidden)
    pieces = []
    for start in range(0, dry.shape[1], chunk_frames):
        rendered, hidden, cell = session.run(
            ("rendered", "next_hidden", "next_cell"),
            {
                "dry": dry[:, start : start + chunk_frames],
                "controls": controls,
                "hidden": hidden,
                "cell": cell,
            },
        )
        pieces.append(rendered)
    return np.concatenate(pieces, axis=1)


def _parity(
    model: DriveForwardRenderer,
    session: onnxruntime.InferenceSession,
) -> dict:
    rng = np.random.default_rng(20260830)
    dry = rng.normal(0.0, 0.12, size=(3, 4_103)).astype(np.float32)
    controls = np.asarray(
        ((0.0, 0.0, 0.0), (0.37, 0.64, 0.41), (1.0, 1.0, 1.0)),
        dtype=np.float32,
    )
    torch_full = _torch_render(model, dry, controls, dry.shape[1])
    torch_chunked = _torch_render(model, dry, controls, 509)
    onnx_full = _onnx_render(session, dry, controls, model.hidden_size, dry.shape[1])
    onnx_chunked = _onnx_render(session, dry, controls, model.hidden_size, 509)
    silence = np.zeros((2, 1_031), dtype=np.float32)
    silence_controls = np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), dtype=np.float32)
    onnx_silence = _onnx_render(
        session,
        silence,
        silence_controls,
        model.hidden_size,
        257,
    )

    def comparison(reference: np.ndarray, candidate: np.ndarray) -> dict:
        difference = candidate.astype(np.float64) - reference.astype(np.float64)
        return {
            "max_absolute_error": float(np.max(np.abs(difference))),
            "rms_error": float(np.sqrt(np.mean(np.square(difference)))),
        }

    metrics = {
        "onnx_vs_torch_full": comparison(torch_full, onnx_full),
        "onnx_vs_torch_chunked": comparison(torch_chunked, onnx_chunked),
        "torch_chunk_vs_full": comparison(torch_full, torch_chunked),
        "onnx_chunk_vs_full": comparison(onnx_full, onnx_chunked),
        "silence_max_absolute_output": float(np.max(np.abs(onnx_silence))),
        "finite": bool(
            np.isfinite(torch_full).all()
            and np.isfinite(onnx_full).all()
            and np.isfinite(onnx_chunked).all()
        ),
    }
    metrics["passed"] = bool(
        metrics["finite"]
        and metrics["onnx_vs_torch_full"]["max_absolute_error"] <= 2.0e-5
        and metrics["onnx_vs_torch_full"]["rms_error"] <= 2.0e-6
        and metrics["onnx_vs_torch_chunked"]["max_absolute_error"] <= 2.0e-5
        and metrics["onnx_vs_torch_chunked"]["rms_error"] <= 2.0e-6
        and metrics["torch_chunk_vs_full"]["max_absolute_error"] <= 2.0e-6
        and metrics["onnx_chunk_vs_full"]["max_absolute_error"] <= 2.0e-5
        and metrics["silence_max_absolute_output"] == 0.0
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=RUN / "drive-forward-candidate.pt")
    parser.add_argument("--output", type=Path, default=RUN / "drive-forward-candidate.onnx")
    parser.add_argument("--contract", type=Path, default=RUN / "runtime-contract.json")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model = _load(args.checkpoint)
    _export(model, args.output, args.opset)
    session = _session(args.output)
    parity = _parity(model, session)
    contract = {
        "schema": 1,
        "status": "passed" if parity["passed"] else "failed",
        "model_family": "drive-forward-causal-lstm",
        "research_scope": "synthetic DSP plus evaluation-only Pedalboard; not real hardware fidelity",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "onnx": str(args.output),
        "onnx_sha256": _sha256(args.output),
        "onnx_opset": args.opset,
        "sample_rate": FORWARD_RATE,
        "sample_format": "float32",
        "input_layout": {
            "dry": "[batch,frames]",
            "controls": "[batch,3] normalized",
            "hidden": f"[1,batch,{model.hidden_size}]",
            "cell": f"[1,batch,{model.hidden_size}]",
        },
        "output_layout": {
            "rendered": "[batch,frames]",
            "next_hidden": f"[1,batch,{model.hidden_size}]",
            "next_cell": f"[1,batch,{model.hidden_size}]",
        },
        "controls": [
            {"name": name, "normalized_range": [0.0, 1.0], "physical_range": list(bounds)}
            for name, bounds in zip(CONTROL_NAMES, CONTROL_RANGES, strict=True)
        ],
        "state_policy": "zero once at stream start; carry next_hidden/next_cell across every chunk",
        "audio_quality_policy": {
            "source_audio_immutable": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "automatic_dither": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
            "silence_must_remain_exact_zero": True,
        },
        "parity": parity,
    }
    args.contract.parent.mkdir(parents=True, exist_ok=True)
    args.contract.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(json.dumps(contract, sort_keys=True))
    if not parity["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
