#!/usr/bin/env python3
"""Export unchanged float32 stable weights with explicit state and audit CPU parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import onnx
import onnxruntime
import torch

from .onnx_stable_effect import load_onnx_stable_effect
from .stable_effect import load_stable_effect


class StatefulExport(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, dry, controls, *flat_state):
        state = tuple((flat_state[i], flat_state[i+1]) for i in range(0, len(flat_state), 2))
        audio, next_state = self.model(dry, controls, state)
        return (audio, *(value for pair in next_state for value in pair))


def export_graph(checkpoint: Path, output: Path) -> None:
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model, _ = load_stable_effect(checkpoint)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_digest:
        raise ValueError('checkpoint changed while loading the export reference')
    widths = getattr(model, 'export_state_widths', [layer.hidden_size for layer in model.modules() if isinstance(layer, torch.nn.LSTM)])
    state = tuple(torch.zeros(1, 2, width) for width in widths for _ in range(2))
    input_names = ["dry", "controls", *(name for i in range(len(widths)) for name in (f"h{i}", f"c{i}"))]
    output_names = ["rendered", *(name for i in range(len(widths)) for name in (f"nh{i}", f"nc{i}"))]
    axes = {name: {1: "batch"} for name in input_names[2:] + output_names[1:]}
    axes.update({name: {0: "batch", 1: "frames"} for name in ("dry", "controls", "rendered")})
    torch.onnx.export(StatefulExport(model).eval(), (torch.zeros(2, 257), torch.full((2, 257, model.control_count), 0.5), *state),
                      output, dynamo=False, opset_version=17, input_names=input_names, output_names=output_names,
                      dynamic_axes=axes, do_constant_folding=True)
    graph = onnx.load(output)
    for key, value in {"checkpoint_sha256": checkpoint_digest,
                       "control_count": str(model.control_count), "state_widths": ",".join(map(str, widths)),
                       "sample_rate": "48000", "precision": "float32", "physical_audio_devices_used": "false"}.items():
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, output)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_digest:
        raise ValueError('checkpoint changed during ONNX export; graph is not admitted')


@torch.inference_mode()
def validate(checkpoint: Path, output: Path) -> dict:
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    graph_digest = hashlib.sha256(output.read_bytes()).hexdigest()
    reference, _ = load_stable_effect(checkpoint)
    model, _ = load_onnx_stable_effect(checkpoint, output)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_digest or hashlib.sha256(output.read_bytes()).hexdigest() != graph_digest:
        raise ValueError('checkpoint or graph changed while loading runtime audit')
    torch.manual_seed(976)
    signal = torch.randn(2, 6173) * 0.04
    dynamic = torch.rand(2, 6173, model.control_count)
    target, _ = reference(signal, dynamic)
    whole, _ = model(signal, dynamic)
    state, chunks, start = None, [], 0
    for stop in (17, 513, 2121, 6173):
        value, state = model(signal[:, start:stop], dynamic[:, start:stop], state)
        chunks.append(value)
        start = stop
    parity = float((whole-target).abs().max())
    stream = float((whole-torch.cat(chunks, 1)).abs().max())
    silence, _ = model(torch.zeros_like(signal), dynamic)
    quiet, _ = model(signal * 1e-5, dynamic)
    dry = torch.randn(1, 48000) * 0.04
    controls = torch.full((1, model.control_count), 0.5)
    long_reference, _ = reference(dry, controls)
    long_onnx, _ = model(dry, controls)
    parity = max(parity, float((long_reference-long_onnx).abs().max()))
    started = time.perf_counter()
    for _ in range(3):
        model(dry, controls)
    full_rtf = (time.perf_counter()-started)/3
    started = time.perf_counter()
    for _ in range(3):
        state = None
        for offset in range(0, 48000, 1024):
            _, state = model(dry[:, offset:offset+1024], controls, state)
    stream_rtf = (time.perf_counter()-started)/3
    safety = parity <= 2e-6 and stream <= 2e-6 and float(silence.abs().max()) == 0 and float(quiet.abs().max()) <= 1e-3
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_digest or hashlib.sha256(output.read_bytes()).hexdigest() != graph_digest:
        raise ValueError('checkpoint or graph changed during runtime audit')
    return {"schema": 1, "checkpoint_sha256": checkpoint_digest,
            "onnx_sha256": graph_digest, "onnx_path": str(output),
            "compute_device": "cpu", "backend": "onnxruntime", "onnxruntime_version": onnxruntime.__version__,
            "threads": 1, "float_precision": "float32", "quantization": False,
            "full_realtime_factor": full_rtf, "streaming_realtime_factor": stream_rtf,
            "streaming_block_frames": 1024, "cpu_realtime_passed": full_rtf < 1 and stream_rtf < 1,
            "torch_onnx_max_absolute_error": parity, "dynamic_control_signal_stream_max_error": stream,
            "dynamic_control_silence_peak": float(silence.abs().max()), "quiet_input_peak": float(quiet.abs().max()),
            "additional_safety_passed": bool(safety), "physical_audio_devices_used": False,
            "ui_integration_allowed": False, "automatic_normalization": False, "automatic_limiting": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise ValueError("ONNX export output or report already exists")
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_graph(args.checkpoint, args.output)
    report = validate(args.checkpoint, args.output)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["additional_safety_passed"] or not report["cpu_realtime_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
