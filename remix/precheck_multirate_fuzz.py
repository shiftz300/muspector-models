"""Synthetic single-ONNX precheck for the standalone two-rate architecture.

No training data, original LSTM, hardware audio interface, or trained-model
admission is involved. The integer block clock is an explicit graph input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .multirate_fuzz import MultirateFuzz


class FlatExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, dry, controls, *state):
        audio, next_state = self.model(dry, controls, state)
        return audio, *next_state


class OrtMultirate:
    def __init__(self, path, reference):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.names = [item.name for item in self.session.get_inputs()]
        self.reference = reference

    def __call__(self, dry, controls, state=None):
        if state is None:
            state = self.reference.initial_state(dry)
        inputs = [dry, controls, *state]
        result = self.session.run(None, {name: value.detach().numpy() for name, value in zip(self.names, inputs)})
        return torch.from_numpy(result[0]), tuple(torch.from_numpy(value) for value in result[1:])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(model, output):
    state_names = [name for index in range(len(model.state_widths)) for name in (f"h{index}", f"c{index}")]
    state_names += ["pending", "held", "phase"]
    inputs, outputs = ["dry", "controls", *state_names], ["rendered", *["next_" + name for name in state_names]]
    axes = {name: {1: "batch"} for name in state_names[:-3] + outputs[1:-3]}
    axes.update({name: {0: "batch"} for name in ("pending", "held", "next_pending", "next_held")})
    axes.update({name: {0: "batch", 1: "frames"} for name in ("dry", "controls", "rendered")})
    dry, controls = torch.randn(2, 257) * .03, torch.rand(2, 257, 2)
    state = list(model.initial_state(dry))
    state[-1] = torch.tensor(17, dtype=torch.int64)
    state[-3][:, :17].normal_(0, .03)
    state[-2].normal_(0, .1)
    torch.onnx.export(FlatExport(model).eval(), (dry, controls, *state), str(output),
                      input_names=inputs, output_names=outputs, dynamic_axes=axes,
                      opset_version=17, dynamo=False, do_constant_folding=True)
    graph = onnx.load(output)
    onnx.checker.check_model(graph, full_check=True)
    for name, value in {"architecture": "experimental-causal-two-rate-gcn", "sample_rate": "48000",
                        "block_frames": "64", "precision": "float32", "clock_precision": "int64",
                        "trained": "false", "physical_audio_devices_used": "false"}.items():
        entry = graph.metadata_props.add()
        entry.key, entry.value = name, value
    onnx.save(graph, output)


@torch.inference_mode()
def streamed(model, dry, controls, sizes):
    state, chunks, first, index = None, [], 0, 0
    while first < dry.shape[1]:
        stop = min(dry.shape[1], first + sizes[index % len(sizes)])
        value, state = model(dry[:, first:stop], controls[:, first:stop], state)
        chunks.append(value)
        first, index = stop, index + 1
    return torch.cat(chunks, 1), state


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new synthetic precheck directory required")
    torch.set_num_threads(1)
    torch.manual_seed(989)
    model = MultirateFuzz().eval()
    with torch.no_grad():
        model.audio.output.weight.normal_(0, .05)
    sources = {name: sha256(Path(__file__).parent / name) for name in (Path(__file__).name, "multirate_fuzz.py")}
    args.output.mkdir(parents=True)
    graph_path = args.output / "model.onnx"
    export(model, graph_path)
    runtime = OrtMultirate(graph_path, model)
    measurements, passed = [], True
    cases = [(1, 144_000, False), (1, 144_000, True), (2, 6173, True), (3, 1033, True), (1, 1_440_000, True)]
    for batch, frames, dynamic in cases:
        dry = torch.randn(batch, frames) * .03
        controls = torch.rand(batch, frames, 2) if dynamic else torch.full((batch, frames, 2), .5)
        started = time.perf_counter()
        reference, ref_state = model(dry, controls)
        rendered, state = runtime(dry, controls)
        chunked, chunk_state = streamed(runtime, dry, controls, [1024] if frames > 10_000 else [1, 17, 63, 64, 257, 1024])
        torch_chunked, _ = streamed(model, dry, controls, [1024] if frames > 10_000 else [1, 17, 63, 64, 257, 1024])
        row = {"batch": batch, "frames": frames, "dynamic": dynamic,
               "torch_onnx_audio_max_error": float((reference - rendered).abs().max()),
               "onnx_stream_max_error": float((rendered - chunked).abs().max()),
               "torch_stream_max_error": float((reference - torch_chunked).abs().max()),
               "torch_onnx_float_state_max_error": max(float((a - b).abs().max()) for a, b in zip(ref_state[:-1], state[:-1])),
               "phase_exact": bool(torch.equal(ref_state[-1], state[-1]) and torch.equal(state[-1], chunk_state[-1])),
               "elapsed_seconds": time.perf_counter() - started}
        passed &= max(row["torch_onnx_audio_max_error"], row["onnx_stream_max_error"], row["torch_stream_max_error"]) <= 2e-6 and row["phase_exact"]
        measurements.append(row)
        print(json.dumps(row), flush=True)
        del reference, rendered, chunked, torch_chunked, ref_state, state, chunk_state
    dry, controls = torch.zeros(2, 6173), torch.rand(2, 6173, 2)
    zero = runtime(dry, controls)[0]
    quiet = runtime(torch.randn_like(dry) * 3e-7, controls)[0]
    probe = torch.randn(1, 6173) * .03
    controls = torch.rand(1, 6173, 2)
    before = runtime(probe, controls)[0]
    probe[:, 17:] += .4
    controls[:, 17:] = 1 - controls[:, 17:]
    future = float((before[:, :17] - runtime(probe, controls)[0][:, :17]).abs().max())
    signal, controls = torch.randn(1, 144_000) * .03, torch.rand(1, 144_000, 2)
    timing = {"full": [], "stream1024": []}
    for _ in range(3):
        started = time.perf_counter()
        runtime(signal, controls)
        timing["full"].append((time.perf_counter() - started) / 3)
        started = time.perf_counter()
        streamed(runtime, signal, controls, [1024])
        timing["stream1024"].append((time.perf_counter() - started) / 3)
    passed &= float(zero.abs().max()) == 0 and float(quiet.abs().max()) <= 1e-3 and future == 0
    passed &= all(max(values) < 1 for values in timing.values())
    report = {"schema": 1, "status": "passed-synthetic-only" if passed else "failed-synthetic-precheck",
              "passed": bool(passed), "cases": measurements, "zero_peak": float(zero.abs().max()),
              "quiet_peak": float(quiet.abs().max()), "future_prefix_error": future,
              "ort_single_thread_rtf": timing, "parameters": sum(value.numel() for value in model.parameters()),
              "state_tensor_bytes": sum(value.numel() * value.element_size() for value in model.initial_state(signal)),
              "source_sha256": sources, "graph_sha256": sha256(graph_path), "graph_bytes": graph_path.stat().st_size,
              "torch_version": str(torch.__version__), "onnxruntime_version": ort.__version__,
              "single_onnx_graph": True, "precision": "float32-with-explicit-int64-clock",
              "physical_audio_devices_used": False, "source_audio_loaded": False, "trained": False, "admitted": False}
    if any(sha256(Path(__file__).parent / name) != digest for name, digest in sources.items()):
        raise ValueError("precheck source changed")
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "parameters", "graph_bytes", "state_tensor_bytes", "ort_single_thread_rtf")}), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
