"""Synthetic full-clock float32 precheck for the separate cascaded architecture.

No source audio, pretrained weights, evaluation set or audio-device access.
A pass permits an experiment, never fidelity/model-card admission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import onnx
import torch

from .audit_multirate_fuzz import CheckedMultirate
from .cascaded_multirate_fuzz import ARCHITECTURE, GEOMETRY, CascadedMultirateFuzz
from .precheck_multirate_fuzz import OrtMultirate, export, sha256, streamed


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new cascaded synthetic precheck directory required")
    torch.set_num_threads(1)
    torch.manual_seed(1011)
    model = CascadedMultirateFuzz(**GEOMETRY).eval()
    model.audio.output.weight.normal_(0, .05)
    names = (Path(__file__).name, "cascaded_multirate_fuzz.py", "multirate_fuzz.py",
             "audit_multirate_fuzz.py", "precheck_multirate_fuzz.py")
    sources = {name: sha256(Path(__file__).parent/name) for name in names}
    args.output.mkdir(parents=True)
    graph_path = args.output/"model.onnx"
    export(model, graph_path)
    graph = onnx.load(graph_path)
    for entry in graph.metadata_props:
        if entry.key == "architecture":
            entry.value = ARCHITECTURE
    onnx.checker.check_model(graph, full_check=True)
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.graph.initializer):
        raise ValueError("external weight sidecar forbidden")
    onnx.save(graph, graph_path)
    graph_hash = sha256(graph_path)
    cpu, ort = CheckedMultirate(model, model), CheckedMultirate(OrtMultirate(graph_path, model), model)
    cases = []
    for batch, frames, dynamic in ((2, 6173, True), (3, 1033, True), (1, 144000, False), (1, 144000, True), (1, 1440000, True)):
        dry = torch.randn(batch, frames)*.03
        controls = torch.rand(batch, frames, 2) if dynamic else torch.full((batch, frames, 2), .5)
        before = dry.clone(), controls.clone()
        a, _ = cpu(dry, controls)
        b, state = ort(dry, controls)
        sizes = [1, 17, 63, 64, 257, 1024] if frames < 10000 else [1024]
        c, stream_state = streamed(ort, dry, controls, sizes)
        d, _ = streamed(cpu, dry, controls, sizes)
        cases.append({"batch": batch, "frames": frames, "dynamic": dynamic, "parity_max": float((a-b).abs().max()),
                      "onnx_stream_max": float((b-c).abs().max()), "cpu_stream_max": float((a-d).abs().max()),
                      "phase_exact": bool(torch.equal(state[-1], stream_state[-1])),
                      "inputs_unchanged": bool(torch.equal(dry, before[0]) and torch.equal(controls, before[1]))})
        print(json.dumps(cases[-1]), flush=True)
        del a, b, c, d, before
    signal, controls = torch.randn(3, 6173)*3e-7, torch.rand(3, 6173, 2)
    safety, timing = {}, {}
    for name, backend in (("cpu", cpu), ("onnx", ort)):
        zero = max(float(backend(torch.zeros_like(signal), knobs)[0].abs().max())
                   for knobs in (controls, torch.tensor([[0., 0.], [.5, .5], [1., 1.]])))
        quiet = float(backend(signal, controls)[0].abs().max())
        probe = torch.cat((torch.randn(1, 1033)*.03, torch.zeros(1, 8199)), 1)
        knobs = torch.rand(1, probe.shape[1], 2)
        previous = backend(probe, knobs)[0]
        tail = float(previous[:, 1033+model.audio.receptive_field-1:].abs().max())
        probe[:, 17:] += .4
        knobs[:, 17:] = 1-knobs[:, 17:]
        future = float((previous[:, :17]-backend(probe, knobs)[0][:, :17]).abs().max())
        safety[name] = {"zero_peak": zero, "quiet_peak": quiet, "expired_tail_peak": tail, "future_prefix_error": future}
        dry, knobs = torch.randn(1, 144000)*.03, torch.rand(1, 144000, 2)
        timing[name] = {"full": [], "stream1024": []}
        for _ in range(3):
            tick = time.perf_counter()
            backend(dry, knobs)
            timing[name]["full"].append((time.perf_counter()-tick)/3)
            tick = time.perf_counter()
            streamed(backend, dry, knobs, [1024])
            timing[name]["stream1024"].append((time.perf_counter()-tick)/3)
    passed = (all(max(row["parity_max"], row["onnx_stream_max"], row["cpu_stream_max"]) <= 2e-6
                  and row["phase_exact"] and row["inputs_unchanged"] for row in cases)
              and all(row["zero_peak"] == row["expired_tail_peak"] == row["future_prefix_error"] == 0.
                      and row["quiet_peak"] <= .001 for row in safety.values())
              and all(max(values) < 1 for times in timing.values() for values in times.values()))
    if any(sha256(Path(__file__).parent/name) != digest for name, digest in sources.items()) or sha256(graph_path) != graph_hash:
        raise ValueError("synthetic precheck implementation/artifact changed")
    result = {"schema": 1, "purpose": "synthetic-cascaded-core-precheck", "passed": passed,
              "architecture": ARCHITECTURE, "geometry": GEOMETRY, "cases": cases, "safety": safety,
              "single_thread_rtf_including_validation": timing, "parameters": sum(value.numel() for value in model.parameters()),
              "state_tensor_bytes": sum(value.numel()*value.element_size() for value in model.initial_state(dry)),
              "source_sha256": sources, "graph_sha256": graph_hash, "graph_bytes": graph_path.stat().st_size,
              "trained": False, "admitted": False, "physical_audio_devices_used": False, "source_audio_loaded": False}
    (args.output/"metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("passed", "parameters", "state_tensor_bytes", "graph_bytes")}), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
