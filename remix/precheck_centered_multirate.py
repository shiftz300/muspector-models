"""Synthetic nonzero-threshold preflight; no data or trained-model admission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import onnx
import torch

from .audit_multirate_fuzz import CheckedMultirate
from .centered_multirate_fuzz import CenteredMultirateFuzz
from .precheck_multirate_fuzz import OrtMultirate, export, sha256, streamed
from .widen_multirate_fuzz import WIDE_GEOMETRY


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new synthetic preflight directory required")
    torch.set_num_threads(1)
    torch.manual_seed(1001)
    model = CenteredMultirateFuzz(**WIDE_GEOMETRY).eval()
    model.audio.output.weight.normal_(0, .05)
    for layer in (*model.audio.activation_current, *model.audio.activation_slow):
        layer.weight.normal_(0, .2)
        if layer.bias is not None:
            layer.bias.normal_(0, .1)
    names = (Path(__file__).name, "centered_multirate_fuzz.py", "multirate_fuzz.py", "audit_multirate_fuzz.py", "precheck_multirate_fuzz.py")
    sources = {name: sha256(Path(__file__).parent / name) for name in names}
    args.output.mkdir(parents=True)
    graph_path = args.output / "model.onnx"
    export(model, graph_path)
    graph = onnx.load(graph_path)
    for entry in graph.metadata_props:
        if entry.key == "architecture":
            entry.value = "experimental-zero-origin-centered-causal-multirate"
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, graph_path)
    graph_hash = sha256(graph_path)
    cpu, ort = CheckedMultirate(model, model), CheckedMultirate(OrtMultirate(graph_path, model), model)
    started, cases = time.perf_counter(), []
    for batch, frames, dynamic in ((1, 144000, False), (1, 144000, True), (2, 6173, True), (1, 1440000, True)):
        dry = torch.randn(batch, frames) * .03
        controls = torch.rand(batch, frames, 2) if dynamic else torch.full((batch, frames, 2), .5)
        before = (dry.clone(), controls.clone())
        reference, ref_state = cpu(dry, controls)
        whole, whole_state = ort(dry, controls)
        sizes = [1, 17, 63, 64, 257, 1024] if frames < 10000 else [1024]
        chunked, chunk_state = streamed(ort, dry, controls, sizes)
        cpu_chunked, _ = streamed(cpu, dry, controls, sizes)
        cases.append({"batch": batch, "frames": frames, "dynamic": dynamic,
                      "parity_max": float((reference - whole).abs().max()), "onnx_stream_max": float((whole - chunked).abs().max()),
                      "cpu_stream_max": float((reference - cpu_chunked).abs().max()),
                      "phase_exact": bool(torch.equal(ref_state[-1], whole_state[-1]) and torch.equal(whole_state[-1], chunk_state[-1])),
                      "inputs_unchanged": bool(torch.equal(before[0], dry) and torch.equal(before[1], controls))})
        print(json.dumps(cases[-1]), flush=True)
        del reference, whole, chunked, cpu_chunked, before
    safety, timings = {}, {}
    for name, backend in (("cpu", cpu), ("onnx", ort)):
        zeros, controls = torch.zeros(3, 6173), torch.rand(3, 6173, 2)
        state = list(model.initial_state(zeros))
        for value in state[2 * model.fast_count:-3]:
            value.normal_(0, .1)
        state[-2].normal_(0, .3)
        zero = max(float(backend(zeros, controls)[0].abs().max()), float(backend(zeros, controls, tuple(state))[0].abs().max()),
                   float(backend(zeros, torch.tensor([[0., 0.], [.5, .5], [1., 1.]]))[0].abs().max()))
        quiet = float(backend(torch.randn_like(zeros) * 3e-7, controls)[0].abs().max())
        dry = torch.cat((torch.randn(1, 1033) * .03, torch.zeros(1, 4099)), 1)
        controls = torch.rand(1, dry.shape[1], 2)
        value = backend(dry, controls)[0]
        tail = float(value[:, 1033 + model.audio.receptive_field - 1:].abs().max())
        dry[:, 17:] += .4
        controls[:, 17:] = 1 - controls[:, 17:]
        future = float((value[:, :17] - backend(dry, controls)[0][:, :17]).abs().max())
        safety[name] = {"zero_peak": zero, "quiet_peak": quiet, "expired_tail_peak": tail, "future_prefix_error": future}
        dry, controls = torch.randn(1, 144000) * .03, torch.rand(1, 144000, 2)
        timings[name] = {"full": [], "stream1024": []}
        for _ in range(3):
            tick = time.perf_counter()
            backend(dry, controls)
            timings[name]["full"].append((time.perf_counter() - tick) / 3)
            tick = time.perf_counter()
            streamed(backend, dry, controls, [1024])
            timings[name]["stream1024"].append((time.perf_counter() - tick) / 3)
    passed = (all(max(row["parity_max"], row["onnx_stream_max"], row["cpu_stream_max"]) <= 2e-6
                  and row["phase_exact"] and row["inputs_unchanged"] for row in cases)
              and all(row["zero_peak"] == row["expired_tail_peak"] == row["future_prefix_error"] == 0.
                      and row["quiet_peak"] <= 1e-3 for row in safety.values())
              and all(max(values) < 1 for backend in timings.values() for values in backend.values()))
    if any(sha256(Path(__file__).parent / name) != digest for name, digest in sources.items()) or sha256(graph_path) != graph_hash:
        raise ValueError("synthetic preflight code/graph changed")
    report = {"schema": 1, "passed": passed, "status": "passed-synthetic-only" if passed else "failed-synthetic-preflight",
              "architecture": "zero-origin-centered-causal-multirate", "geometry": WIDE_GEOMETRY,
              "parameters": sum(value.numel() for value in model.parameters()), "nonzero_threshold_heads_tested": True,
              "cases": cases, "safety": safety, "single_thread_rtf_including_validation": timings,
              "source_sha256": sources, "graph_sha256": graph_hash, "graph_bytes": graph_path.stat().st_size,
              "source_audio_loaded": False, "physical_audio_devices_used": False, "trained": False, "admitted": False,
              "elapsed_seconds": time.perf_counter() - started}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "status", "parameters", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
