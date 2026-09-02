"""Audit immutable trained multirate weights, not random-weight architecture.

Experimental CPU/ONNX diagnostics only. Does not register or promote a model,
open official eval/locked-final audio, alter gains, or use physical devices.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnx
import torch

from .asrnn_effects import read_effect_pair
from .multirate_fuzz import BLOCK, MultirateFuzz
from .precheck_multirate_fuzz import OrtMultirate, export, sha256, streamed
from .train_asrnn_phase7 import _evaluate


GEOMETRY = {"audio_width": 16, "audio_blocks": 10, "controller_width": 8, "controller_blocks": 10}


def validate_call(reference, dry, controls, state):
    if not isinstance(dry, torch.Tensor) or dry.device.type != "cpu" or dry.dtype != torch.float32:
        raise ValueError("CPU float32 Dry required")
    if dry.ndim != 2 or min(dry.shape) < 1 or not torch.isfinite(dry).all():
        raise ValueError("finite nonempty batch/time audio required")
    if not isinstance(controls, torch.Tensor) or controls.device != dry.device or controls.dtype != dry.dtype:
        raise ValueError("CPU float32 controls required")
    if controls.shape not in ((dry.shape[0], 2), (dry.shape[0], dry.shape[1], 2)):
        raise ValueError("two controls with matching batch/time required")
    if not torch.isfinite(controls).all() or (controls < 0).any() or (controls > 1).any():
        raise ValueError("finite normalized controls required")
    if state is not None:
        expected = reference.initial_state(dry)
        if not isinstance(state, (tuple, list)) or len(state) != len(expected):
            raise ValueError("complete multirate state required")
        for index, (value, template) in enumerate(zip(state, expected)):
            if (not isinstance(value, torch.Tensor) or value.shape != template.shape
                    or value.dtype != template.dtype or value.device != template.device):
                raise ValueError(f"invalid state tensor{index}")
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite state")
        if not 0 <= int(state[-1]) < BLOCK:
            raise ValueError("clock phase outside0..63")
    return controls[:, None].expand(-1, dry.shape[1], -1) if controls.ndim == 2 else controls


class CheckedMultirate:
    """Fail-closed diagnostic boundary shared by CPU and exported backends."""

    def __init__(self, backend, reference):
        self.backend, self.reference = backend, reference

    def eval(self):
        self.reference.eval()
        return self

    def __call__(self, dry, controls, state=None):
        controls = validate_call(self.reference, dry, controls, state)
        rendered, next_state = self.backend(dry, controls, state)
        if rendered.shape != dry.shape or rendered.dtype != dry.dtype or rendered.device != dry.device:
            raise ValueError("invalid output audio tensor")
        if not torch.isfinite(rendered).all():
            raise ValueError("nonfinite output audio")
        validate_call(self.reference, dry, controls, next_state)
        previous_phase = 0 if state is None else int(state[-1])
        if int(next_state[-1]) != (previous_phase + dry.shape[1]) % BLOCK:
            raise ValueError("backend returned wrong sample clock")
        return rendered, next_state


def load_trained(checkpoint, report):
    evidence = json.loads(report.read_text())
    if evidence.get("checkpoint_sha256") != sha256(checkpoint) or not evidence.get("source_and_audio_reverified"):
        raise ValueError("complete immutable training evidence required")
    for name, digest in evidence["source_sha256"].items():
        if sha256(Path(__file__).parent / name) != digest:
            raise ValueError("trained source changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (payload.get("experimental_schema") != 1 or payload.get("architecture") != "causal-multirate-gcn"
            or payload.get("device") != "dfz" or payload.get("sample_rate") != 48_000
            or payload.get("geometry") != GEOMETRY):
        raise ValueError("unsupported experimental multirate checkpoint")
    model = MultirateFuzz(**GEOMETRY).eval()
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in payload["state_dict"].values()):
        raise ValueError("finite float32 model weights required")
    model.load_state_dict(payload["state_dict"], strict=True)
    if not model.audio.output.weight.count_nonzero():
        raise ValueError("zero-output architecture is not a trained fidelity candidate")
    return model, evidence


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh audit output directory required")
    report_path = args.checkpoint.with_name("metrics.json")
    torch.set_num_threads(1)
    torch.manual_seed(991)
    started = time.perf_counter()
    model, evidence = load_trained(args.checkpoint, report_path)
    names = (Path(__file__).name, "multirate_fuzz.py", "precheck_multirate_fuzz.py", "train_asrnn_phase7.py",
             "asrnn_effects.py", "asrnn_data.py")
    sources = {name: sha256(Path(__file__).parent / name) for name in names}
    provenance = evidence["audio_provenance"]
    if set(provenance) != {"fit", "calibration"} or len(provenance["fit"]) != 234 or len(provenance["calibration"]) != 54:
        raise ValueError("frozen234/54 development partition required")
    checkpoint_hash, report_hash = sha256(args.checkpoint), sha256(report_path)

    def unchanged():
        if sha256(args.checkpoint) != checkpoint_hash or sha256(report_path) != report_hash:
            raise ValueError("checkpoint/training evidence changed during audit")
        if any(sha256(Path(__file__).parent / name) != digest for name, digest in sources.items()):
            raise ValueError("audit source changed")
        if any(sha256(row["path"]) != row["sha256"] for rows in provenance.values() for row in rows):
            raise ValueError("source audio changed")

    unchanged()
    calibration = []
    for record in provenance["calibration"]:
        path = Path(record["path"])
        if int(path.stem.split(",")[-1]) % 5 != 0 or "train" not in path.parts:
            raise ValueError("only original train/calibration takes may be decoded")
        dry, wet, controls = read_effect_pair(path, "dfz")
        calibration.append({"dry": torch.from_numpy(dry), "wet": torch.from_numpy(wet),
                            "controls": torch.from_numpy(controls), "attack": int(path.stem.split(",")[0]), "name": path.name})
    args.output.mkdir(parents=True)
    graph_path = args.output / "model.onnx"
    export(model, graph_path)
    graph = onnx.load(graph_path)
    for entry in graph.metadata_props:
        if entry.key == "trained":
            entry.value = "true"
    for key, value in {"checkpoint_sha256": checkpoint_hash, "admitted": "false", "usage": "offline-development-audit"}.items():
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(graph, full_check=True)
    if any(value.data_location == onnx.TensorProto.EXTERNAL for value in graph.graph.initializer):
        raise ValueError("external model tensor sidecars forbidden")
    onnx.save(graph, graph_path)
    graph_hash = sha256(graph_path)
    cpu, runtime = CheckedMultirate(model, model), CheckedMultirate(OrtMultirate(graph_path, model), model)
    real_cases, checks = [], []
    for index, row in enumerate(calibration):
        dry = row["dry"][None]
        controls = row["controls"][None]
        reference, _ = cpu(dry, controls)
        full, _ = runtime(dry, controls)
        dynamic = controls[:, None].expand(-1, dry.shape[1], -1)
        chunked, _ = streamed(runtime, dry, dynamic, [1024])
        real_cases.append({"name": row["name"], "parity_max": float((reference - full).abs().max()),
                           "onnx_stream_max": float((full - chunked).abs().max())})
        if (index + 1) % 9 == 0:
            print(json.dumps({"stage": "trained-real-calibration-parity", "files": index + 1,
                              "max_error": max(item["parity_max"] for item in real_cases)}), flush=True)
    for batch, frames in ((2, 6173), (1, 144000), (1, 1440000)):
        dry, controls = torch.randn(batch, frames) * .03, torch.rand(batch, frames, 2)
        dry_copy, controls_copy = dry.clone(), controls.clone()
        ref, ref_state = cpu(dry, controls)
        full, full_state = runtime(dry, controls)
        sizes = [1, 17, 63, 64, 257, 1024] if frames < 10000 else [1024]
        chunked, state = streamed(runtime, dry, controls, sizes)
        cpu_chunked, _ = streamed(cpu, dry, controls, sizes)
        checks.append({"batch": batch, "frames": frames,
                       "parity_max": float((ref - full).abs().max()),
                       "onnx_stream_max": float((full - chunked).abs().max()),
                       "cpu_stream_max": float((ref - cpu_chunked).abs().max()),
                       "state_parity_max": max(float((a - b).abs().max()) for a, b in zip(ref_state[:-1], full_state[:-1])),
                       "phase_exact": bool(torch.equal(ref_state[-1], full_state[-1]) and torch.equal(full_state[-1], state[-1])),
                       "inputs_unchanged": bool(torch.equal(dry, dry_copy) and torch.equal(controls, controls_copy))})
        print(json.dumps({"dynamic_probe": checks[-1]}), flush=True)
        del ref, full, chunked, cpu_chunked, dry_copy, controls_copy
    zeros, quiet, future = {}, {}, {}
    controls = torch.rand(3, 6173, 2)
    for name, backend in (("cpu", cpu), ("onnx", runtime)):
        zeros[name] = max(float(backend(torch.zeros(3, 6173), value)[0].abs().max())
                          for value in (controls, torch.tensor([[0., 0.], [.5, .5], [1., 1.]])))
        quiet[name] = float(backend(torch.randn(3, 6173) * 3e-7, controls)[0].abs().max())
        dry, knobs = torch.randn(1, 6173) * .03, controls[:1].clone()
        before = backend(dry, knobs)[0]
        dry[:, 17:] += .4
        knobs[:, 17:] = 1 - knobs[:, 17:]
        future[name] = float((before[:, :17] - backend(dry, knobs)[0][:, :17]).abs().max())
    timing = {name: {"full": [], "stream1024": []} for name in ("cpu", "onnx")}
    dry, controls = calibration[0]["dry"][None], torch.rand(1, 144000, 2)
    for name, backend in (("cpu", cpu), ("onnx", runtime)):
        for _ in range(3):
            tick = time.perf_counter()
            backend(dry, controls)
            timing[name]["full"].append((time.perf_counter() - tick) / 3)
            tick = time.perf_counter()
            streamed(backend, dry, controls, [1024])
            timing[name]["stream1024"].append((time.perf_counter() - tick) / 3)
    measured = {name: _evaluate(backend, calibration, torch.device("cpu"), 9) for name, backend in (("cpu", cpu), ("onnx", runtime))}
    runtime_passed = (all(row["parity_max"] <= 2e-6 and row["onnx_stream_max"] <= 2e-6 for row in real_cases)
                      and all(max(row["parity_max"], row["onnx_stream_max"], row["cpu_stream_max"]) <= 2e-6
                              and row["phase_exact"] and row["inputs_unchanged"] for row in checks)
                      and max(zeros.values()) == 0 and max(quiet.values()) <= 1e-3 and max(future.values()) == 0
                      and all(max(values) < 1 for backend in timing.values() for values in backend.values()))
    unchanged()
    if sha256(graph_path) != graph_hash:
        raise ValueError("exported graph changed")
    result = {"schema": 1, "runtime_passed": runtime_passed,
              "calibration_passed": all(row["passes_selection_gate"] for row in measured.values()),
              "calibration": measured, "real_audio_cases": real_cases, "dynamic_probes": checks,
              "zero_peak": zeros, "quiet_peak": quiet, "future_prefix_error": future,
              "single_thread_rtf_including_validation": timing, "checkpoint_sha256": checkpoint_hash,
              "training_report_sha256": report_hash, "graph_sha256": graph_hash, "graph_bytes": graph_path.stat().st_size,
              "source_sha256": sources, "source_and_audio_reverified": True,
              "trained": True, "admitted": False, "official_eval_opened": False, "physical_audio_devices_used": False,
              "gain_or_normalization_applied": False, "source_audio_modified": False,
              "scope": "experimental calibrated54 diagnostic; full development/quality admission still required",
              "elapsed_seconds": time.perf_counter() - started}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("runtime_passed", "calibration_passed", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
