"""Trained centered-core runtime audit on train/calibration54 only.

The synthetic preflight is not reused as trained evidence. Every audio, state,
parity and timing check is rerun with the selected materialized weights. No
official eval, model-card activation, physical device or source write occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import onnx
import torch

from .asrnn_effects import read_effect_pair
from .audit_multirate_fuzz import CheckedMultirate
from .evaluate_multirate_fuzz import CENTERED_ARCHITECTURE, graph_runtime, load_candidate
from .precheck_multirate_fuzz import export, sha256, streamed
from .train_asrnn_phase7 import _evaluate
from .widen_multirate_fuzz import WIDE_GEOMETRY


@torch.inference_mode()
def audit(checkpoint, output):
    if output.exists():
        raise ValueError("new trained audit directory required")
    model, payload, parent = load_candidate(checkpoint)
    if payload["architecture"] != CENTERED_ARCHITECTURE or payload["geometry"] != WIDE_GEOMETRY:
        raise ValueError("materialized centered wide model required")
    checkpoint_hash, parent_hash = sha256(checkpoint), sha256(checkpoint.with_name("metrics.json"))
    names = (Path(__file__).name, "evaluate_multirate_fuzz.py", "centered_multirate_fuzz.py",
             "audit_multirate_fuzz.py", "precheck_multirate_fuzz.py", "train_asrnn_phase7.py", "asrnn_effects.py")
    sources = {**parent["source_sha256"], **{name: sha256(Path(__file__).parent / name) for name in names}}
    provenance = parent["audio_provenance"]
    if set(provenance) != {"fit", "calibration"} or len(provenance["fit"]) != 234 or len(provenance["calibration"]) != 54:
        raise ValueError("original234/54 training partition required")

    def unchanged():
        if sha256(checkpoint) != checkpoint_hash or sha256(checkpoint.with_name("metrics.json")) != parent_hash:
            raise ValueError("source checkpoint/report changed")
        if any(sha256(Path(__file__).parent / name) != digest for name, digest in sources.items()):
            raise ValueError("runtime audit source changed")
        if any(sha256(row["path"]) != row["sha256"] for rows in provenance.values() for row in rows):
            raise ValueError("source audio changed")

    unchanged()
    torch.manual_seed(1003)
    started, rows = time.perf_counter(), []
    for record in provenance["calibration"]:
        path = Path(record["path"])
        if "train" not in path.parts or int(path.stem.split(",")[-1]) % 5 != 0:
            raise ValueError("only train/calibration audio may be decoded")
        dry, wet, controls = read_effect_pair(path, "dfz")
        rows.append({"dry": torch.from_numpy(dry), "wet": torch.from_numpy(wet), "controls": torch.from_numpy(controls),
                     "attack": int(path.stem.split(",")[0]), "name": path.name})
    output.mkdir(parents=True)
    graph_path = output / "model.onnx"
    export(model, graph_path)
    graph = onnx.load(graph_path)
    metadata = {entry.key: entry.value for entry in graph.metadata_props}
    metadata.update(architecture=CENTERED_ARCHITECTURE, checkpoint_sha256=checkpoint_hash, trained="true", admitted="false")
    del graph.metadata_props[:]
    for key, value in metadata.items():
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(graph, graph_path)
    graph_hash = sha256(graph_path)
    cpu, ort = CheckedMultirate(model, model), graph_runtime(graph_path, model, checkpoint_hash)
    cases, probes = [], []
    for index, row in enumerate(rows):
        dry = row["dry"][None]
        controls = row["controls"][None, None].expand(-1, dry.shape[1], -1)
        before = dry.clone(), controls.clone()
        a, b = cpu(dry, controls)[0], ort(dry, controls)[0]
        c = streamed(ort, dry, controls, [1024])[0]
        if not torch.equal(dry, before[0]) or not torch.equal(controls, before[1]):
            raise ValueError("in-memory source changed")
        cases.append({"name": row["name"], "parity_max": float((a - b).abs().max()), "onnx_stream_max": float((b - c).abs().max())})
        if (index + 1) % 9 == 0:
            print(json.dumps({"files": index + 1, "parity_max": max(value["parity_max"] for value in cases)}), flush=True)
    for batch, frames in ((2, 6173), (1, 144000), (1, 1440000)):
        dry, controls = torch.randn(batch, frames) * .03, torch.rand(batch, frames, 2)
        before = dry.clone(), controls.clone()
        a, _ = cpu(dry, controls)
        b, state = ort(dry, controls)
        c, chunk_state = streamed(ort, dry, controls, [1, 17, 63, 64, 257, 1024] if frames < 10000 else [1024])
        d, _ = streamed(cpu, dry, controls, [1024])
        probes.append({"batch": batch, "frames": frames, "parity_max": float((a - b).abs().max()),
                       "onnx_stream_max": float((b - c).abs().max()), "cpu_stream_max": float((a - d).abs().max()),
                       "phase_exact": bool(torch.equal(state[-1], chunk_state[-1])),
                       "inputs_unchanged": bool(torch.equal(dry, before[0]) and torch.equal(controls, before[1]))})
        print(json.dumps({"dynamic_probe": probes[-1]}), flush=True)
        del a, b, c, d, before
    signal, knobs = torch.randn(3, 6173) * 3e-7, torch.rand(3, 6173, 2)
    safety, timing = {}, {}
    for name, backend in (("cpu", cpu), ("onnx", ort)):
        zero = max(float(backend(torch.zeros_like(signal), controls)[0].abs().max())
                   for controls in (knobs, torch.tensor([[0., 0.], [.5, .5], [1., 1.]])))
        quiet = float(backend(signal, knobs)[0].abs().max())
        excited = torch.cat((torch.randn(1, 1033) * .03, torch.zeros(1, 4099)), 1)
        controls = torch.rand(1, excited.shape[1], 2)
        value = backend(excited, controls)[0]
        tail = float(value[:, 1033 + model.audio.receptive_field - 1:].abs().max())
        excited[:, 17:] += .4
        controls[:, 17:] = 1 - controls[:, 17:]
        future = float((value[:, :17] - backend(excited, controls)[0][:, :17]).abs().max())
        safety[name] = {"zero_peak": zero, "quiet_peak": quiet, "expired_tail_peak": tail, "future_prefix_error": future}
        dry, controls = rows[0]["dry"][None], torch.rand(1, 144000, 2)
        timing[name] = {"full": [], "stream1024": []}
        for _ in range(3):
            tick = time.perf_counter()
            backend(dry, controls)
            timing[name]["full"].append((time.perf_counter() - tick) / 3)
            tick = time.perf_counter()
            streamed(backend, dry, controls, [1024])
            timing[name]["stream1024"].append((time.perf_counter() - tick) / 3)
    metrics = {name: _evaluate(backend, rows, torch.device("cpu"), 9) for name, backend in (("cpu", cpu), ("onnx", ort))}
    passed = (all(max(row["parity_max"], row["onnx_stream_max"]) <= 2e-6 for row in cases)
              and all(max(row["parity_max"], row["onnx_stream_max"], row["cpu_stream_max"]) <= 2e-6
                      and row["phase_exact"] and row["inputs_unchanged"] for row in probes)
              and all(row["zero_peak"] == row["expired_tail_peak"] == row["future_prefix_error"] == 0.
                      and row["quiet_peak"] <= .001 for row in safety.values())
              and all(max(values) < 1 for times in timing.values() for values in times.values()))
    unchanged()
    if sha256(graph_path) != graph_hash:
        raise ValueError("audited graph changed")
    result = {"schema": 1, "purpose": "trained-centered-core-audit", "architecture": CENTERED_ARCHITECTURE,
              "runtime_passed": passed, "calibration_passed": all(value["passes_selection_gate"] for value in metrics.values()),
              "calibration": metrics, "real_audio_cases": cases, "dynamic_probes": probes, "safety": safety,
              "single_thread_rtf_including_validation": timing, "parameters": sum(value.numel() for value in model.parameters()),
              "checkpoint_sha256": checkpoint_hash, "parent_checkpoint_sha256": checkpoint_hash, "parent_report_sha256": parent_hash,
              "graph_sha256": graph_hash, "graph_bytes": graph_path.stat().st_size, "geometry": WIDE_GEOMETRY,
              "source_sha256": sources, "audio_provenance": provenance, "source_and_audio_reverified": True,
              "official_eval_opened": False, "physical_audio_devices_used": False, "source_audio_modified": False,
              "automatic_gain_or_normalization": False, "admitted": False, "elapsed_seconds": time.perf_counter() - started}
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("runtime_passed", "calibration_passed", "elapsed_seconds")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
