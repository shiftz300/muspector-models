"""Complete trained cascaded CPU/ONNX audit; calibration only, no admission.

Reads the explicit frozen training manifest, never enumerates official eval.
Runtime evidence and full original waveform-quality metrics are both required.
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
from .cascaded_candidate import load_candidate
from .cascaded_multirate_fuzz import ARCHITECTURE, GEOMETRY
from .cascaded_runtime import graph_runtime, structure_bound
from .evaluate_asrnn_effect import _quality_gate, manifest_sha256, source_record
from .evaluate_multirate_fuzz import quality_metrics
from .precheck_multirate_fuzz import export, sha256, streamed
from .promote_dfz_forward import CALIBRATION_TAKES, FIT_TAKES, finite_numbers, validate_records


@torch.inference_mode()
def audit(checkpoint, output, corpus):
    if output.exists():
        raise ValueError("new trained cascaded audit directory required")
    model, parent = load_candidate(checkpoint)
    bound = structure_bound(model)
    checkpoint_hash, parent_hash = sha256(checkpoint), sha256(checkpoint.with_name("metrics.json"))
    names = (Path(__file__).name, "cascaded_runtime.py", "evaluate_multirate_fuzz.py", "evaluate_asrnn_effect.py",
             "promote_dfz_forward.py", "forward_drive.py")
    sources = {**parent["source_sha256"], **{name: sha256(Path(__file__).parent/name) for name in names}}
    provenance = parent["audio_provenance"]
    if set(provenance) != {"fit", "calibration"} or len(provenance["fit"]) != 234 or len(provenance["calibration"]) != 54:
        raise ValueError("original234/54 manifest required")

    def unchanged():
        if sha256(checkpoint) != checkpoint_hash or sha256(checkpoint.with_name("metrics.json")) != parent_hash:
            raise ValueError("source checkpoint/report changed")
        if any(sha256(Path(__file__).parent/name) != digest for name, digest in sources.items()):
            raise ValueError("audit implementation changed")
        if any(sha256(row["path"]) != row["sha256"] for rows in provenance.values() for row in rows):
            raise ValueError("original audio changed")

    unchanged()
    started, rows, partitions = time.perf_counter(), [], {}
    for split in ("fit", "calibration"):
        records = []
        for record in provenance[split]:
            path = Path(record["path"])
            path.resolve().relative_to(corpus.resolve())
            if "train" not in path.parts or (int(path.stem.split(",")[-1])%5 == 0) != (split == "calibration"):
                raise ValueError("training partition differs")
            dry, wet, controls = read_effect_pair(path, "dfz")
            records.append(source_record(path, corpus, dry, wet, controls))
            if split == "calibration":
                rows.append({"dry": torch.from_numpy(dry), "wet": torch.from_numpy(wet), "controls": torch.from_numpy(controls),
                             "name": path.name})
        validate_records(records, "train", FIT_TAKES if split == "fit" else CALIBRATION_TAKES)
        partitions[split] = records
    output.mkdir(parents=True)
    graph_path = output/"model.onnx"
    export(model, graph_path)
    graph = onnx.load(graph_path)
    metadata = {entry.key: entry.value for entry in graph.metadata_props}
    metadata.update(architecture=ARCHITECTURE, checkpoint_sha256=checkpoint_hash, trained="true", admitted="false")
    del graph.metadata_props[:]
    for key, value in metadata.items():
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(graph, graph_path)
    graph_hash = sha256(graph_path)
    cpu, ort = CheckedMultirate(model, model), graph_runtime(graph_path, model, checkpoint_hash)
    torch.manual_seed(1013)
    cases, probes = [], []
    for index, row in enumerate(rows):
        dry = row["dry"][None]
        controls = row["controls"][None, None].expand(-1, dry.shape[1], -1)
        before = dry.clone(), controls.clone()
        a, b = cpu(dry, controls)[0], ort(dry, controls)[0]
        c = streamed(ort, dry, controls, [1024])[0]
        if not torch.equal(dry, before[0]) or not torch.equal(controls, before[1]):
            raise ValueError("in-memory source changed")
        cases.append({"name": row["name"], "parity_max": float((a-b).abs().max()), "onnx_stream_max": float((b-c).abs().max())})
        if (index+1)%9 == 0:
            print(json.dumps({"real_files": index+1, "parity_max": max(value["parity_max"] for value in cases)}), flush=True)
    for batch, frames in ((2, 6173), (1, 144000), (1, 1440000)):
        dry, controls = torch.randn(batch, frames)*.03, torch.rand(batch, frames, 2)
        before = dry.clone(), controls.clone()
        a, _ = cpu(dry, controls)
        b, state = ort(dry, controls)
        c, chunk_state = streamed(ort, dry, controls, [1, 17, 63, 64, 257, 1024] if frames < 10000 else [1024])
        d, _ = streamed(cpu, dry, controls, [1024])
        probes.append({"batch": batch, "frames": frames, "parity_max": float((a-b).abs().max()),
                       "onnx_stream_max": float((b-c).abs().max()), "cpu_stream_max": float((a-d).abs().max()),
                       "phase_exact": bool(torch.equal(state[-1], chunk_state[-1])),
                       "inputs_unchanged": bool(torch.equal(dry, before[0]) and torch.equal(controls, before[1]))})
        print(json.dumps({"dynamic_probe": probes[-1]}), flush=True)
        del a, b, c, d, before
    safety, timing = {}, {}
    for name, backend in (("cpu", cpu), ("onnx", ort)):
        signal, knobs = torch.randn(3, 6173)*3e-7, torch.rand(3, 6173, 2)
        zero = max(float(backend(torch.zeros_like(signal), controls)[0].abs().max())
                   for controls in (knobs, torch.tensor([[0., 0.], [.5, .5], [1., 1.]])))
        quiet = float(backend(signal, knobs)[0].abs().max())
        excited = torch.cat((torch.randn(1, 1033)*.03, torch.zeros(1, 8199)), 1)
        controls = torch.rand(1, excited.shape[1], 2)
        value = backend(excited, controls)[0]
        tail = float(value[:, 1033+model.audio.receptive_field-1:].abs().max())
        excited[:, 17:] += .4
        controls[:, 17:] = 1-controls[:, 17:]
        future = float((value[:, :17]-backend(excited, controls)[0][:, :17]).abs().max())
        safety[name] = {"zero_peak": zero, "quiet_peak": quiet, "expired_tail_peak": tail, "future_prefix_error": future}
        dry, controls = rows[0]["dry"][None], torch.rand(1, 144000, 2)
        timing[name] = {"full": [], "stream1024": []}
        for _ in range(3):
            tick = time.perf_counter()
            backend(dry, controls)
            timing[name]["full"].append((time.perf_counter()-tick)/3)
            tick = time.perf_counter()
            streamed(backend, dry, controls, [1024])
            timing[name]["stream1024"].append((time.perf_counter()-tick)/3)
    metrics = {name: quality_metrics(backend, rows) for name, backend in (("cpu", cpu), ("onnx", ort))}
    if not finite_numbers({"metrics": metrics, "cases": cases, "probes": probes, "safety": safety, "timing": timing}):
        raise ValueError("nonfinite audit evidence")
    runtime_passed = (all(max(row["parity_max"], row["onnx_stream_max"]) <= 2e-6 for row in cases)
                      and all(max(row["parity_max"], row["onnx_stream_max"], row["cpu_stream_max"]) <= 2e-6
                              and row["phase_exact"] and row["inputs_unchanged"] for row in probes)
                      and all(row["zero_peak"] == row["expired_tail_peak"] == row["future_prefix_error"] == 0.
                              and row["quiet_peak"] <= .001 for row in safety.values())
                      and all(max(values) < 1 for times in timing.values() for values in times.values()))
    quality_pass = {name: bool(_quality_gate(value) and max(row["absolute_peak_error_p95"]
                    for row in value["by_first_control"].values()) <= .03) for name, value in metrics.items()}
    unchanged()
    if sha256(graph_path) != graph_hash:
        raise ValueError("audited graph changed")
    result = {"schema": 1, "purpose": "trained-cascaded-core-calibration-audit", "architecture": ARCHITECTURE, "geometry": GEOMETRY,
              "runtime_passed": runtime_passed, "calibration_passed": all(quality_pass.values()), "quality_pass": quality_pass,
              "calibration": metrics, "real_audio_cases": cases, "dynamic_probes": probes, "safety": safety,
              "single_thread_rtf_including_validation": timing, "structure_bound": bound,
              "parameters": sum(value.numel() for value in model.parameters()), "checkpoint_sha256": checkpoint_hash,
              "training_report_sha256": parent_hash, "graph_sha256": graph_hash, "graph_bytes": graph_path.stat().st_size,
              "source_sha256": sources, "audio_provenance": provenance, "source_and_audio_reverified": True,
              "train_partition_manifest": {"partitions": partitions, "sha256": manifest_sha256(partitions),
                                           "independent_session_holdout": False, "fit_audio_used_for_model_evaluation": False},
              "official_eval_opened": False, "physical_audio_devices_used": False, "source_audio_modified": False,
              "automatic_gain_or_normalization": False, "automatic_limiting": False,
              "admitted": False, "elapsed_seconds": time.perf_counter()-started}
    (output/"metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("runtime_passed", "calibration_passed", "elapsed_seconds")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/asrnn-physical-effects"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit(args.checkpoint, args.output, args.corpus)


if __name__ == "__main__":
    main()
