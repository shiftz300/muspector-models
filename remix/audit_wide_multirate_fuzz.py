"""Trained-weight audit and optional channel-widening preflight.

Only frozen train/calibration54 audio, no eval or physical devices. A passing
runtime audit is not a fidelity admission. Original narrow weights are retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import onnx
import torch

from .asrnn_effects import read_effect_pair
from .audit_multirate_fuzz import CheckedMultirate, load_trained
from .multirate_fuzz import MultirateFuzz
from .precheck_multirate_fuzz import OrtMultirate, export, sha256, streamed
from .train_asrnn_phase7 import _evaluate
from .widen_multirate_fuzz import WIDE_GEOMETRY, widen


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widen-source", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new audit directory required")
    torch.set_num_threads(1)
    torch.manual_seed(994)
    parent_path = args.checkpoint.with_name("metrics.json")
    parent = json.loads(parent_path.read_text())
    parent_hash, source_hash = sha256(parent_path), sha256(args.checkpoint)
    if not parent.get("source_and_audio_reverified") or parent.get("checkpoint_sha256") != source_hash:
        raise ValueError("completed immutable source training required")
    source = None
    if args.widen_source:
        source, _ = load_trained(args.checkpoint, parent_path)
        model = widen(source)
    else:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if (payload.get("geometry") != WIDE_GEOMETRY or payload.get("architecture") != "causal-multirate-gcn"
                or payload.get("experimental_schema") != 1 or payload.get("device") != "dfz" or payload.get("sample_rate") != 48000):
            raise ValueError("supported wide experimental schema required")
        model = MultirateFuzz(**WIDE_GEOMETRY).eval()
        model.load_state_dict(payload["state_dict"], strict=True)
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("finite float32 weights required")
    names = (Path(__file__).name, "widen_multirate_fuzz.py", "audit_multirate_fuzz.py", "multirate_fuzz.py",
             "precheck_multirate_fuzz.py", "train_asrnn_phase7.py", "asrnn_effects.py", "asrnn_data.py")
    sources = {**parent["source_sha256"], **{name: sha256(Path(__file__).parent / name) for name in names}}
    provenance = parent["audio_provenance"]
    if set(provenance) != {"fit", "calibration"} or len(provenance["fit"]) != 234 or len(provenance["calibration"]) != 54:
        raise ValueError("original234/54 partition required")

    def unchanged():
        if sha256(parent_path) != parent_hash or sha256(args.checkpoint) != source_hash:
            raise ValueError("source report/weights changed")
        if any(sha256(Path(__file__).parent / name) != digest for name, digest in sources.items()):
            raise ValueError("source code changed")
        if any(sha256(row["path"]) != row["sha256"] for rows in provenance.values() for row in rows):
            raise ValueError("original audio changed")

    unchanged()
    started = time.perf_counter()
    rows = []
    for record in provenance["calibration"]:
        path = Path(record["path"])
        if "train" not in path.parts or int(path.stem.split(",")[-1]) % 5 != 0:
            raise ValueError("only frozen train/calibration audio may be decoded")
        dry, wet, controls = read_effect_pair(path, "dfz")
        rows.append({"dry": torch.from_numpy(dry), "wet": torch.from_numpy(wet),
                     "controls": torch.from_numpy(controls), "attack": int(path.stem.split(",")[0]), "name": path.name})
    args.output.mkdir(parents=True)
    checkpoint = args.checkpoint
    if args.widen_source:
        checkpoint = args.output / "candidate.pt"
        torch.save({"experimental_schema": 1, "architecture": "causal-multirate-gcn", "device": "dfz", "sample_rate": 48000,
                    "geometry": WIDE_GEOMETRY, "state_dict": model.state_dict()}, checkpoint)
    checkpoint_hash = sha256(checkpoint)
    graph_path = args.output / "model.onnx"
    export(model, graph_path)
    graph = onnx.load(graph_path)
    for entry in graph.metadata_props:
        if entry.key == "trained":
            entry.value = "true"
    for key, value in {"checkpoint_sha256": checkpoint_hash, "admitted": "false", "geometry": json.dumps(WIDE_GEOMETRY, sort_keys=True)}.items():
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(graph, full_check=True)
    if any(value.data_location == onnx.TensorProto.EXTERNAL for value in graph.graph.initializer):
        raise ValueError("external tensor sidecars forbidden")
    onnx.save(graph, graph_path)
    graph_hash = sha256(graph_path)
    cpu, ort = CheckedMultirate(model, model), CheckedMultirate(OrtMultirate(graph_path, model), model)
    cases = []
    for index, row in enumerate(rows):
        dry = row["dry"][None]
        controls = row["controls"][None, None].expand(-1, len(row["dry"]), -1)
        expected, _ = cpu(dry, controls)
        rendered, _ = ort(dry, controls)
        chunked, _ = streamed(ort, dry, controls, [1024])
        item = {"name": row["name"], "parity_max": float((expected - rendered).abs().max()),
                "onnx_stream_max": float((rendered - chunked).abs().max())}
        if source is not None:
            item["narrow_wide_max"] = float((source(dry, controls)[0] - expected).abs().max())
        cases.append(item)
        if (index + 1) % 9 == 0:
            print(json.dumps({"files": index + 1, "parity_max": max(value["parity_max"] for value in cases),
                              "narrow_wide_max": max(value.get("narrow_wide_max", 0.) for value in cases)}), flush=True)
    probes = []
    for batch, frames in ((2, 6173), (1, 144000), (1, 1440000)):
        dry, controls = torch.randn(batch, frames) * .03, torch.rand(batch, frames, 2)
        before = (dry.clone(), controls.clone())
        a, _ = cpu(dry, controls)
        b, state = ort(dry, controls)
        c, streamed_state = streamed(ort, dry, controls, [1, 17, 63, 64, 257, 1024] if frames < 10000 else [1024])
        d, _ = streamed(cpu, dry, controls, [1024])
        probes.append({"frames": frames, "batch": batch, "parity_max": float((a - b).abs().max()),
                       "onnx_stream_max": float((b - c).abs().max()), "cpu_stream_max": float((a - d).abs().max()),
                       "phase_exact": bool(torch.equal(state[-1], streamed_state[-1])),
                       "inputs_unchanged": bool(torch.equal(before[0], dry) and torch.equal(before[1], controls))})
        print(json.dumps({"dynamic_probe": probes[-1]}), flush=True)
        del a, b, c, d, before
    signal = torch.randn(3, 6173) * 3e-7
    controls = torch.rand(3, 6173, 2)
    safety, timing = {}, {}
    for name, backend in (("cpu", cpu), ("onnx", ort)):
        zero = max(float(backend(torch.zeros_like(signal), knobs)[0].abs().max())
                   for knobs in (controls, torch.tensor([[0., 0.], [.5, .5], [1., 1.]])))
        quiet = float(backend(signal, controls)[0].abs().max())
        excited = torch.cat((torch.randn(1, 1033) * .03, torch.zeros(1, 4099)), 1)
        knobs = torch.rand(1, excited.shape[1], 2)
        value = backend(excited, knobs)[0]
        tail = float(value[:, 1033 + model.audio.receptive_field - 1:].abs().max())
        excited[:, 17:] += .4
        knobs[:, 17:] = 1 - knobs[:, 17:]
        future = float((value[:, :17] - backend(excited, knobs)[0][:, :17]).abs().max())
        safety[name] = {"zero_peak": zero, "quiet_peak": quiet, "expired_tail_peak": tail, "future_prefix_error": future}
        dry, knobs = rows[0]["dry"][None], torch.rand(1, 144000, 2)
        timing[name] = {"full": [], "stream1024": []}
        for _ in range(3):
            tick = time.perf_counter()
            backend(dry, knobs)
            timing[name]["full"].append((time.perf_counter() - tick) / 3)
            tick = time.perf_counter()
            streamed(backend, dry, knobs, [1024])
            timing[name]["stream1024"].append((time.perf_counter() - tick) / 3)
    metrics = {name: _evaluate(backend, rows, torch.device("cpu"), 9) for name, backend in (("cpu", cpu), ("onnx", ort))}
    passed = (all(max(row["parity_max"], row["onnx_stream_max"], row.get("narrow_wide_max", 0.)) <= 2e-6 for row in cases)
              and all(max(row["parity_max"], row["onnx_stream_max"], row["cpu_stream_max"]) <= 2e-6
                      and row["phase_exact"] and row["inputs_unchanged"] for row in probes)
              and all(row["zero_peak"] == row["expired_tail_peak"] == row["future_prefix_error"] == 0.
                      and row["quiet_peak"] <= 1e-3 for row in safety.values())
              and all(max(value) < 1 for backend in timing.values() for value in backend.values()))
    unchanged()
    if sha256(checkpoint) != checkpoint_hash or sha256(graph_path) != graph_hash:
        raise ValueError("audited artifact changed")
    result = {"schema": 1, "purpose": "function-preserving-widen-preflight" if args.widen_source else "trained-wide-core-audit",
              "runtime_passed": passed, "calibration_passed": all(value["passes_selection_gate"] for value in metrics.values()),
              "calibration": metrics, "real_audio_cases": cases, "dynamic_probes": probes, "safety": safety,
              "single_thread_rtf_including_validation": timing, "parameters": sum(value.numel() for value in model.parameters()),
              "parent_checkpoint_sha256": source_hash, "parent_report_sha256": parent_hash,
              "checkpoint_sha256": checkpoint_hash, "graph_sha256": graph_hash, "graph_bytes": graph_path.stat().st_size,
              "source_sha256": sources, "audio_provenance": provenance, "source_and_audio_reverified": True,
              "geometry": WIDE_GEOMETRY, "official_eval_opened": False, "physical_audio_devices_used": False,
              "source_audio_modified": False, "automatic_gain_or_normalization": False, "admitted": False,
              "elapsed_seconds": time.perf_counter() - started}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("runtime_passed", "calibration_passed", "parameters", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
