"""Full frozen DFZ quality metrics for experimental finite-history models.

Calibration reads only train54 plus train234 for provenance. Official eval288
is already a development set, but remains inaccessible here until a complete,
source-bound CPU calibration report passes every original quality threshold.
This evaluator never promotes a model or accesses an audio device/locked final.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch

from .asrnn_data import LICENSE, RECORD_URL
from .asrnn_effects import effect_files, read_effect_pair
from .audit_multirate_fuzz import CheckedMultirate
from .centered_multirate_fuzz import CenteredMultirateFuzz
from .evaluate_asrnn_effect import (_quality_gate, dfz_train_partition_manifest, manifest_sha256, source_record)
from .forward_drive import multiresolution_spectral_loss, pre_emphasis
from .multirate_fuzz import MultirateFuzz
from .precheck_multirate_fuzz import OrtMultirate, sha256, streamed
from .promote_dfz_forward import (ALL_TAKES, CALIBRATION_TAKES, FIT_TAKES, finite_numbers,
                                 validate_data_manifest, validate_metrics, validate_records)


GEOMETRIES = ({"audio_width": 16, "audio_blocks": 10, "controller_width": 8, "controller_blocks": 10},
              {"audio_width": 32, "audio_blocks": 10, "controller_width": 16, "controller_blocks": 10})
PLAIN_ARCHITECTURE = "causal-multirate-gcn"
CENTERED_ARCHITECTURE = "zero-origin-centered-causal-multirate"
ARCHITECTURES = (PLAIN_ARCHITECTURE, CENTERED_ARCHITECTURE)


def model_from_payload(payload):
    architecture = payload.get("architecture")
    if (payload.get("experimental_schema") != 1 or architecture not in ARCHITECTURES
            or payload.get("device") != "dfz" or payload.get("sample_rate") != 48000
            or payload.get("geometry") not in GEOMETRIES
            or (architecture == CENTERED_ARCHITECTURE and payload["geometry"] != GEOMETRIES[1])):
        raise ValueError("supported finite-history experimental DFZ schema required")
    kind = MultirateFuzz if architecture == PLAIN_ARCHITECTURE else CenteredMultirateFuzz
    model = kind(**payload["geometry"]).eval().requires_grad_(False)
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in payload["state_dict"].values()):
        raise ValueError("finite float32 weights required")
    return model


def load_candidate(checkpoint):
    parent_path = checkpoint.with_name("metrics.json")
    parent = json.loads(parent_path.read_text())
    digest = sha256(checkpoint)
    if not parent.get("source_and_audio_reverified") or parent.get("checkpoint_sha256") != digest:
        raise ValueError("completed source-bound training evidence required")
    for name, expected in parent["source_sha256"].items():
        if sha256(Path(__file__).parent / name) != expected:
            raise ValueError("training source changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = model_from_payload(payload)
    return model, payload, parent


def validate_calibration_entry(report, checkpoint_digest, architecture=None):
    if (report.get("schema") != 1 or report.get("device") != "dfz"
            or report.get("architecture") not in ARCHITECTURES
            or (architecture is not None and report["architecture"] != architecture)
            or report.get("runtime") != "multirate-pytorch-cpu"
            or report.get("checkpoint_sha256") != checkpoint_digest or report.get("compute_device") != "cpu"
            or report.get("control_names") != ["blend", "filter"] or report.get("onnx_sha256") is not None):
        raise ValueError("same checkpoint's CPU calibration entry required")
    policy = report.get("quality_policy", {})
    if policy.get("source_files_read_only") is not True or any(policy.get(key) is not False for key in
            ("automatic_normalization", "automatic_limiting", "lossy_reencoding", "physical_audio_devices_used")):
        raise ValueError("calibration audio-quality policy differs")
    records = validate_data_manifest(report, "calibration")
    validate_metrics(report, "calibration", records)
    manifest = report["train_partition_manifest"]
    partitions = manifest["partitions"]
    if (manifest.get("schema") != 1 or manifest.get("device") != "dfz" or manifest.get("official_split") != "train"
            or manifest.get("selection") != "take_id_modulo_5_equals_0_for_calibration"
            or manifest.get("fit_audio_used_for_model_evaluation") is not False
            or set(partitions) != {"fit", "calibration"} or manifest.get("sha256") != manifest_sha256(partitions)
            or manifest.get("independent_session_holdout") is not False):
        raise ValueError("complete fit/calibration provenance required")
    validate_records(partitions["fit"], "train", FIT_TAKES)
    if validate_records(partitions["calibration"], "train", CALIBRATION_TAKES) != records:
        raise ValueError("calibration source manifest differs")
    if not report.get("source_and_audio_reverified") or report.get("admitted") is not False:
        raise ValueError("verified experimental calibration report required")
    return partitions


def graph_runtime(path, model, checkpoint_digest):
    graph = onnx.load(path, load_external_data=False)
    metadata = {entry.key: entry.value for entry in graph.metadata_props}
    if (metadata.get("checkpoint_sha256") != checkpoint_digest or metadata.get("sample_rate") != "48000"
            or metadata.get("precision") != "float32" or metadata.get("trained") != "true"):
        raise ValueError("same trained float32 checkpoint graph required")
    expected_architecture = CENTERED_ARCHITECTURE if type(model) is CenteredMultirateFuzz else "experimental-causal-two-rate-gcn"
    if type(model) not in (MultirateFuzz, CenteredMultirateFuzz) or metadata.get("architecture") != expected_architecture:
        raise ValueError("graph/reference architecture differs")
    for tensor in graph.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
            raise ValueError("external tensor sidecars forbidden")
        kind = onnx.TensorProto.DataType.Name(tensor.data_type)
        if ("FLOAT" in kind or kind == "DOUBLE") and kind != "FLOAT":
            raise ValueError("non-float32 model weights forbidden")
    if any("Quantize" in node.op_type or node.op_type.startswith("QLinear") for node in graph.graph.node):
        raise ValueError("quantized operators forbidden")
    onnx.checker.check_model(graph, full_check=True)
    runtime = OrtMultirate(path, model)
    states = [name for index in range(len(model.state_widths)) for name in (f"h{index}", f"c{index}")]
    states += ["pending", "held", "phase"]
    if runtime.names != ["dry", "controls", *states]:
        raise ValueError("multirate graph state signature differs")
    if [value.name for value in runtime.session.get_outputs()] != ["rendered", *["next_" + name for name in states]]:
        raise ValueError("multirate graph output signature differs")
    return CheckedMultirate(runtime, model)


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def quality_metrics(backend, rows, batch_size=9):
    """Same formulas/0.85 pre-emphasis/spectral definition as the frozen gate."""
    total = defaultdict(float)
    files, ratios, peaks, quiet = [], [], [], []
    groups, tuples = defaultdict(list), defaultdict(list)
    spectral, bypass_spectral, batch_counts = [], [], []
    for first in range(0, len(rows), batch_size):
        batch = rows[first:first + batch_size]
        dry, wet, controls = (torch.stack([row[key] for row in batch]) for key in ("dry", "wet", "controls"))
        before = tuple(tensor_hash(value) for value in (dry, wet, controls))
        rendered, _ = streamed(backend, dry, controls[:, None].expand(-1, dry.shape[1], -1), [4096])
        if before != tuple(tensor_hash(value) for value in (dry, wet, controls)):
            raise ValueError("in-memory input audio/controls changed")
        rendered, dry, wet = rendered[:, 1024:], dry[:, 1024:], wet[:, 1024:]
        error, energy, bypass = (rendered - wet).square(), wet.square(), (dry - wet).square()
        total["error"] += float(error.sum())
        total["energy"] += float(energy.sum())
        total["bypass"] += float(bypass.sum())
        total["absolute"] += float((rendered - wet).abs().sum())
        total["bypass_absolute"] += float((dry - wet).abs().sum())
        expected_pre = pre_emphasis(wet)
        total["pre_error"] += float((pre_emphasis(rendered) - expected_pre).square().sum())
        total["pre_bypass"] += float((pre_emphasis(dry) - expected_pre).square().sum())
        total["pre_energy"] += float(expected_pre.square().sum())
        total["frames"] += dry.numel()
        spectral.append(float(multiresolution_spectral_loss(rendered, wet)))
        bypass_spectral.append(float(multiresolution_spectral_loss(dry, wet)))
        batch_counts.append(len(batch))
        esr = (error.mean(1) / energy.mean(1).clamp_min(1e-8)).tolist()
        target_peak, predicted_peak = wet.abs().amax(1), rendered.abs().amax(1)
        peak_errors = (predicted_peak - target_peak).abs().tolist()
        audible = target_peak >= 1e-3
        ratios.extend((predicted_peak[audible] / target_peak[audible]).tolist())
        quiet.extend(predicted_peak[~audible].tolist())
        files.extend(esr)
        peaks.extend(peak_errors)
        for knobs, value, peak in zip(controls.tolist(), esr, peak_errors):
            a, b = (int(round(control * 100)) for control in knobs)
            groups[str(a)].append((value, peak))
            tuples[f"{a},{b}"].append((value, peak))
        print(json.dumps({"stage": "full-quality", "completed_files": first + len(batch), "files": len(rows)}), flush=True)
    if not rows or total["energy"] <= 0 or total["pre_energy"] <= 0 or not ratios:
        raise ValueError("nonempty audible reference energy required")
    torch.manual_seed(20260830)
    static = backend(torch.zeros(3, 4096), torch.tensor([[0., 0.], [.5, .5], [1., 1.]]))[0]
    time_axis = torch.linspace(0., 1., 4096)
    knobs = torch.stack([torch.sin(time_axis * (7 + 2 * index)) * .5 + .5 for index in range(2)], 1)[None]
    dynamic = backend(torch.zeros(1, 4096), knobs)[0]
    probe, controls = torch.randn(2, 6173) * .04, torch.tensor([[.2, .8], [.8, .2]])
    whole = backend(probe, controls)[0]
    state, values = None, []
    for start, stop in ((0, 17), (17, 513), (513, 2121), (2121, 6173)):
        value, state = backend(probe[:, start:stop], controls, state)
        values.append(value)
    stream_error = float((whole - torch.cat(values, 1)).abs().max())

    def summarize(data):
        return {key: {"files": len(values), "mean_esr": float(np.mean([value[0] for value in values])),
                      "absolute_peak_error_p95": float(np.quantile([value[1] for value in values], .95))}
                for key, values in sorted(data.items())}

    spec, bypass_spec = float(np.average(spectral, weights=batch_counts)), float(np.average(bypass_spectral, weights=batch_counts))
    return {"examples": len(files), "audible_examples": len(ratios), "quiet_examples": len(quiet),
            "quiet_dataset_gate_vacuous": not quiet, "scored_frames": int(total["frames"]),
            "global_esr": total["error"] / total["energy"], "bypass_global_esr": total["bypass"] / total["energy"],
            "global_esr_relative_improvement": 1 - total["error"] / max(total["bypass"], 1e-12),
            "global_mae": total["absolute"] / total["frames"], "bypass_global_mae": total["bypass_absolute"] / total["frames"],
            "global_mae_relative_improvement": 1 - total["absolute"] / max(total["bypass_absolute"], 1e-12),
            "preemphasis_esr": total["pre_error"] / total["pre_energy"], "bypass_preemphasis_esr": total["pre_bypass"] / total["pre_energy"],
            "preemphasis_esr_relative_improvement": 1 - total["pre_error"] / max(total["pre_bypass"], 1e-12),
            "multiresolution_spectral_loss": spec, "bypass_multiresolution_spectral_loss": bypass_spec,
            "spectral_loss_relative_improvement": 1 - spec / max(bypass_spec, 1e-12),
            "mean_per_file_esr": float(np.mean(files)), "median_per_file_esr": float(np.median(files)),
            "p95_per_file_esr": float(np.quantile(files, .95)), "peak_ratio_median": float(np.median(ratios)),
            "peak_ratio_p95": float(np.quantile(ratios, .95)), "absolute_peak_error_p95": float(np.quantile(peaks, .95)),
            "quiet_prediction_peak_maximum": max(quiet, default=0.), "static_silence_max_absolute_output": float(static.abs().max()),
            "dynamic_control_silence_max_absolute_output": float(dynamic.abs().max()), "stream_max_absolute_error": stream_error,
            "by_first_control": summarize(groups), "by_control_tuple": summarize(tuples)}


@torch.inference_mode()
def evaluate(checkpoint, corpus, partition="calibration", onnx_path=None, entry_path=None):
    if partition not in ("calibration", "development"):
        raise ValueError("calibration or previously observed development only; locked-final forbidden")
    if partition == "development" and entry_path is None:
        raise ValueError("no development audio access without passing CPU calibration entry")
    model, payload, parent = load_candidate(checkpoint)
    original_audio = parent["audio_provenance"]
    if set(original_audio) != {"fit", "calibration"} or len(original_audio["fit"]) != 234 or len(original_audio["calibration"]) != 54:
        raise ValueError("original training audio provenance incomplete")
    for records in original_audio.values():
        for record in records:
            path = Path(record["path"])
            path.resolve().relative_to(corpus.resolve())
            if sha256(path) != record["sha256"]:
                raise ValueError("original training/calibration audio changed")
    checkpoint_hash, parent_hash = sha256(checkpoint), sha256(checkpoint.with_name("metrics.json"))
    graph_hash = sha256(onnx_path) if onnx_path else None
    names = (Path(__file__).name, "multirate_fuzz.py", "centered_multirate_fuzz.py", "audit_multirate_fuzz.py", "precheck_multirate_fuzz.py",
             "asrnn_effects.py", "asrnn_data.py", "evaluate_asrnn_effect.py", "promote_dfz_forward.py", "forward_drive.py")
    sources = {**parent["source_sha256"], **{name: sha256(Path(__file__).parent / name) for name in names}}
    entry_hash, partitions = None, None
    if partition == "development":
        if entry_path is None:
            raise ValueError("no development audio access without passing CPU calibration entry")
        entry = json.loads(entry_path.read_text())
        partitions = validate_calibration_entry(entry, checkpoint_hash, payload["architecture"])
        if Path(entry["data_manifest"]["corpus_root"]).resolve() != corpus.resolve():
            raise ValueError("development corpus differs from calibrated corpus")
        for name, digest in entry["source_sha256"].items():
            if sha256(Path(__file__).parent / name) != digest:
                raise ValueError("calibration evaluator source changed")
        for records in partitions.values():
            for record in records:
                if sha256(corpus / record["relative_path"]) != record["file_sha256"]:
                    raise ValueError("calibration/training source changed before development access")
        entry_hash = sha256(entry_path)
    # The eval path is not listed or decoded until the full calibration gate above.
    split = "train" if partition == "calibration" else "eval"
    paths = effect_files(corpus, "dfz", split)
    if partition == "calibration":
        paths = [path for path in paths if int(path.stem.split(",")[-1]) in CALIBRATION_TAKES]
    rows, records = [], []
    for path in paths:
        dry, wet, controls = read_effect_pair(path, "dfz")
        records.append(source_record(path, corpus, dry, wet, controls))
        rows.append({"dry": torch.from_numpy(dry), "wet": torch.from_numpy(wet), "controls": torch.from_numpy(controls)})
    validate_records(records, split, CALIBRATION_TAKES if partition == "calibration" else ALL_TAKES)
    train_manifest = dfz_train_partition_manifest(corpus, records) if partition == "calibration" else None
    if partition == "development":
        train_hashes = {row["dry_sha256"] for values in partitions.values() for row in values}
        if train_hashes & {row["dry_sha256"] for row in records}:
            raise ValueError("official train/development exact Dry overlap")
    backend = graph_runtime(onnx_path, model, checkpoint_hash) if onnx_path else CheckedMultirate(model, model)
    metrics = quality_metrics(backend, rows)
    if not finite_numbers(metrics):
        raise ValueError("nonfinite quality evidence")
    passed = _quality_gate(metrics) and max(row["absolute_peak_error_p95"] for row in metrics["by_first_control"].values()) <= .03
    report = {"schema": 1, "architecture": payload["architecture"], "geometry": payload["geometry"],
              "device": "dfz", "control_names": ["blend", "filter"], "compute_device": "cpu",
              "runtime": "onnxruntime-cpu" if onnx_path else "multirate-pytorch-cpu", "accepted": bool(passed), "admitted": False,
              "status": ("passed-" if passed else "rejected-") + partition + "-quality-not-admitted",
              "calibration" if partition == "calibration" else "official_eval": metrics,
              "checkpoint_sha256": checkpoint_hash, "onnx_sha256": graph_hash, "training_report_sha256": parent_hash,
              "calibration_entry_sha256": entry_hash, "source_sha256": sources,
              "data_manifest": {"schema": 1, "device": "dfz", "official_split": split, "partition": partition,
                                "corpus_root": str(corpus.resolve()), "records": records, "sha256": manifest_sha256(records),
                                "warmup_frames_excluded_from_scoring": 1024, "full_causal_prefix_processed": True},
              **({"train_partition_manifest": train_manifest} if train_manifest else {}),
              "quality_policy": {"source_files_read_only": True, "automatic_normalization": False, "automatic_limiting": False,
                                 "lossy_reencoding": False, "physical_audio_devices_used": False},
              "source_and_audio_reverified": True, "dataset_record": RECORD_URL, "dataset_license": LICENSE,
              "release_limitations": ["development only; no product admission", "not independent locked-final validation",
                                      "no UI or physical audio devices", "unseen continuous controls not validated"]}
    # Recheck every source after rendering, including provenance-only fit files.
    bound = records + ([row for values in train_manifest["partitions"].values() for row in values] if train_manifest else
                       [row for values in partitions.values() for row in values])
    if any(sha256(corpus / row["relative_path"]) != row["file_sha256"] for row in bound):
        raise ValueError("source audio changed during evaluation")
    if any(sha256(row["path"]) != row["sha256"] for values in original_audio.values() for row in values):
        raise ValueError("original training audio changed during evaluation")
    if sha256(checkpoint) != checkpoint_hash or sha256(checkpoint.with_name("metrics.json")) != parent_hash:
        raise ValueError("candidate weights/training report changed")
    if onnx_path and sha256(onnx_path) != graph_hash:
        raise ValueError("ONNX graph changed")
    if entry_path and entry_hash and sha256(entry_path) != entry_hash:
        raise ValueError("calibration entry changed")
    if any(sha256(Path(__file__).parent / name) != digest for name, digest in sources.items()):
        raise ValueError("quality evaluator source changed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/asrnn-physical-effects"))
    parser.add_argument("--partition", choices=("calibration", "development"), default="calibration")
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--calibration-entry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new quality evidence file required")
    torch.set_num_threads(1)
    report = evaluate(args.checkpoint, args.corpus, args.partition, args.onnx, args.calibration_entry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "accepted": report["accepted"], "admitted": report["admitted"]}), flush=True)


if __name__ == "__main__":
    main()
