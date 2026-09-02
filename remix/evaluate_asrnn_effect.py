#!/usr/bin/env python3
"""Evaluate a generic stable ASRNN effect under Muspector audio-quality gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .asrnn_data import LICENSE, RECORD_URL
from .asrnn_effects import effect_files, effect_spec, read_effect_pair
from .centered_stable_effect import load_centered_stable_effect
from .forward_drive import multiresolution_spectral_loss, pre_emphasis
from .stable_effect import load_stable_effect
from .train import device


def manifest_sha256(value: object) -> str:
    """Hash a canonical JSON manifest, not just potentially repeated filenames."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def assert_artifacts_unchanged(checkpoint: Path, checkpoint_digest: str,
                               onnx_path: Path | None = None, onnx_digest: str | None = None) -> None:
    """Do not bind an in-memory model to bytes changed by a concurrent trainer."""
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_digest:
        raise ValueError("checkpoint changed while evaluating; use an immutable candidate snapshot")
    if onnx_path is not None and hashlib.sha256(onnx_path.read_bytes()).hexdigest() != onnx_digest:
        raise ValueError("ONNX graph changed while evaluating; use an immutable candidate snapshot")


def source_record(path: Path, corpus: Path, dry, wet, controls) -> dict:
    """Bind source bytes and the exact float32 arrays admitted for analysis."""
    return {
        "relative_path": path.resolve().relative_to(corpus.resolve()).as_posix(),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dry_sha256": hashlib.sha256(dry.tobytes()).hexdigest(),
        "pair_sha256": hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest(),
        "frames": len(dry),
        "sample_rate": 48_000,
        "channels": 2,
        "controls": controls.tolist(),
        "target_peak_after_warmup": float(np.abs(wet[1_024:]).max()),
    }


def dfz_train_partition_manifest(corpus: Path, calibration_records: list[dict]) -> dict:
    """Freeze train provenance without running the model on fit or eval audio.

    The 234 fit files are decoded only for source/Dry hashes. This prevents an
    identical basename in official train and eval from being mistaken for the
    same recording, and permits a complete train/eval exact-Dry overlap check.
    """
    cached = {row["relative_path"]: row for row in calibration_records}
    partitions: dict[str, list[dict]] = {"fit": [], "calibration": []}
    for path in effect_files(corpus, "dfz", "train"):
        relative = path.resolve().relative_to(corpus.resolve()).as_posix()
        row = cached.get(relative)
        if row is None:
            row = source_record(path, corpus, *read_effect_pair(path, "dfz"))
        partition = "calibration" if int(path.stem.split(",")[-1]) % 5 == 0 else "fit"
        partitions[partition].append(row)
    return {
        "schema": 1,
        "device": "dfz",
        "official_split": "train",
        "selection": "take_id_modulo_5_equals_0_for_calibration",
        "corpus_root": str(corpus.resolve()),
        "partitions": partitions,
        "sha256": manifest_sha256(partitions),
        "fit_audio_used_for_model_evaluation": False,
        "independent_session_holdout": False,
    }


class EffectClips(Dataset):
    def __init__(self, paths: list[Path], device_key: str, *, trace_corpus: Path | None = None) -> None:
        self.paths = paths
        self.device_key = device_key
        self.trace_corpus = trace_corpus

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        dry, wet, controls = read_effect_pair(self.paths[index], self.device_key)
        result = {
            "dry": torch.from_numpy(dry),
            "wet": torch.from_numpy(wet),
            "controls": torch.from_numpy(controls),
        }
        if self.trace_corpus is not None:
            record = source_record(self.paths[index], self.trace_corpus, dry, wet, controls)
            result["source_record_json"] = json.dumps(record, sort_keys=True, allow_nan=False)
        return result


def _quality_gate(metrics: dict) -> bool:
    return bool(
        metrics["global_esr"] <= 0.05
        and metrics["global_esr_relative_improvement"] >= 0.50
        and metrics["global_mae_relative_improvement"] >= 0.50
        and metrics["preemphasis_esr_relative_improvement"] >= 0.50
        and metrics["spectral_loss_relative_improvement"] >= 0.30
        and metrics["mean_per_file_esr"] <= 0.10
        and metrics["median_per_file_esr"] <= 0.05
        and metrics["p95_per_file_esr"] <= 0.25
        and 0.75 <= metrics["peak_ratio_median"] <= 1.25
        and metrics["peak_ratio_p95"] <= 1.35
        and metrics["absolute_peak_error_p95"] <= 0.02
        and metrics["quiet_prediction_peak_maximum"] <= 1.0e-3
        and metrics["static_silence_max_absolute_output"] == 0.0
        and metrics["dynamic_control_silence_max_absolute_output"] == 0.0
        and metrics["stream_max_absolute_error"] <= 2.0e-6
    )


@torch.inference_mode()
def evaluate(
    checkpoint: Path,
    corpus: Path,
    device_key: str,
    frames: int = 2_048,
    centered: bool = False,
    onnx_path: Path | None = None,
    split: str = "eval",
    compute: str | None = None,
) -> dict:
    spec = effect_spec(device_key)
    if split not in {"eval", "dfz-calibration"} or (split == "dfz-calibration" and spec.key != "dfz"):
        raise ValueError("only official eval or the frozen DFZ calibration partition is supported")
    if onnx_path and centered:
        raise ValueError("ONNX stable export cannot use the centered loader")
    if onnx_path and compute not in (None, "cpu"):
        raise ValueError("ONNX effect evaluation is CPU-only")
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    onnx_digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest() if onnx_path else None
    target = torch.device("cpu") if onnx_path else torch.device(compute) if compute else device()
    loader = load_centered_stable_effect if centered else load_stable_effect
    if onnx_path:
        from .onnx_stable_effect import load_onnx_stable_effect
        model, payload = load_onnx_stable_effect(checkpoint, onnx_path)
    else:
        model, payload = loader(checkpoint)
    assert_artifacts_unchanged(checkpoint, checkpoint_digest, onnx_path, onnx_digest)
    if payload["device"] != spec.key:
        raise ValueError(
            f"checkpoint device {payload['device']!r} differs from requested {spec.key!r}"
        )
    model = model.to(target)
    paths = effect_files(corpus, spec.key, "train" if split == "dfz-calibration" else "eval")
    if split == "dfz-calibration":
        paths = [path for path in paths if int(path.stem.split(",")[-1]) % 5 == 0]
    eval_loader = DataLoader(EffectClips(paths, spec.key, trace_corpus=corpus), batch_size=32)
    per_file, ratios, peak_errors, quiet_peaks = [], [], [], []
    per_control: dict[str, list[dict]] = defaultdict(list)
    per_control_tuple: dict[str, list[dict]] = defaultdict(list)
    source_records = []
    total_error = total_energy = total_bypass_error = 0.0
    total_absolute_error = total_bypass_absolute_error = 0.0
    total_emphasized_error = total_emphasized_bypass_error = 0.0
    total_emphasized_energy = 0.0
    spectral_model, spectral_bypass, spectral_weights = [], [], []
    total_frames = 0
    completed_files = 0
    for batch in eval_loader:
        dry = batch["dry"].to(target)
        wet = batch["wet"].to(target)
        controls = batch["controls"].to(target)
        source_records.extend(json.loads(value) for value in batch["source_record_json"])
        state = None
        chunks = []
        for start in range(0, dry.shape[1], frames):
            stop = min(start + frames, dry.shape[1])
            rendered, state = model(dry[:, start:stop], controls, state)
            chunks.append(rendered)
        rendered = torch.cat(chunks, dim=1)[:, 1_024:]
        dry = dry[:, 1_024:]
        wet = wet[:, 1_024:]
        error = (rendered - wet).square()
        bypass_error = (dry - wet).square()
        energy = wet.square()
        total_error += float(error.sum())
        total_bypass_error += float(bypass_error.sum())
        total_energy += float(energy.sum())
        total_absolute_error += float((rendered - wet).abs().sum())
        total_bypass_absolute_error += float((dry - wet).abs().sum())
        emphasized_wet = pre_emphasis(wet)
        total_emphasized_error += float(
            (pre_emphasis(rendered) - emphasized_wet).square().sum()
        )
        total_emphasized_bypass_error += float(
            (pre_emphasis(dry) - emphasized_wet).square().sum()
        )
        total_emphasized_energy += float(emphasized_wet.square().sum())
        spectral_model.append(float(multiresolution_spectral_loss(rendered, wet)))
        spectral_bypass.append(float(multiresolution_spectral_loss(dry, wet)))
        spectral_weights.append(len(controls))
        per_file.extend((error.mean(1) / energy.mean(1).clamp_min(1.0e-8)).cpu().tolist())
        target_peak = wet.abs().amax(1)
        rendered_peak = rendered.abs().amax(1)
        peak_errors.extend((rendered_peak - target_peak).abs().cpu().tolist())
        audible = target_peak >= 1.0e-3
        ratios.extend((rendered_peak[audible] / target_peak[audible]).cpu().tolist())
        quiet_peaks.extend(rendered_peak[~audible].cpu().tolist())
        for control, file_esr, peak_error in zip(
            controls[:, 0].cpu().tolist(),
            (error.mean(1) / energy.mean(1).clamp_min(1.0e-8)).cpu().tolist(),
            (rendered_peak - target_peak).abs().cpu().tolist(),
        ):
            per_control[str(int(round(control * 100.0)))].append(
                {"esr": file_esr, "peak_error": peak_error}
            )
        for control, file_esr, peak_error in zip(
            controls.cpu().tolist(),
            (error.mean(1) / energy.mean(1).clamp_min(1.0e-8)).cpu().tolist(),
            (rendered_peak - target_peak).abs().cpu().tolist(),
        ):
            key = ",".join(str(int(round(value * 100.0))) for value in control)
            per_control_tuple[key].append({"esr": file_esr, "peak_error": peak_error})
        total_frames += dry.numel()
        completed_files += len(controls)
        print(json.dumps({"evaluation_split": split, "files_completed": completed_files, "files_total": len(eval_loader.dataset)}), flush=True)

    silence = torch.zeros(3, 4_096, device=target)
    static_controls = torch.stack(
        (
            torch.zeros(model.control_count, device=target),
            torch.full((model.control_count,), 0.5, device=target),
            torch.ones(model.control_count, device=target),
        )
    )
    static_output, _ = model(silence, static_controls)
    time = torch.linspace(0.0, 1.0, 4_096, device=target)
    dynamic_controls = torch.stack(
        [
            torch.sin(time * (7.0 + 2.0 * index)).mul(0.5).add(0.5)
            for index in range(model.control_count)
        ],
        dim=1,
    ).unsqueeze(0)
    dynamic_output, _ = model(torch.zeros(1, 4_096, device=target), dynamic_controls)
    torch.manual_seed(20260830)
    probe = torch.randn(2, 6_173, device=target) * 0.04
    probe_controls = torch.stack(
        (
            torch.linspace(0.2, 0.8, model.control_count, device=target),
            torch.linspace(0.8, 0.2, model.control_count, device=target),
        )
    )
    whole, _ = model(probe, probe_controls)
    state = None
    streamed = []
    for start, stop in ((0, 17), (17, 513), (513, 2_121), (2_121, 6_173)):
        value, state = model(probe[:, start:stop], probe_controls, state)
        streamed.append(value)
    stream_error = whole - torch.cat(streamed, dim=1)

    global_esr = total_error / max(total_energy, 1.0e-12)
    bypass_esr = total_bypass_error / max(total_energy, 1.0e-12)
    global_mae = total_absolute_error / total_frames
    bypass_mae = total_bypass_absolute_error / total_frames
    emphasized_esr = total_emphasized_error / max(total_emphasized_energy, 1.0e-12)
    bypass_emphasized_esr = total_emphasized_bypass_error / max(
        total_emphasized_energy, 1.0e-12
    )
    # A final partial batch must not receive the same weight as 32 recordings.
    spectral = float(np.average(spectral_model, weights=spectral_weights))
    bypass_spectral = float(np.average(spectral_bypass, weights=spectral_weights))
    metrics = {
        "examples": len(per_file),
        "audible_examples": len(ratios),
        "quiet_examples": len(quiet_peaks),
        "quiet_dataset_gate_vacuous": not quiet_peaks,
        "scored_frames": total_frames,
        "global_esr": global_esr,
        "bypass_global_esr": bypass_esr,
        "global_esr_relative_improvement": 1.0 - global_esr / max(bypass_esr, 1.0e-12),
        "global_mae": global_mae,
        "bypass_global_mae": bypass_mae,
        "global_mae_relative_improvement": 1.0 - global_mae / max(bypass_mae, 1.0e-12),
        "preemphasis_esr": emphasized_esr,
        "bypass_preemphasis_esr": bypass_emphasized_esr,
        "preemphasis_esr_relative_improvement": 1.0
        - emphasized_esr / max(bypass_emphasized_esr, 1.0e-12),
        "multiresolution_spectral_loss": spectral,
        "bypass_multiresolution_spectral_loss": bypass_spectral,
        "spectral_loss_relative_improvement": 1.0
        - spectral / max(bypass_spectral, 1.0e-12),
        "mean_per_file_esr": float(np.mean(per_file)),
        "median_per_file_esr": float(np.median(per_file)),
        "p95_per_file_esr": float(np.quantile(per_file, 0.95)),
        "peak_ratio_median": float(np.median(ratios)),
        "peak_ratio_p95": float(np.quantile(ratios, 0.95)),
        "absolute_peak_error_p95": float(np.quantile(peak_errors, 0.95)),
        "quiet_prediction_peak_maximum": max(quiet_peaks, default=0.0),
        "static_silence_max_absolute_output": float(static_output.abs().max()),
        "dynamic_control_silence_max_absolute_output": float(dynamic_output.abs().max()),
        "stream_max_absolute_error": float(stream_error.abs().max()),
        "by_first_control": {
            key: {
                "files": len(values),
                "mean_esr": float(np.mean([value["esr"] for value in values])),
                "absolute_peak_error_p95": float(
                    np.quantile([value["peak_error"] for value in values], 0.95)
                ),
            }
            for key, values in sorted(per_control.items(), key=lambda item: int(item[0]))
        },
        "by_control_tuple": {
            key: {
                "files": len(values),
                "mean_esr": float(np.mean([value["esr"] for value in values])),
                "absolute_peak_error_p95": float(np.quantile([value["peak_error"] for value in values], 0.95)),
            }
            for key, values in sorted(per_control_tuple.items())
        },
    }
    accepted = _quality_gate(metrics)
    calibration_entry_checks = None
    if split == "dfz-calibration":
        expected_names = {f"{blend},{filter_value},{take}.wav"
                          for blend in (0, 50, 100) for filter_value in (0, 50, 100)
                          for take in (5, 10, 15, 20, 25, 30)}
        calibration_entry_checks = {
            "frozen_audio_quality_gate": accepted,
            "complete54_recordings": len(paths) == 54 and {path.name for path in paths} == expected_names,
            "worst_blend_peak_p95_le_003": max(
                value["absolute_peak_error_p95"] for value in metrics["by_first_control"].values()
            ) <= 0.03,
        }
        accepted = all(calibration_entry_checks.values())
    report = {
        "schema": 1,
        "status": ("passed-calibration-entry" if accepted else "rejected-calibration-entry")
        if split == "dfz-calibration" else "accepted-internal-noncommercial-pilot" if accepted else "rejected",
        "accepted": accepted,
        "device": spec.key,
        "device_name": spec.display_name,
        "runtime": "onnxruntime-cpu" if onnx_path else "zero-centered" if centered else "stable",
        "compute_device": str(target),
        "onnx_sha256": onnx_digest,
        "control_names": list(spec.control_names),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "dataset_record": RECORD_URL,
        "dataset_license": LICENSE,
        "calibration" if split == "dfz-calibration" else "official_eval": metrics,
        **({"calibration_entry_checks": calibration_entry_checks} if calibration_entry_checks is not None else {}),
        "data_manifest": {
            "schema": 1,
            "device": spec.key,
            "official_split": "train" if split == "dfz-calibration" else "eval",
            "partition": "calibration" if split == "dfz-calibration" else "development",
            "corpus_root": str(corpus.resolve()),
            "records": source_records,
            "sha256": manifest_sha256(source_records),
            "warmup_frames_excluded_from_scoring": 1_024,
            "full_causal_prefix_processed": True,
        },
        **({"train_partition_manifest": dfz_train_partition_manifest(corpus, source_records)} if split == "dfz-calibration" else {}),
        "quality_policy": {
            "source_files_read_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
        },
        "release_limitations": [
            "CC-BY-NC-4.0 data and weights prohibit commercial promotion",
            "official eval was observed during development and is not locked-final",
            "no Rust or UI integration is authorized by this pilot",
            *(["calibration entry passing is selection evidence, not forward-model admission"]
              if split == "dfz-calibration" else []),
        ],
    }
    assert_artifacts_unchanged(checkpoint, checkpoint_digest, onnx_path, onnx_digest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("rat", "dfz", "cs3"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, help="evaluate the provenance-matched float32 CPU graph")
    parser.add_argument("--split", choices=("eval", "dfz-calibration"), default="eval")
    parser.add_argument("--compute", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--centered",
        action="store_true",
        help="load the zero-centered stable checkpoint format",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"stable effect evaluation already exists: {args.output}")
    torch.set_num_threads(2)
    report = evaluate(
        args.checkpoint, args.corpus, args.device, centered=args.centered, onnx_path=args.onnx,
        split=args.split, compute=args.compute,
    )
    if args.output.exists():
        raise ValueError(f"stable effect evaluation already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
