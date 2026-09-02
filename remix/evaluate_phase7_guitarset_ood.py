#!/usr/bin/env python3
"""Evaluate Phase-7 runtime stability on unpaired GuitarSet Dry audio."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import soundfile
import torch
from scipy.signal import resample_poly

from .centered_stable_effect import load_centered_stable_effect
from .stable_effect import load_stable_effect
from .train import device


EXPECTED_MD5 = "aecce79f425a44e2055e46f680e10f6a"
MEMBER = re.compile(r"^(?P<player>\d\d)_.+_(?P<style>comp|solo)_mix\.wav$")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _members(archive: zipfile.ZipFile, per_player_style: int) -> list[str]:
    groups: dict[tuple[str, str], list[str]] = {}
    for name in archive.namelist():
        match = MEMBER.match(Path(name).name)
        if match:
            key = (match.group("player"), match.group("style"))
            groups.setdefault(key, []).append(name)
    expected = {(f"{player:02d}", style) for player in range(6) for style in ("comp", "solo")}
    if set(groups) != expected:
        raise ValueError("GuitarSet archive does not contain all player/style groups")
    if per_player_style <= 0 or any(per_player_style > len(names) for names in groups.values()):
        raise ValueError("GuitarSet per-player/style count is outside the archive coverage")
    return [
        sorted(groups[key])[index * len(groups[key]) // per_player_style]
        for key in sorted(groups)
        for index in range(per_player_style)
    ]


def _read_segment(archive: zipfile.ZipFile, name: str, seconds: float) -> np.ndarray:
    with archive.open(name) as source:
        audio, sample_rate = soundfile.read(
            io.BytesIO(source.read()), dtype="float32", always_2d=True
        )
    if sample_rate != 44_100 or audio.shape[1] != 1 or not np.isfinite(audio).all():
        raise ValueError(f"unexpected GuitarSet audio geometry: {name}")
    mono = audio[:, 0]
    frames = min(len(mono), int(round(seconds * sample_rate)))
    start = max(0, (len(mono) - frames) // 2)
    segment = mono[start : start + frames]
    return resample_poly(segment, 160, 147).astype(np.float32, copy=False)


@torch.inference_mode()
def _render(model, audio: np.ndarray, control: float) -> tuple[np.ndarray, float, float]:
    target = next(model.parameters()).device
    dry = torch.from_numpy(audio).to(target)
    if dry.ndim != 2:
        raise ValueError("GuitarSet batch must be [batch,frames]")
    controls = torch.full((dry.shape[0], 1), control, dtype=torch.float32, device=target)
    state = None
    chunks = []
    for start in range(0, dry.shape[1], 32_768):
        value, state = model(dry[:, start : start + 32_768], controls, state)
        chunks.append(value)
    streamed = torch.cat(chunks, dim=1)
    probe_frames = min(8_192, dry.shape[1])
    whole_probe, _ = model(dry[:, :probe_frames], controls)
    probe_state = None
    probe_chunks = []
    start = 0
    for stop in sorted({min(boundary, probe_frames) for boundary in (17, 513, 2_121, probe_frames)}):
        value, probe_state = model(dry[:, start:stop], controls, probe_state)
        probe_chunks.append(value)
        start = stop
    error = max(
        float((whole_probe - torch.cat(probe_chunks, dim=1)).abs().max()),
        float((whole_probe - streamed[:, :probe_frames]).abs().max()),
    )
    silence = torch.zeros(dry.shape[0], 8_192, device=target)
    for _ in range(4):
        tail, state = model(silence, controls, state)
    return streamed.cpu().numpy(), error, float(tail.abs().max())


def _high_frequency_ratio(audio: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(audio.astype(np.float64))) ** 2
    frequencies = np.fft.rfftfreq(len(audio), 1.0 / 48_000.0)
    return float(spectrum[frequencies >= 18_000.0].sum() / max(spectrum.sum(), 1.0e-12))


@torch.inference_mode()
def evaluate(args) -> dict:
    if args.output.exists():
        raise ValueError(f"GuitarSet OOD report already exists: {args.output}")
    if args.seconds <= 0.0 or args.batch_size <= 0:
        raise ValueError("GuitarSet duration and batch size must be positive")
    if _md5(args.archive) != EXPECTED_MD5:
        raise ValueError("GuitarSet archive MD5 differs from the official record")
    base, base_payload = load_stable_effect(args.base)
    candidate_loader = load_stable_effect if getattr(args, "stable", False) else load_centered_stable_effect
    onnx_path = getattr(args, "onnx", None)
    if onnx_path:
        from .onnx_stable_effect import load_onnx_stable_effect
        candidate, candidate_payload = load_onnx_stable_effect(args.checkpoint, onnx_path)
    else:
        candidate, candidate_payload = candidate_loader(args.checkpoint)
    if base_payload["device"] != "cs3" or candidate_payload["device"] != "cs3":
        raise ValueError("GuitarSet OOD evaluation requires CS-3 models")
    backend = getattr(args, "backend", "auto")
    target = torch.device("cpu") if onnx_path else device() if backend == "auto" else torch.device(backend)
    baseline = base.to(target).eval()
    candidate = candidate.to(target).eval()
    rows = []
    sample_rates = set()
    with zipfile.ZipFile(args.archive) as archive:
        damaged = archive.testzip()
        if damaged is not None:
            raise ValueError(f"GuitarSet archive member failed CRC validation: {damaged}")
        names = _members(archive, args.per_player_style)
        for offset in range(0, len(names), args.batch_size):
            batch_names = names[offset : offset + args.batch_size]
            audio_rows = []
            for name in batch_names:
                with archive.open(name) as source:
                    sample_rates.add(soundfile.info(io.BytesIO(source.read())).samplerate)
                audio_rows.append(_read_segment(archive, name, args.seconds))
            frames = min(map(len, audio_rows))
            audio_batch = np.stack([audio[:frames] for audio in audio_rows])
            for control in (0.0, 0.5, 1.0):
                expected_batch, base_stream_error, _ = _render(baseline, audio_batch, control)
                rendered_batch, stream_error, tail_peak = _render(candidate, audio_batch, control)
                for name, audio, expected, rendered in zip(
                    batch_names, audio_batch, expected_batch, rendered_batch
                ):
                    input_rms = float(np.sqrt(np.mean(np.square(audio))))
                    expected_rms = float(np.sqrt(np.mean(np.square(expected))))
                    rendered_rms = float(np.sqrt(np.mean(np.square(rendered))))
                    delta_rms = float(np.sqrt(np.mean(np.square(rendered - expected))))
                    rows.append({
                        "member": name,
                        "control": control,
                        "input_rms": input_rms,
                        "candidate_peak": float(np.max(np.abs(rendered))),
                        "candidate_rms": rendered_rms,
                        "candidate_dc": float(np.mean(rendered)),
                        "candidate_high_frequency_energy_ratio": _high_frequency_ratio(
                            rendered
                        ),
                        "candidate_to_base_peak_ratio": float(
                            np.max(np.abs(rendered))
                            / max(float(np.max(np.abs(expected))), 1.0e-8)
                        ),
                        "candidate_to_base_delta_rms_ratio": delta_rms
                        / max(expected_rms, 1.0e-8),
                        "candidate_output_to_input_rms_ratio": rendered_rms
                        / max(input_rms, 1.0e-8),
                        "stream_max_absolute_error": stream_error,
                        "base_stream_max_absolute_error": base_stream_error,
                        "post_signal_silence_tail_peak": tail_peak,
                        "finite": bool(np.isfinite(rendered).all()),
                    })
            print(json.dumps({"ood_source_members_completed": offset + len(batch_names), "ood_source_members_total": len(names), "control_cases_completed": len(rows)}), flush=True)
    silence = torch.zeros(3, 4_096, device=target)
    controls = torch.tensor(((0.0,), (0.5,), (1.0,)), device=target)
    silence_output, _ = candidate(silence, controls)
    metrics = {
        "examples": len(rows),
        "source_members": len({row["member"] for row in rows}),
        "candidate_peak_maximum": max(row["candidate_peak"] for row in rows),
        "candidate_absolute_dc_maximum": max(abs(row["candidate_dc"]) for row in rows),
        "candidate_high_frequency_energy_ratio_p95": float(
            np.quantile(
                [row["candidate_high_frequency_energy_ratio"] for row in rows], 0.95
            )
        ),
        "candidate_to_base_peak_ratio_p05": float(
            np.quantile([row["candidate_to_base_peak_ratio"] for row in rows], 0.05)
        ),
        "candidate_to_base_peak_ratio_p95": float(
            np.quantile([row["candidate_to_base_peak_ratio"] for row in rows], 0.95)
        ),
        "candidate_to_base_delta_rms_ratio_p95": float(
            np.quantile(
                [row["candidate_to_base_delta_rms_ratio"] for row in rows], 0.95
            )
        ),
        "candidate_output_to_input_rms_ratio_p95": float(
            np.quantile(
                [row["candidate_output_to_input_rms_ratio"] for row in rows], 0.95
            )
        ),
        "stream_max_absolute_error": max(
            row["stream_max_absolute_error"] for row in rows
        ),
        "static_silence_max_absolute_output": float(silence_output.abs().max()),
        "post_signal_silence_tail_peak_maximum": max(
            row["post_signal_silence_tail_peak"] for row in rows
        ),
        "all_finite": all(row["finite"] for row in rows),
    }
    checks = {
        "finite": metrics["all_finite"],
        "peak_le_2": metrics["candidate_peak_maximum"] <= 2.0,
        "absolute_dc_le_001": metrics["candidate_absolute_dc_maximum"] <= 0.01,
        "high_frequency_energy_p95_le_010": metrics["candidate_high_frequency_energy_ratio_p95"] <= 0.10,
        "base_peak_ratio_p05_ge_050": metrics["candidate_to_base_peak_ratio_p05"] >= 0.50,
        "base_peak_ratio_p95_le_150": metrics["candidate_to_base_peak_ratio_p95"] <= 1.50,
        "base_delta_rms_ratio_p95_le_035": metrics["candidate_to_base_delta_rms_ratio_p95"] <= 0.35,
        "stream_error_le_2e_minus6": metrics["stream_max_absolute_error"] <= 2.0e-6,
        "static_silence_exact_zero": metrics["static_silence_max_absolute_output"] == 0.0,
        "late_tail_le_1e_minus4": metrics["post_signal_silence_tail_peak_maximum"] <= 1.0e-4,
    }
    passed = all(checks.values())
    return {
        "schema": 1,
        "phase": "phase-7-guitarset-unpaired-ood-runtime",
        "candidate_diagnostic_only": bool(candidate_payload.get("diagnostic_only", False)),
        "candidate_runtime": "onnxruntime-cpu" if onnx_path else "stable" if getattr(args, "stable", False) else "zero-centered",
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest() if onnx_path else None,
        "baseline_runtime": "stable; verified source zero response is exact",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "base_checkpoint_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
        "base_source_checkpoint_sha256": base_payload.get("source_checkpoint_sha256"),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "archive": str(args.archive),
        "archive_bytes": args.archive.stat().st_size,
        "archive_md5": EXPECTED_MD5,
        "archive_tested": True,
        "source_sample_rates": sorted(sample_rates),
        "source_players": sorted({Path(name).name.split("_")[0] for name in names}),
        "source_genres": sorted(
            {re.match(r"[A-Za-z]+", Path(name).name.split("_")[1]).group(0) for name in names}
        ),
        "compute_device": str(target),
        "segment_seconds": args.seconds,
        "selection": "evenly spaced filenames within each player/style group; five genres at default count",
        "analysis_sample_rate": 48_000,
        "stream_block_frames": 32_768,
        "stream_parity_probe_boundaries": [17, 513, 2121, 8192],
        "post_signal_silence_tail_window_seconds": [0.512, 32768.0 / 48000.0],
        "analysis_copy_resampling": "polyphase 160/147 in memory",
        "license": "CC-BY-4.0",
        "dataset_doi": "10.5281/zenodo.3371780",
        "metrics": metrics,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "largest_drift_examples": sorted(
            rows, key=lambda row: row["candidate_to_base_delta_rms_ratio"], reverse=True
        )[:6],
        "limitations": [
            "unpaired clean guitar cannot measure CS-3 clone fidelity",
            "this is a runtime stability and quality-drift stress test only",
            "no rendered audio or modified source audio is retained",
        ],
        "quality_policy": {
            "source_archive_read_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-player-style", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stable", action="store_true", help="load an independently imported stable checkpoint")
    parser.add_argument("--onnx", type=Path, help="evaluate the provenance-matched float32 CPU graph")
    parser.add_argument("--backend", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    report = evaluate(args)
    if args.output.exists():
        raise ValueError(f"GuitarSet OOD report already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
