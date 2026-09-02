"""Bounded-memory CPU runtime for wet-to-clean development checkpoints.

This module reads arrays and files only. It never enumerates or opens an audio
device, and the restorer is intended for a background worker, not an audio
callback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .net import SpectralNet


RATE = 48_000
LIMITS = {
    "artifact_bytes": 8 * 1024 * 1024,
    "parameters": 250_000,
    "cpu_threads": 2,
    "realtime_factor": 0.50,
    "process_rss_mb": 512.0,
    "parity_max_absolute_error": 2.0e-5,
}


def _audio(value: np.ndarray) -> np.ndarray:
    audio = np.asarray(value)
    if audio.dtype != np.float32 or audio.ndim != 1 or len(audio) < 512:
        raise ValueError("runtime input must be float32 mono audio with at least 512 frames")
    if not np.isfinite(audio).all():
        raise ValueError("runtime input must be finite")
    return audio


def _rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw / (1024.0 * 1024.0) if platform.system() == "Darwin" else raw / 1024.0


class CleanRuntime:
    """Fixed-working-set inference with a read-only global scale prepass."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        core: int = 32_768,
        halo: int = 4_096,
        threads: int = 2,
    ):
        self.checkpoint = Path(checkpoint)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if payload.get("architecture") != "complex-stft" or payload.get("sample_rate") != RATE:
            raise ValueError("unsupported clean checkpoint")
        self.model = SpectralNet(
            payload["channels"], payload["n_fft"], payload["hop"],
            tuple(tuple(value) for value in payload.get("dilations", ((1, 1), (2, 1), (4, 2), (8, 4), (16, 8)))),
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.core, self.halo, self.threads = int(core), int(halo), int(threads)
        minimum_halo = self.model.minimum_halo
        if self.core <= 0 or self.core % self.model.hop:
            raise ValueError("core must be a positive multiple of the STFT hop")
        if self.halo < minimum_halo or self.halo % self.model.hop:
            raise ValueError("halo is too short or is not hop-aligned")
        if not 1 <= self.threads <= LIMITS["cpu_threads"]:
            raise ValueError("runtime thread budget exceeded")
        torch.set_num_threads(self.threads)
        self.parameters = sum(value.numel() for value in self.model.parameters())
        self.artifact_bytes = self.checkpoint.stat().st_size
        self.artifact_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()

    @torch.inference_mode()
    def scale(self, value: np.ndarray, block: int = 256) -> torch.Tensor:
        """Compute the model's exact whole-clip STFT scale with bounded frames."""

        audio = _audio(value)
        source = torch.from_numpy(audio)
        half, length = self.model.n_fft // 2, len(audio)
        offsets = torch.arange(-half, half, dtype=torch.int64)
        window = self.model.window.to(source.dtype)
        total, count = 0.0, 0
        frame_count = length // self.model.hop + 1
        period = 2 * (length - 1)
        for first in range(0, frame_count, block):
            centers = torch.arange(first, min(first + block, frame_count), dtype=torch.int64)
            indices = centers[:, None] * self.model.hop + offsets[None, :]
            folded = torch.remainder(indices, period)
            reflected = torch.where(folded < length, folded, period - folded)
            spectrum = torch.fft.rfft(source[reflected] * window, dim=1)
            total += float(spectrum.abs().sum())
            count += spectrum.numel()
        return torch.tensor(total / max(count, 1), dtype=torch.float32).reshape(1, 1, 1).clamp_min(1e-6)

    @torch.inference_mode()
    def chunks(self, value: np.ndarray, strength: float = 1.0) -> Iterator[np.ndarray]:
        """Yield restored core chunks; model activation memory is duration-independent."""

        audio = _audio(value)
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError("restoration strength must be between zero and one")
        if strength == 0.0:
            for start in range(0, len(audio), self.core):
                yield audio[start : start + self.core].copy()
            return
        scale = self.scale(audio)
        for start in range(0, len(audio), self.core):
            end = min(len(audio), start + self.core)
            left, right = max(0, start - self.halo), min(len(audio), end + self.halo)
            wet = torch.from_numpy(audio[left:right].copy()).unsqueeze(0)
            restored = self.model.scaled(wet, scale, strength)[0]
            yield restored[start - left : end - left].numpy().copy()

    def render(self, value: np.ndarray, strength: float = 1.0) -> np.ndarray:
        audio = _audio(value)
        before = hashlib.sha256(audio.tobytes()).hexdigest()
        restored = np.concatenate(tuple(self.chunks(audio, strength)))
        if restored.shape != audio.shape or restored.dtype != np.float32:
            raise RuntimeError("runtime changed audio geometry")
        if not np.isfinite(restored).all():
            raise RuntimeError("runtime produced non-finite audio")
        if hashlib.sha256(audio.tobytes()).hexdigest() != before:
            raise RuntimeError("runtime mutated source audio")
        return restored


def benchmark(runtime: CleanRuntime, seconds: int, seed: int, strength: float = 1.0) -> dict:
    if seconds < 10:
        raise ValueError("runtime benchmark must cover at least ten seconds")
    rng = np.random.default_rng(seed)
    frames = seconds * RATE
    timeline = np.arange(frames, dtype=np.float32) / RATE
    source = (
        0.035 * rng.standard_normal(frames)
        + 0.025 * np.sin(2.0 * math.pi * 110.0 * timeline)
        + 0.015 * np.sin(2.0 * math.pi * 329.63 * timeline)
    ).astype(np.float32)
    before = hashlib.sha256(source.tobytes()).hexdigest()
    runtime.render(source[:RATE], strength)
    started = time.perf_counter()
    rendered = runtime.render(source, strength)
    elapsed = time.perf_counter() - started
    realtime_factor = elapsed / seconds
    measured = {
        "artifact_bytes": runtime.artifact_bytes,
        "parameters": runtime.parameters,
        "cpu_threads": runtime.threads,
        "core_frames": runtime.core,
        "halo_frames": runtime.halo,
        "maximum_model_input_frames": runtime.core + 2 * runtime.halo,
        "seconds": seconds,
        "elapsed_seconds": elapsed,
        "realtime_factor": realtime_factor,
        "process_peak_rss_mb": _rss_mb(),
        "finite": bool(np.isfinite(rendered).all()),
        "geometry_preserved": bool(rendered.shape == source.shape),
        "source_audio_modified": hashlib.sha256(source.tobytes()).hexdigest() != before,
        "strength": strength,
        "output_sha256": hashlib.sha256(rendered.tobytes()).hexdigest(),
    }
    checks = {
        "artifact": measured["artifact_bytes"] <= LIMITS["artifact_bytes"],
        "parameters": measured["parameters"] <= LIMITS["parameters"],
        "threads": measured["cpu_threads"] <= LIMITS["cpu_threads"],
        "speed": measured["realtime_factor"] <= LIMITS["realtime_factor"],
        "memory": measured["process_peak_rss_mb"] <= LIMITS["process_rss_mb"],
        "finite": measured["finite"],
        "geometry": measured["geometry_preserved"],
        "immutable": not measured["source_audio_modified"],
    }
    return {
        "schema": 1,
        "status": "accepted-runtime-development" if all(checks.values()) else "rejected",
        "accepted": all(checks.values()),
        "backend": "pytorch-cpu-reference",
        "execution": "background-offline",
        "audio_thread_safe": False,
        "full_clip_model_inference_forbidden": True,
        "activation_memory_duration_independent": True,
        "physical_audio_devices_used": False,
        "rendered_audio_retained": False,
        "artifact_sha256": runtime.artifact_sha256,
        "limits": LIMITS,
        "measured": measured,
        "checks": checks,
    }


@torch.inference_mode()
def parity(runtime: CleanRuntime, corpus: Path, files: int, strength: float = 1.0) -> dict:
    """Compare bounded inference with the admitted whole-record reference."""

    from .asrnn_data import rat_files, read_rat_pair

    paths = rat_files(corpus, "eval")[:files]
    if not paths:
        raise ValueError("no RAT parity files")
    rows, mutations = [], 0
    for path in paths:
        _, wet, _ = read_rat_pair(path)
        before = hashlib.sha256(wet.tobytes()).hexdigest()
        reference = runtime.model(torch.from_numpy(wet).unsqueeze(0), strength=strength)[0].numpy()
        bounded = runtime.render(wet, strength)
        difference = np.abs(reference.astype(np.float64) - bounded.astype(np.float64))
        rows.append({
            "file": path.name,
            "maximum_absolute_error": float(np.max(difference)),
            "mean_absolute_error": float(np.mean(difference)),
        })
        mutations += int(hashlib.sha256(wet.tobytes()).hexdigest() != before)
    maximum = max(row["maximum_absolute_error"] for row in rows)
    passed = maximum <= LIMITS["parity_max_absolute_error"] and mutations == 0
    return {
        "schema": 1,
        "status": "accepted-runtime-parity" if passed else "rejected",
        "accepted": passed,
        "files": len(rows),
        "maximum_absolute_error": maximum,
        "mean_absolute_error": float(np.mean([row["mean_absolute_error"] for row in rows])),
        "gate": LIMITS["parity_max_absolute_error"],
        "source_mutations": mutations,
        "physical_audio_devices_used": False,
        "rendered_audio_retained": False,
        "artifact_sha256": runtime.artifact_sha256,
        "strength": strength,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/clean/model4/clean.pt"))
    parser.add_argument("--cycle", type=Path, default=Path("cycles/clean8.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/clean/model8/runtime.json"))
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--core", type=int, default=32_768)
    parser.add_argument("--halo", type=int, default=4_096)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/asrnn-physical-effects"))
    parser.add_argument("--files", type=int, default=128)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace runtime report {args.output}")
    cycle_bytes = args.cycle.read_bytes()
    cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "clean1" or cycle.get("round", 0) < 8 or cycle.get("status") != "planned":
        raise ValueError("invalid clean runtime cycle")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runtime = CleanRuntime(args.checkpoint, core=args.core, halo=args.halo, threads=args.threads)
    report = parity(runtime, args.corpus, args.files, args.strength) if args.parity else benchmark(runtime, args.seconds, args.seed, args.strength)
    report["cycle_sha256"] = hashlib.sha256(cycle_bytes).hexdigest()
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
