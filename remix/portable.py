"""Export and verify the portable ONNX core for wet-to-clean restoration.

STFT, scale prepass, chunking, overlap context and iSTFT stay native. The ONNX
graph contains only the learned convolutional core and never touches a physical
audio device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from .net import SpectralNet
from .runtime import CleanRuntime, LIMITS, RATE, _audio


class Core(nn.Module):
    def __init__(self, model: SpectralNet):
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model.core(features)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw / (1024.0 * 1024.0) if platform.system() == "Darwin" else raw / 1024.0


def session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


class PortableRuntime:
    """Bounded Python contract reference for native-DSP plus ONNX inference."""

    def __init__(self, checkpoint: Path, graph: Path, *, core: int, halo: int, threads: int):
        self.reference = CleanRuntime(checkpoint, core=core, halo=halo, threads=threads)
        self.model = self.reference.model
        self.core_frames, self.halo, self.threads = core, halo, threads
        self.ort = session(graph, threads)

    def _piece(self, audio: np.ndarray, scale: torch.Tensor, strength: float) -> np.ndarray:
        wet = torch.from_numpy(audio.copy()).unsqueeze(0)
        spectrum = torch.stft(
            wet, self.model.n_fft, self.model.hop,
            window=self.model.window, return_complex=True,
        )
        features = torch.stack((
            spectrum.real / scale,
            spectrum.imag / scale,
            torch.log1p(spectrum.abs() / scale),
        ), 1).numpy()
        residual = self.ort.run(("residual",), {"features": features})[0]
        estimate = spectrum + torch.complex(
            torch.from_numpy(residual[:, 0]), torch.from_numpy(residual[:, 1])
        ) * scale * float(strength)
        return torch.istft(
            estimate, self.model.n_fft, self.model.hop,
            window=self.model.window, length=len(audio),
        )[0].numpy().copy()

    def render(self, value: np.ndarray, strength: float) -> np.ndarray:
        audio = _audio(value)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be between zero and one")
        if strength == 0.0:
            return audio.copy()
        scale = self.reference.scale(audio)
        pieces = []
        for start in range(0, len(audio), self.core_frames):
            end = min(len(audio), start + self.core_frames)
            left, right = max(0, start - self.halo), min(len(audio), end + self.halo)
            restored = self._piece(audio[left:right], scale, strength)
            pieces.append(restored[start - left : end - left])
        result = np.concatenate(pieces)
        if result.shape != audio.shape or result.dtype != np.float32 or not np.isfinite(result).all():
            raise RuntimeError("portable runtime violated audio geometry")
        return result


def describe(graph: Path) -> dict:
    model = onnx.load(str(graph))
    onnx.checker.check_model(model)
    return {
        "artifact": str(graph),
        "artifact_sha256": sha256(graph),
        "artifact_bytes": graph.stat().st_size,
        "operators": sorted({node.op_type for node in model.graph.node}),
    }


def export(checkpoint: Path, graph: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = SpectralNet(
        payload["channels"], payload["n_fft"], payload["hop"],
        tuple(tuple(value) for value in payload.get("dilations", ((1, 1), (2, 1), (4, 2), (8, 4), (16, 8)))),
    )
    model.load_state_dict(payload["state_dict"])
    core = Core(model).eval()
    probe = torch.zeros((1, 3, model.n_fft // 2 + 1, 129), dtype=torch.float32)
    graph.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        core, (probe,), graph,
        input_names=("features",), output_names=("residual",),
        dynamic_axes={"features": {0: "batch", 3: "time"}, "residual": {0: "batch", 3: "time"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    return describe(graph)


def verify(checkpoint: Path, graph: Path, *, strength: float, seconds: int, core: int, halo: int, threads: int) -> dict:
    portable = PortableRuntime(checkpoint, graph, core=core, halo=halo, threads=threads)
    rng = np.random.default_rng(20260901)
    probes = []
    for frames in (4_097, 32_768, 72_123):
        value = (rng.standard_normal(frames) * 0.035).astype(np.float32)
        reference = portable.reference.render(value, strength)
        candidate = portable.render(value, strength)
        delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
        probes.append({"frames": frames, "maximum_absolute_error": float(delta.max()), "mean_absolute_error": float(delta.mean())})
    bypass_source = (rng.standard_normal(8_193) * 0.02).astype(np.float32)
    bypass = portable.render(bypass_source, 0.0)
    benchmark = (rng.standard_normal(seconds * RATE) * 0.035).astype(np.float32)
    portable.render(benchmark[:RATE], strength)
    started = time.perf_counter()
    output = portable.render(benchmark, strength)
    elapsed = time.perf_counter() - started
    measured = {
        "seconds": seconds, "elapsed_seconds": elapsed, "realtime_factor": elapsed / seconds,
        "process_peak_rss_mb": rss_mb(), "parameters": portable.reference.parameters,
        "graph_bytes": graph.stat().st_size,
        "maximum_absolute_error": max(row["maximum_absolute_error"] for row in probes),
        "mean_absolute_error": float(np.mean([row["mean_absolute_error"] for row in probes])),
        "bypass_bit_exact": bool(np.array_equal(bypass, bypass_source)),
        "finite": bool(np.isfinite(output).all()), "geometry_preserved": bool(output.shape == benchmark.shape),
        "cpu_threads": threads, "maximum_model_input_frames": core + 2 * halo,
    }
    checks = {
        "artifact": measured["graph_bytes"] <= LIMITS["artifact_bytes"],
        "parameters": measured["parameters"] <= LIMITS["parameters"],
        "parity": measured["maximum_absolute_error"] <= 2.0e-6,
        "speed": measured["realtime_factor"] <= LIMITS["realtime_factor"],
        "memory": measured["process_peak_rss_mb"] <= LIMITS["process_rss_mb"],
        "threads": threads <= LIMITS["cpu_threads"], "bypass": measured["bypass_bit_exact"],
        "finite": measured["finite"], "geometry": measured["geometry_preserved"],
    }
    return {"accepted": all(checks.values()), "measured": measured, "checks": checks, "probes": probes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/clean/model11/clean.pt"))
    parser.add_argument("--cycle", type=Path, default=Path("cycles/clean13.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/clean/model11/portable.json"))
    parser.add_argument("--graph", type=Path, default=Path("runs/clean/model11/core.onnx"))
    parser.add_argument("--strength", type=float, default=0.875)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--core", type=int, default=16_384)
    parser.add_argument("--halo", type=int, default=4_096)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--reuse", action="store_true", help="verify an already exported graph")
    args = parser.parse_args()
    if args.output.exists() or (args.graph.exists() and not args.reuse):
        raise FileExistsError("refusing to replace portable artifacts")
    if args.reuse and not args.graph.exists():
        raise FileNotFoundError("portable graph does not exist")
    cycle_bytes = args.cycle.read_bytes()
    cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "clean1" or cycle.get("round") != 13 or cycle.get("status") != "planned":
        raise ValueError("invalid clean13 cycle")
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != cycle["model"]["checkpoint_sha256"]:
        raise ValueError("portable source checkpoint changed")
    artifact = describe(args.graph) if args.reuse else export(args.checkpoint, args.graph)
    result = verify(args.checkpoint, args.graph, strength=args.strength, seconds=args.seconds, core=args.core, halo=args.halo, threads=args.threads)
    report = {
        "schema": 1, "status": "accepted-portable-development" if result["accepted"] else "rejected",
        "accepted": result["accepted"], "checkpoint_sha256": checkpoint_hash,
        "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest(), "artifact": artifact,
        "contract": {
            "sample_rate": RATE, "n_fft": 512, "hop": 128, "window": "periodic-hann",
            "center": True, "padding": "reflect",
            "scale": "whole-clip mean STFT magnitude prepass, clamp minimum 1e-6",
            "features": ["real/scale", "imag/scale", "log1p(magnitude/scale)"],
            "output": "complex residual times scale times strength, then native iSTFT",
            "strength": args.strength, "core_frames": args.core, "halo_frames": args.halo,
            "execution": "background-offline", "full_clip_model_inference_forbidden": True,
        },
        **result,
        "quality": {
            "source_audio_modified": False, "physical_audio_devices_used": False,
            "rendered_audio_retained": False, "automatic_normalization": False,
            "automatic_limiting": False, "automatic_dither": False, "lossy_reencoding": False,
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": report["accepted"], "artifact": artifact, "measured": result["measured"], "checks": result["checks"]}, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
