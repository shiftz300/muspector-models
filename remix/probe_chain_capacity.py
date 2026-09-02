"""Probe a compact third Delay stage against the offline runtime budget."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .forward_chain import FORWARD_RATE
from .net import SpectralNet
from .portable import PortableRuntime, export, rss_mb


DELAY_DILATIONS = ((1, 1), (2, 4), (4, 16), (8, 64))


def save_probe(path: Path, channels: int) -> None:
    model = SpectralNet(channels, 512, 128, DELAY_DILATIONS)
    torch.save({
        "schema": 1,
        "architecture": "complex-stft",
        "sample_rate": FORWARD_RATE,
        "channels": model.channels,
        "n_fft": model.n_fft,
        "hop": model.hop,
        "dilations": [list(value) for value in model.dilations],
        "kind": "delay-inverse-capacity-probe",
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", type=Path, default=Path("runs/chain/model7/drive.pt"))
    parser.add_argument("--reverb", type=Path, default=Path("runs/chain/model7/reverb.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/chain/model7/capacity.json"))
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--channels", type=int, default=8)
    args = parser.parse_args()
    rng = np.random.default_rng(20260921)
    source = (rng.standard_normal(args.seconds * FORWARD_RATE) * 0.035).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="muspector-delay-probe-") as temporary:
        root = Path(temporary)
        delay_path, delay_graph = root / "delay.pt", root / "delay.onnx"
        save_probe(delay_path, args.channels)
        drive_graph, reverb_graph = root / "drive.onnx", root / "reverb.onnx"
        drive_artifact = export(args.drive, drive_graph)
        reverb_artifact = export(args.reverb, reverb_graph)
        delay_artifact = export(delay_path, delay_graph)
        drive = PortableRuntime(args.drive, drive_graph, core=16_384, halo=4_096, threads=2)
        delay = PortableRuntime(delay_path, delay_graph, core=16_384, halo=12_288, threads=2)
        reverb = PortableRuntime(args.reverb, reverb_graph, core=16_384, halo=12_288, threads=2)
        drive.render(source[:FORWARD_RATE], 1.0)
        delay.render(source[:FORWARD_RATE], 1.0)
        reverb.render(source[:FORWARD_RATE], 1.0)
        started = time.perf_counter()
        value = reverb.render(delay.render(drive.render(source, 1.0), 1.0), 1.0)
        elapsed = time.perf_counter() - started
        measured = {
            "seconds": args.seconds,
            "elapsed_seconds": elapsed,
            "realtime_factor": elapsed / args.seconds,
            "process_peak_rss_mb": rss_mb(),
            "parameters": drive.reference.parameters + delay.reference.parameters + reverb.reference.parameters,
            "artifact_bytes": drive_graph.stat().st_size + delay_graph.stat().st_size + reverb_graph.stat().st_size,
            "delay_channels": args.channels,
            "delay_minimum_halo_frames": SpectralNet(args.channels, 512, 128, DELAY_DILATIONS).minimum_halo,
            "cpu_threads": 2,
            "finite": bool(np.isfinite(value).all()),
            "geometry_preserved": value.shape == source.shape,
        }
        checks = {
            "speed": measured["realtime_factor"] <= 0.5,
            "memory": measured["process_peak_rss_mb"] <= 512,
            "parameters": measured["parameters"] <= 250_000,
            "artifact": measured["artifact_bytes"] <= 8 * 1024 * 1024,
            "finite": measured["finite"],
            "geometry": measured["geometry_preserved"],
        }
        result = {
            "schema": 1,
            "accepted": all(checks.values()),
            "candidate": "8-channel Delay spectral stage with 12,288-frame halo",
            "measured": measured,
            "checks": checks,
            "artifacts": [drive_artifact, delay_artifact, reverb_artifact],
            "quality": {"source_audio_modified": False, "physical_audio_devices_used": False},
            "limitation": "Untrained capacity probe only; no fidelity or Delay inversion claim.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
