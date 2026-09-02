"""Train the Stage-5 time-domain Drive and spectral Reverb oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .oracle import BASELINE, examples, evaluate, gates, loss, model as spectral, train
from .stages import CORPUS, sha256, state
from .tnet import TimeNet
from .forward_chain import ForwardChainRuntime


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "cycles/stage5.json"
RUN = ROOT / "runs/stage/model5"


def save(net: torch.nn.Module, path: Path, kind: str, initialization: str) -> None:
    payload = {
        "schema": 1,
        "sample_rate": 48_000,
        "stage": kind,
        "target": "immediate-predecessor",
        "controls": [f"{kind}.{name}" for name in (("gain_db", "tone", "level_db") if kind == "drive" else ("decay_s", "damping", "mix"))],
        "state_dict": state(net),
        "initialization": initialization,
    }
    if isinstance(net, TimeNet):
        payload.update({"architecture": "conditional-time-tcn", "channels": net.channels, "dilations": list(net.dilations)})
    else:
        payload.update({"architecture": "conditional-complex-stft", "channels": net.channels, "n_fft": net.n_fft, "hop": net.hop, "dilations": [list(value) for value in net.dilations]})
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, default=CYCLE); parser.add_argument("--corpus", type=Path, default=CORPUS); parser.add_argument("--baseline", type=Path, default=BASELINE); parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--drive-train", type=int, default=800); parser.add_argument("--reverb-train", type=int, default=640); parser.add_argument("--calibrate", type=int, default=128); parser.add_argument("--valid", type=int, default=160); parser.add_argument("--drive-frames", type=int, default=8192); parser.add_argument("--reverb-frames", type=int, default=16384); parser.add_argument("--drive-epochs", type=int, default=24); parser.add_argument("--reverb-epochs", type=int, default=40); parser.add_argument("--batch", type=int, default=8); parser.add_argument("--drive-rate", type=float, default=1e-4); parser.add_argument("--reverb-rate", type=float, default=3e-4); parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace hybrid run {args.output}")
    cycle_bytes = args.cycle.read_bytes(); cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "stage" or cycle.get("round") != 5 or cycle.get("status") != "planned":
        raise ValueError("invalid stage5 cycle")
    if sha256(args.baseline / "reverb.pt") != cycle["baseline"]["reverb_sha256"]:
        raise ValueError("Reverb baseline changed")
    args.output.mkdir(parents=True)
    torch.manual_seed(20261020); np.random.seed(20261020); torch.set_num_threads(args.threads)
    target = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    runtime = ForwardChainRuntime(); candidates, histories, reports, checks = {}, {}, {}, {}
    settings = {
        "drive": (TimeNet(20).to(target), args.drive_train, args.drive_frames, args.drive_epochs, args.drive_rate, "scratch"),
        "reverb": (spectral("reverb", args.baseline).to(target), args.reverb_train, args.reverb_frames, args.reverb_epochs, args.reverb_rate, sha256(args.baseline / "reverb.pt")),
    }
    for offset, kind in enumerate(("drive", "reverb")):
        net, count, frames, epochs, rate, initialization = settings[kind]
        fit = examples(runtime, args.corpus, kind, "train", count, frames, 20261040 + offset)
        calibration = examples(runtime, args.corpus, kind, "calibrate", args.calibrate, frames, 20261050 + offset)
        candidate, history = train(kind, net, fit, calibration, target, epochs, args.batch, rate)
        valid = examples(runtime, args.corpus, kind, "valid", args.valid, frames * 2, 20261060 + offset)
        report = evaluate(candidate, valid, target); check = gates(report)
        candidates[kind], histories[kind], reports[kind], checks[kind] = candidate.cpu(), history, report, check
        (args.output / f"{kind}.history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
        save(candidates[kind], args.output / f"{kind}.candidate.pt", kind, initialization)
        print(json.dumps({"stage": kind, "valid": report, "gates": check}), flush=True)
    accepted = all(all(value.values()) for value in checks.values())
    if accepted:
        for kind in candidates:
            (args.output / f"{kind}.candidate.pt").rename(args.output / f"{kind}.pt")
    result = {
        "schema": 1, "status": "accepted-oracle" if accepted else "rejected-oracle", "accepted": accepted,
        "artifacts": {kind: {"path": f"{kind}.{'pt' if accepted else 'candidate.pt'}", "sha256": sha256(args.output / f"{kind}.{'pt' if accepted else 'candidate.pt'}")} for kind in candidates},
        "reports": reports, "gates": checks,
        "training": {"drive": "conditional-time-tcn", "reverb": "conditional-complex-stft", "shared_optimizer": False, "chain_gradient": False},
        "data": {"source_read_only": True, "physical_audio_devices_used": False, "rendered_audio_retained": False},
        "quality": cycle["quality"], "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest(),
    }
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": accepted, "output": str(args.output)}), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
