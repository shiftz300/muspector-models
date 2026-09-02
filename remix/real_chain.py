"""Validate frozen real single-stage inverses under both chain orders."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .oracle import gates
from .real_pairs import chain
from .real_wiener import restore
from .stages import CORPUS, metric, sha256


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "cycles/stage16.json"
DRIVE = ROOT / "runs/stage/model15/drive.npz"
REVERB = ROOT / "runs/stage/model14/reverb.npz"
RUN = ROOT / "runs/stage/model16"
ORDERS = (("drive", "reverb"), ("reverb", "drive"))


def load(path: Path) -> tuple[np.ndarray, float, int]:
    with np.load(path, allow_pickle=False) as payload:
        if int(payload["schema"]) != 1 or int(payload["sample_rate"]) != 48000:
            raise ValueError(f"unsupported real inverse: {path}")
        return payload["response"].astype(np.complex64), float(payload["strength"]), int(payload["frames"])


def run(source: np.ndarray, order: tuple[str, ...], bank: dict[str, tuple[np.ndarray, float, int]]) -> np.ndarray:
    if order not in ORDERS:
        raise ValueError(f"unsupported real chain order: {order}")
    value = np.asarray(source, dtype=np.float32)
    for kind in reversed(order):
        response, strength, frames = bank[kind]
        if len(value) != frames:
            raise ValueError("real chain frame geometry differs")
        value = restore(value, response, strength)
    return value


def report(rows: list[dict], predictions: list[np.ndarray]) -> dict:
    value = metric([row["source"] for row in rows], predictions, [row["target"] for row in rows])
    value["groups"] = len({row["group"] for row in rows})
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, default=CYCLE)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--drive", type=Path, default=DRIVE)
    parser.add_argument("--reverb", type=Path, default=REVERB)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--frames", type=int, default=32768)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace real chain run {args.output}")
    cycle_bytes = args.cycle.read_bytes()
    cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "stage" or cycle.get("round") != 16 or cycle.get("status") != "planned":
        raise ValueError("invalid stage16 cycle")
    for name, path in (("drive", args.drive), ("reverb", args.reverb)):
        if sha256(path) != cycle["baseline"][name + "_sha256"]:
            raise ValueError(f"{name} baseline changed")
    args.output.mkdir(parents=True)
    bank = {"drive": load(args.drive), "reverb": load(args.reverb)}
    if any(value[2] != args.frames for value in bank.values()):
        raise ValueError("real model package frame geometry differs")
    rows = chain(args.corpus, "valid", args.frames, 20261230)
    start = time.perf_counter()
    predictions = [run(row["source"], row["order"], bank) for row in rows]
    seconds = time.perf_counter() - start
    reports = {"combined": report(rows, predictions)}
    for order in ORDERS:
        indices = [index for index, row in enumerate(rows) if row["order"] == order]
        reports["-".join(order)] = report([rows[index] for index in indices], [predictions[index] for index in indices])
    checks = {name: gates(value) for name, value in reports.items()}
    accepted = all(all(values.values()) for values in checks.values())
    if accepted:
        shutil.copy2(args.drive, args.output / "drive.npz")
        shutil.copy2(args.reverb, args.output / "reverb.npz")
    benchmark = {"buffers": len(rows), "frames": args.frames, "wall_seconds": seconds, "audio_seconds": len(rows) * args.frames / 48000, "real_time_factor": seconds / (len(rows) * args.frames / 48000), "device": "cpu", "stages_per_buffer": 2}
    result = {"schema": 1, "status": "accepted-real-chain-development" if accepted else "rejected-real-chain", "accepted": accepted, "reports": reports, "gates": checks, "benchmark": benchmark, "artifacts": {name: {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for name, path in ({"drive": args.output / "drive.npz", "reverb": args.output / "reverb.npz"}.items() if accepted else [])}, "executor": cycle["executor"], "limitations": ["known forward order used to isolate restoration quality", "fixed 32768-frame development geometry; overlap runtime remains to be sealed", "Telecaster test remains locked"], "data": cycle["data"], "quality": cycle["quality"], "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest()}
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if accepted:
        model = {"schema": 1, "id": "clean", "version": "0.7.0-real", "status": "accepted-real-chain-development", "registry": {"drive": {"processor": "wiener", "path": "drive.npz"}, "reverb": {"processor": "wiener", "path": "reverb.npz"}}, "executor": cycle["executor"], "frame": {"sample_rate": 48000, "frames": args.frames}, "quality": cycle["quality"]}
        (args.output / "model.json").write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": accepted, "reports": reports, "gates": checks, "benchmark": benchmark, "output": str(args.output)}), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
