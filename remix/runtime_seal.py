"""Validate the arbitrary-length real inverse runtime on long phrases."""

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
from .real_runtime import Runtime
from .stages import CORPUS, metric, sha256


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "cycles/stage19.json"
DRIVE = ROOT / "runs/stage/model18/drive.npz"
REVERB = ROOT / "runs/stage/model18/reverb.npz"
RUN = ROOT / "runs/stage/model19"
ORDERS = (("drive", "reverb"), ("reverb", "drive"))


def report(rows: list[dict], predictions: list[np.ndarray]) -> dict:
    value = metric([row["source"] for row in rows], predictions, [row["target"] for row in rows])
    value["groups"] = len({row["group"] for row in rows})
    value["frames"] = len(rows[0]["source"])
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, default=CYCLE)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--drive", type=Path, default=DRIVE)
    parser.add_argument("--reverb", type=Path, default=REVERB)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--frames", type=int, default=131072)
    parser.add_argument("--examples", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace runtime seal {args.output}")
    cycle_bytes = args.cycle.read_bytes()
    cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "stage" or cycle.get("round") != 19 or cycle.get("status") != "planned":
        raise ValueError("invalid stage19 cycle")
    for name, path in (("drive", args.drive), ("reverb", args.reverb)):
        if sha256(path) != cycle["baseline"][name + "_sha256"]:
            raise ValueError(f"{name} baseline changed")
    args.output.mkdir(parents=True)
    runtime = Runtime(args.drive, args.reverb, cycle["runtime"]["frame"], cycle["runtime"]["hop"])
    rows = chain(args.corpus, "valid", args.frames, 20261260, args.examples)
    hashes = [hashlib.sha256(row["source"].tobytes()).hexdigest() for row in rows]
    start = time.perf_counter()
    predictions = [runtime.run(row["source"], row["order"]) for row in rows]
    seconds = time.perf_counter() - start
    reports = {"combined": report(rows, predictions)}
    for order in ORDERS:
        indices = [index for index, row in enumerate(rows) if row["order"] == order]
        reports["-".join(order)] = report([rows[index] for index in indices], [predictions[index] for index in indices])
    checks = {name: gates(value) for name, value in reports.items()}
    preservation = {"frames": all(prediction.shape == row["source"].shape for row, prediction in zip(rows, predictions, strict=True)), "finite": all(np.isfinite(value).all() for value in predictions), "source_unchanged": hashes == [hashlib.sha256(row["source"].tobytes()).hexdigest() for row in rows], "bypass_exact": all(np.array_equal(runtime.run(row["source"], ()), row["source"]) for row in rows[:8])}
    accepted = all(all(values.values()) for values in checks.values()) and all(preservation.values())
    if accepted:
        shutil.copy2(args.drive, args.output / "drive.npz")
        shutil.copy2(args.reverb, args.output / "reverb.npz")
    benchmark = {"buffers": len(rows), "frames": args.frames, "wall_seconds": seconds, "audio_seconds": len(rows) * args.frames / 48000, "real_time_factor": seconds / (len(rows) * args.frames / 48000), "device": "cpu", "stages_per_buffer": 2}
    result = {"schema": 1, "status": "accepted-real-runtime-development" if accepted else "rejected-real-runtime", "accepted": accepted, "reports": reports, "gates": checks, "preservation": preservation, "benchmark": benchmark, "artifacts": {name: {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for name, path in ({"drive": args.output / "drive.npz", "reverb": args.output / "reverb.npz"}.items() if accepted else [])}, "runtime": cycle["runtime"], "limitations": ["known forward order used; wet-only order recognition is not yet sealed", "Telecaster test remains locked"], "data": cycle["data"], "quality": cycle["quality"], "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest()}
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if accepted:
        model = {"schema": 1, "id": "clean", "version": "0.9.0-real-runtime", "status": "accepted-real-runtime-development", "registry": {"drive": {"processor": "wiener-waveshaper", "path": "drive.npz"}, "reverb": {"processor": "wiener", "path": "reverb.npz"}}, "executor": {"policy": "reverse forward order", "fixed_stage_order": False}, "runtime": cycle["runtime"], "quality": cycle["quality"]}
        (args.output / "model.json").write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": accepted, "reports": reports, "gates": checks, "preservation": preservation, "benchmark": benchmark, "output": str(args.output)}), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
