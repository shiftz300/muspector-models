"""One-shot locked Telecaster seal for the real arbitrary-length runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .data import order_manifest
from .oracle import gates
from .real_pairs import chain
from .real_runtime import Runtime
from .stages import CORPUS, metric, sha256


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "cycles/stage20.json"
DRIVE = ROOT / "runs/stage/model19/drive.npz"
REVERB = ROOT / "runs/stage/model19/reverb.npz"
RUNTIME_REPORT = ROOT / "runs/stage/model19/valid.json"
RUN = ROOT / "runs/stage/model20"
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
    parser.add_argument("--runtime-report", type=Path, default=RUNTIME_REPORT)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--frames", type=int, default=131072)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace locked final seal {args.output}")
    cycle_bytes = args.cycle.read_bytes()
    cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "stage" or cycle.get("round") != 20 or cycle.get("status") != "planned-locked-final":
        raise ValueError("invalid or already-opened stage20 cycle")
    for name, path in (("drive", args.drive), ("reverb", args.reverb)):
        if sha256(path) != cycle["baseline"][name + "_sha256"]:
            raise ValueError(f"{name} baseline changed")
    if sha256(args.runtime_report) != cycle["baseline"]["runtime_report_sha256"]:
        raise ValueError("runtime development report changed")
    args.output.mkdir(parents=True)
    runtime = Runtime(args.drive, args.reverb, cycle["runtime"]["frame"], cycle["runtime"]["hop"])
    rows = chain(args.corpus, "test", args.frames, 20261270)
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
    accepted = len(rows) == 400 and all(all(values.values()) for values in checks.values()) and all(preservation.values())
    if accepted:
        shutil.copy2(args.drive, args.output / "drive.npz")
        shutil.copy2(args.reverb, args.output / "reverb.npz")
    benchmark = {"buffers": len(rows), "frames": args.frames, "wall_seconds": seconds, "audio_seconds": len(rows) * args.frames / 48000, "real_time_factor": seconds / (len(rows) * args.frames / 48000), "device": "cpu", "stages_per_buffer": 2}
    manifest = order_manifest(args.corpus)
    result = {"schema": 1, "status": "accepted-real-locked-final" if accepted else "rejected-real-locked-final", "accepted": accepted, "reports": reports, "gates": checks, "preservation": preservation, "benchmark": benchmark, "artifacts": {name: {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for name, path in ({"drive": args.output / "drive.npz", "reverb": args.output / "reverb.npz"}.items() if accepted else [])}, "dataset": {"records": len(rows), "split": "test", "guitar": "Telecaster", "metadata_sha256": manifest["metadata_sha256"], "source": manifest["source"], "license": manifest["license"], "opened_once_without_tuning": True}, "runtime": cycle["runtime"], "limitations": ["known forward order used; wet-only order recognition is not sealed", "sealed for the DAFx real Drive/Reverb domain, not arbitrary physical pedals", "clean-profile and absolute-level policy remain explicit client inputs"], "data": cycle["data"], "quality": cycle["quality"], "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest()}
    (args.output / "seal.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if accepted:
        model = {"schema": 1, "id": "clean", "version": "1.0.0", "status": "accepted-real-locked-final", "registry": {"drive": {"processor": "wiener-waveshaper", "path": "drive.npz"}, "reverb": {"processor": "wiener", "path": "reverb.npz"}}, "executor": {"policy": "reverse forward order", "fixed_stage_order": False, "order_input": "required"}, "runtime": cycle["runtime"], "quality": cycle["quality"]}
        (args.output / "model.json").write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": accepted, "reports": reports, "gates": checks, "preservation": preservation, "benchmark": benchmark, "output": str(args.output)}), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
