"""Validate real chain quality and correct-vs-wrong order sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .oracle import gates
from .real_drive_nl import restore as drive_restore
from .real_pairs import chain
from .real_wiener import restore as linear_restore
from .stages import CORPUS, metric, sha256


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "cycles/stage18.json"
DRIVE = ROOT / "runs/stage/model17/drive.npz"
REVERB = ROOT / "runs/stage/model14/reverb.npz"
RUN = ROOT / "runs/stage/model18"
ORDERS = (("drive", "reverb"), ("reverb", "drive"))


def load_drive(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        if int(payload["schema"]) != 2 or int(payload["sample_rate"]) != 48000:
            raise ValueError("unsupported nonlinear Drive inverse")
        return {name: payload[name].copy() for name in payload.files}


def load_reverb(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        if int(payload["schema"]) != 1 or int(payload["sample_rate"]) != 48000:
            raise ValueError("unsupported real Reverb inverse")
        return {name: payload[name].copy() for name in payload.files}


def stage(source: np.ndarray, kind: str, drive: dict, reverb: dict) -> np.ndarray:
    if kind == "drive":
        return drive_restore(source, drive["response"], float(drive["linear_strength"]), drive["nonlinear_coefficients"], float(drive["nonlinear_strength"]))
    if kind == "reverb":
        return linear_restore(source, reverb["response"], float(reverb["strength"]))
    raise ValueError(f"unsupported real stage: {kind}")


def run(source: np.ndarray, stages: tuple[str, ...], drive: dict, reverb: dict) -> np.ndarray:
    if stages not in ORDERS:
        raise ValueError(f"unsupported stage sequence: {stages}")
    value = np.asarray(source, dtype=np.float32)
    for kind in stages:
        value = stage(value, kind, drive, reverb)
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
        raise FileExistsError(f"refusing to replace nonlinear real chain run {args.output}")
    cycle_bytes = args.cycle.read_bytes()
    cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "stage" or cycle.get("round") != 18 or cycle.get("status") != "planned":
        raise ValueError("invalid stage18 cycle")
    for name, path in (("drive", args.drive), ("reverb", args.reverb)):
        if sha256(path) != cycle["baseline"][name + "_sha256"]:
            raise ValueError(f"{name} baseline changed")
    args.output.mkdir(parents=True)
    drive, reverb = load_drive(args.drive), load_reverb(args.reverb)
    if int(drive["frames"]) != args.frames or int(reverb["frames"]) != args.frames:
        raise ValueError("nonlinear real chain geometry differs")
    rows = chain(args.corpus, "valid", args.frames, 20261250)
    start = time.perf_counter()
    correct = [run(row["source"], tuple(reversed(row["order"])), drive, reverb) for row in rows]
    wrong = [run(row["source"], row["order"], drive, reverb) for row in rows]
    seconds = time.perf_counter() - start
    reports = {"combined": report(rows, correct), "wrong_order": report(rows, wrong)}
    for order in ORDERS:
        indices = [index for index, row in enumerate(rows) if row["order"] == order]
        reports["-".join(order)] = report([rows[index] for index in indices], [correct[index] for index in indices])
    quality = {name: gates(reports[name]) for name in ("combined", "drive-reverb", "reverb-drive")}
    ratios = [float(np.linalg.norm(left - right) / max(np.linalg.norm(row["target"] - row["source"]), 1.0e-12)) for row, left, right in zip(rows, correct, wrong, strict=True)]
    order = {"mean_delta_ratio": float(np.mean(ratios)), "median_delta_ratio": float(np.median(ratios)), "fraction_above_0_05": float(np.mean(np.asarray(ratios) >= 0.05)), "correct_aligned_esr_better": reports["combined"]["aligned_esr_improvement"] > reports["wrong_order"]["aligned_esr_improvement"], "correct_sidr_better": reports["combined"]["sidr_improvement_db"] > reports["wrong_order"]["sidr_improvement_db"]}
    order_checks = {"mean_delta": order["mean_delta_ratio"] >= cycle["order_gates"]["mean_delta_ratio_minimum"], "median_delta": order["median_delta_ratio"] >= cycle["order_gates"]["median_delta_ratio_minimum"], "aligned_win": order["correct_aligned_esr_better"], "sidr_win": order["correct_sidr_better"]}
    accepted = all(all(values.values()) for values in quality.values()) and all(order_checks.values())
    if accepted:
        shutil.copy2(args.drive, args.output / "drive.npz")
        shutil.copy2(args.reverb, args.output / "reverb.npz")
    benchmark = {"buffers": len(rows), "frames": args.frames, "passes_per_buffer": 4, "wall_seconds": seconds, "audio_seconds": len(rows) * args.frames / 48000, "real_time_factor": seconds / (len(rows) * args.frames / 48000), "device": "cpu"}
    result = {"schema": 1, "status": "accepted-real-order-development" if accepted else "rejected-real-order", "accepted": accepted, "reports": reports, "quality_gates": quality, "order": order, "order_gates": order_checks, "benchmark": benchmark, "artifacts": {name: {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for name, path in ({"drive": args.output / "drive.npz", "reverb": args.output / "reverb.npz"}.items() if accepted else [])}, "executor": cycle["executor"], "limitations": ["known forward order used; wet-only order recognition is not yet sealed", "order difference is material but modest and must not be marketed as a dramatic effect", "fixed 32768-frame development geometry; overlap runtime remains to be sealed", "Telecaster test remains locked"], "data": cycle["data"], "quality": cycle["quality"], "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest()}
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if accepted:
        model = {"schema": 1, "id": "clean", "version": "0.8.0-real-order", "status": "accepted-real-order-development", "registry": {"drive": {"processor": "wiener-waveshaper", "path": "drive.npz"}, "reverb": {"processor": "wiener", "path": "reverb.npz"}}, "executor": cycle["executor"], "frame": {"sample_rate": 48000, "frames": args.frames}, "quality": cycle["quality"]}
        (args.output / "model.json").write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": accepted, "reports": reports, "quality_gates": quality, "order": order, "order_gates": order_checks, "benchmark": benchmark, "output": str(args.output)}), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
