"""Fit a tiny nonlinear residual on the accepted real Drive inverse."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .oracle import gates
from .real_pairs import single
from .real_wiener import restore as linear
from .refine import rank
from .stages import CORPUS, metric, sha256


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "cycles/stage17.json"
BASELINE = ROOT / "runs/stage/model15/drive.npz"
RUN = ROOT / "runs/stage/model17"
SCALE = 0.25


def design(audio: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(audio, dtype=np.float64) / SCALE, -2.0, 2.0)
    return np.stack((value, value * np.abs(value), value**3, value * np.abs(value) ** 3, value**5), axis=-1)


def fit(rows: list[dict], response: np.ndarray, linear_strength: float, stride: int = 16) -> np.ndarray:
    gram = np.zeros((5, 5), dtype=np.float64)
    cross = np.zeros(5, dtype=np.float64)
    for index, row in enumerate(rows):
        base = linear(row["source"], response, linear_strength)
        take = np.arange(index % stride, len(base), stride)
        features = design(base[take])
        target = row["target"][take].astype(np.float64) / SCALE
        gram += features.T @ features
        cross += features.T @ target
    ridge = np.diag((1.0e-3, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2))
    return np.linalg.solve(gram + ridge, cross)


def restore(source: np.ndarray, response: np.ndarray, linear_strength: float, coefficients: np.ndarray, nonlinear_strength: float) -> np.ndarray:
    base = linear(source, response, linear_strength)
    nonlinear = np.asarray(SCALE * (design(base) @ coefficients), dtype=np.float32)
    result = np.asarray(base + float(nonlinear_strength) * (nonlinear - base), dtype=np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonlinear Drive inverse produced non-finite audio")
    return result


def evaluate(rows: list[dict], response: np.ndarray, linear_strength: float, coefficients: np.ndarray, nonlinear_strength: float) -> dict:
    predictions = [restore(row["source"], response, linear_strength, coefficients, nonlinear_strength) for row in rows]
    report = metric([row["source"] for row in rows], predictions, [row["target"] for row in rows])
    report["groups"] = len({row["group"] for row in rows})
    report["nonlinear_strength"] = float(nonlinear_strength)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, default=CYCLE)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--frames", type=int, default=32768)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace nonlinear Drive run {args.output}")
    cycle_bytes = args.cycle.read_bytes()
    cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "stage" or cycle.get("round") != 17 or cycle.get("status") != "planned":
        raise ValueError("invalid stage17 cycle")
    if sha256(args.baseline) != cycle["baseline"]["drive_sha256"]:
        raise ValueError("Drive baseline changed")
    with np.load(args.baseline, allow_pickle=False) as payload:
        response = payload["response"].astype(np.complex64)
        linear_strength = float(payload["strength"])
        if int(payload["frames"]) != args.frames or int(payload["sample_rate"]) != 48000:
            raise ValueError("Drive baseline geometry differs")
    args.output.mkdir(parents=True)
    fit_rows = single(args.corpus, "train", "drive", args.frames, 20261240)
    coefficients = fit(fit_rows, response, linear_strength)
    calibration = single(args.corpus, "calibrate", "drive", args.frames, 20261241)
    candidates = []
    for strength in cycle["calibration"]["nonlinear_strength"]:
        report = evaluate(calibration, response, linear_strength, coefficients, float(strength))
        candidate = {"strength": strength, "rank": rank(report), "report": report, "gates": gates(report)}
        candidates.append(candidate)
        print(json.dumps(candidate), flush=True)
    selected = max(candidates, key=lambda row: row["rank"])
    valid = single(args.corpus, "valid", "drive", args.frames, 20261242)
    report = evaluate(valid, response, linear_strength, coefficients, float(selected["strength"]))
    checks = gates(report)
    accepted = float(selected["strength"]) > 0.0 and all(checks.values())
    artifact = args.output / ("drive.npz" if accepted else "drive.candidate.npz")
    np.savez_compressed(artifact, schema=np.asarray(2, dtype=np.int64), sample_rate=np.asarray(48000, dtype=np.int64), frames=np.asarray(args.frames, dtype=np.int64), response=response, linear_strength=np.asarray(linear_strength), nonlinear_scale=np.asarray(SCALE), nonlinear_coefficients=np.asarray(coefficients, dtype=np.float64), nonlinear_strength=np.asarray(selected["strength"], dtype=np.float64))
    start = time.perf_counter()
    for row in valid[:32]:
        restore(row["source"], response, linear_strength, coefficients, float(selected["strength"]))
    seconds = time.perf_counter() - start
    benchmark = {"buffers": 32, "frames": args.frames, "wall_seconds": seconds, "audio_seconds": 32 * args.frames / 48000, "real_time_factor": seconds / (32 * args.frames / 48000), "device": "cpu"}
    result = {"schema": 1, "status": "accepted-real-nonlinear-development" if accepted else "rejected-real-nonlinear", "accepted": accepted, "selection": {"candidates": candidates, "selected_strength": selected["strength"]}, "coefficients": coefficients.tolist(), "report": report, "gates": {**checks, "nonlinear": float(selected["strength"]) > 0.0}, "benchmark": benchmark, "artifact": {"path": artifact.name, "sha256": sha256(artifact), "bytes": artifact.stat().st_size}, "training": cycle["training"], "limitations": ["single real Drive family only", "fixed 32768-frame development geometry; overlap runtime remains to be sealed", "Telecaster test remains locked"], "data": cycle["data"], "quality": cycle["quality"], "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest()}
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": accepted, "selected": selected, "valid": report, "gates": checks, "benchmark": benchmark, "output": str(args.output)}), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
