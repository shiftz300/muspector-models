"""Fit and validate a real-paired compact Reverb Wiener inverse."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .oracle import gates
from .real_pairs import single
from .refine import rank
from .stages import CORPUS, metric, sha256


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "cycles/stage14.json"
RUN = ROOT / "runs/stage/model14"


def fit(rows: list[dict], regularization: float) -> np.ndarray:
    frames = len(rows[0]["source"])
    cross = np.zeros(frames // 2 + 1, dtype=np.complex128)
    power = np.zeros(frames // 2 + 1, dtype=np.float64)
    for row in rows:
        source = np.fft.rfft(row["source"])
        target = np.fft.rfft(row["target"])
        cross += np.conj(source) * target
        power += np.square(np.abs(source))
    return np.asarray(cross / (power + float(regularization) * np.mean(power)), dtype=np.complex64)


def restore(source: np.ndarray, response: np.ndarray, strength: float) -> np.ndarray:
    value = np.asarray(source, dtype=np.float32)
    if value.ndim != 1 or len(response) != len(value) // 2 + 1:
        raise ValueError("real Wiener inverse geometry differs")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("real Wiener strength must be within [0,1]")
    estimate = np.fft.irfft(np.fft.rfft(value) * response, len(value)).astype(np.float32)
    result = np.asarray(value + float(strength) * (estimate - value), dtype=np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("real Wiener inverse produced non-finite audio")
    return result


def evaluate(rows: list[dict], response: np.ndarray, strength: float) -> dict:
    predictions = [restore(row["source"], response, strength) for row in rows]
    report = metric(
        [row["source"] for row in rows], predictions, [row["target"] for row in rows]
    )
    report["groups"] = len({row["group"] for row in rows})
    report["strength"] = float(strength)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, default=CYCLE)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--frames", type=int, default=32768)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace real Wiener run {args.output}")
    cycle_bytes = args.cycle.read_bytes()
    cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "stage" or cycle.get("round") != 14 or cycle.get("status") != "planned":
        raise ValueError("invalid stage14 cycle")
    args.output.mkdir(parents=True)
    fit_rows = single(args.corpus, "train", "reverb", args.frames, 20261210)
    calibration = single(args.corpus, "calibrate", "reverb", args.frames, 20261211)
    candidates = []
    responses = {}
    for regularization in cycle["calibration"]["regularization"]:
        response = fit(fit_rows, float(regularization))
        responses[regularization] = response
        for strength in cycle["calibration"]["strength"]:
            report = evaluate(calibration, response, float(strength))
            candidate = {"regularization": regularization, "strength": strength, "rank": rank(report), "report": report, "gates": gates(report)}
            candidates.append(candidate)
            print(json.dumps(candidate), flush=True)
    selected = max(candidates, key=lambda row: row["rank"])
    response = responses[selected["regularization"]]
    valid = single(args.corpus, "valid", "reverb", args.frames, 20261212)
    report = evaluate(valid, response, float(selected["strength"]))
    checks = gates(report)
    accepted = all(checks.values())
    artifact = args.output / ("reverb.npz" if accepted else "reverb.candidate.npz")
    np.savez_compressed(
        artifact,
        schema=np.asarray(1, dtype=np.int64),
        sample_rate=np.asarray(48000, dtype=np.int64),
        frames=np.asarray(args.frames, dtype=np.int64),
        response=response,
        regularization=np.asarray(selected["regularization"], dtype=np.float64),
        strength=np.asarray(selected["strength"], dtype=np.float64),
    )
    start = time.perf_counter()
    for row in valid[:32]:
        restore(row["source"], response, float(selected["strength"]))
    seconds = time.perf_counter() - start
    benchmark = {"buffers": 32, "frames": args.frames, "wall_seconds": seconds, "audio_seconds": 32 * args.frames / 48000, "real_time_factor": seconds / (32 * args.frames / 48000), "device": "cpu"}
    result = {
        "schema": 1,
        "status": "accepted-real-development" if accepted else "rejected-real",
        "accepted": accepted,
        "selection": {"candidates": candidates, "selected_regularization": selected["regularization"], "selected_strength": selected["strength"]},
        "report": report,
        "gates": checks,
        "benchmark": benchmark,
        "artifact": {"path": artifact.name, "sha256": sha256(artifact), "bytes": artifact.stat().st_size},
        "training": cycle["training"],
        "limitations": ["single real Reverb family only", "fixed 32768-frame development geometry; overlap runtime remains to be sealed", "Telecaster test remains locked"],
        "data": cycle["data"],
        "quality": cycle["quality"],
        "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest(),
    }
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": accepted, "selected": selected, "valid": report, "gates": checks, "benchmark": benchmark, "output": str(args.output)}), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
