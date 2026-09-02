"""Temporal dereverberation metrics that cannot be replaced by global ESR."""

from __future__ import annotations

import numpy as np


def envelope(audio: np.ndarray, frame: int = 480, hop: int = 120) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float64)
    if value.ndim != 1 or len(value) < frame or not np.isfinite(value).all():
        raise ValueError("dereverberation envelope expects finite mono audio")
    power = np.convolve(np.square(value), np.ones(frame) / frame, mode="valid")
    return np.sqrt(np.maximum(power[::hop], 1.0e-12))


def measure(wet: np.ndarray, restored: np.ndarray, clean: np.ndarray) -> dict:
    clean_env, wet_env, restored_env = envelope(clean), envelope(wet), envelope(restored)
    length = min(len(clean_env), len(wet_env), len(restored_env))
    clean_env, wet_env, restored_env = clean_env[:length], wet_env[:length], restored_env[:length]
    active = clean_env > np.quantile(clean_env, 0.75)
    # Forty 2.5 ms hops mark quiet frames following activity in the last 100 ms.
    recent = np.convolve(active.astype(np.float64), np.ones(40), mode="full")[:length] > 0
    tail = (clean_env <= np.quantile(clean_env, 0.35)) & recent
    if int(tail.sum()) < 4:
        return {"eligible": False}
    wet_excess = float(np.mean(np.maximum(wet_env[tail] - clean_env[tail], 0.0)))
    restored_excess = float(np.mean(np.maximum(restored_env[tail] - clean_env[tail], 0.0)))
    measurable = max(1.0e-5, 0.02 * float(np.mean(wet_env[tail])))
    if wet_excess < measurable:
        return {"eligible": False}
    wet_error = float(np.mean(np.square(wet_env[tail] - clean_env[tail])))
    restored_error = float(np.mean(np.square(restored_env[tail] - clean_env[tail])))
    reduction = 1.0 - restored_excess / max(wet_excess, 1.0e-12)
    envelope_improvement = 1.0 - restored_error / max(wet_error, 1.0e-12)
    added = restored_excess > wet_excess * 1.02 + 1.0e-6
    passed = reduction >= 0.20 and envelope_improvement >= 0.15 and not added
    return {"eligible": True, "tail_frames": int(tail.sum()), "wet_tail_excess": wet_excess, "restored_tail_excess": restored_excess, "tail_excess_reduction": reduction, "tail_envelope_esr_improvement": envelope_improvement, "added_reverb": bool(added), "passed": bool(passed)}


def summarize(wet: list[np.ndarray], restored: list[np.ndarray], clean: list[np.ndarray]) -> dict:
    rows = [measure(left, middle, right) for left, middle, right in zip(wet, restored, clean, strict=True)]
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ValueError("dereverberation audit has no eligible tails")
    mean = lambda name: float(np.mean([row[name] for row in eligible]))
    result = {"examples": len(rows), "eligible": len(eligible), "mean_tail_excess_reduction": mean("tail_excess_reduction"), "median_tail_excess_reduction": float(np.median([row["tail_excess_reduction"] for row in eligible])), "mean_tail_envelope_esr_improvement": mean("tail_envelope_esr_improvement"), "median_tail_envelope_esr_improvement": float(np.median([row["tail_envelope_esr_improvement"] for row in eligible])), "pass_fraction": float(np.mean([row["passed"] for row in eligible])), "added_reverb_fraction": float(np.mean([row["added_reverb"] for row in eligible]))}
    result["gates"] = {"tail_reduction": result["median_tail_excess_reduction"] >= 0.20, "tail_envelope": result["median_tail_envelope_esr_improvement"] >= 0.15, "pass_fraction": result["pass_fraction"] >= 0.50, "added_reverb": result["added_reverb_fraction"] <= 0.05}
    result["accepted"] = all(result["gates"].values())
    return result
