#!/usr/bin/env python3
"""Evaluate capture-fitted Reverb profiles with phase-appropriate metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .forward_drive import FORWARD_RATE, _latin_hypercube
from .forward_reverb import ReverbDeviceProfile, fit_reverb_profile, load_reverb_profile
from .pedalboard_renderer import MIN_REVERB_DECAY_SECONDS
from .render import render_chain
from .pedalboard_renderer import render_pedalboard_chain
from .spec import ChainSpec, Reverb


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/reverb-forward-phase1"
DOMAINS = ("reference", "alternate", "stress", "pedalboard")


def _render_target(dry: np.ndarray, effect: Reverb, domain: str) -> np.ndarray:
    spec = ChainSpec((effect,))
    return (
        render_pedalboard_chain(dry, spec, FORWARD_RATE)
        if domain == "pedalboard"
        else render_chain(dry, spec, FORWARD_RATE, domain)
    )


def _edc_db(audio: np.ndarray, points: int = 384) -> np.ndarray:
    energy = np.square(audio.astype(np.float64))
    curve = np.cumsum(energy[::-1])[::-1]
    curve /= max(float(curve[0]), 1.0e-20)
    indexes = np.linspace(0, len(curve) - 1, points).astype(np.int64)
    return np.maximum(10.0 * np.log10(np.maximum(curve[indexes], 1.0e-10)), -80.0)


def _log_spectrum(audio: np.ndarray) -> np.ndarray:
    power = np.square(np.abs(np.fft.rfft(audio.astype(np.float64))))
    # Pool fine comb-filter phase into a stable spectral envelope. Exact FFT
    # bins are not a perceptual fidelity metric for stochastic Reverb tails.
    return np.asarray(
        [np.log1p(math.sqrt(float(np.mean(bucket)))) for bucket in np.array_split(power, 512)]
    )


def _errors(
    prediction: np.ndarray,
    target: np.ndarray,
    prediction_tail: np.ndarray,
    target_tail: np.ndarray,
) -> dict[str, float]:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    energy = max(float(np.mean(np.square(target, dtype=np.float64))), 1.0e-12)
    prediction_edc = _edc_db(prediction_tail)
    target_edc = _edc_db(target_tail)
    audible_decay = target_edc > -60.0
    return {
        "esr": float(np.mean(np.square(difference)) / energy),
        "mae": float(np.mean(np.abs(difference))),
        "edc": float(np.mean(np.abs(prediction_edc[audible_decay] - target_edc[audible_decay]))),
        "spectral": float(
            np.mean(np.abs(_log_spectrum(prediction_tail) - _log_spectrum(target_tail)))
        ),
    }


def evaluate_profile(
    profile: ReverbDeviceProfile,
    domain: str,
    samples: int,
    frames: int,
    seed: int,
) -> dict:
    controls = _latin_hypercube(samples, 3, seed)
    totals = defaultdict(float)
    peak_ratios = []
    finite = True
    for values in controls:
        minimum_decay = MIN_REVERB_DECAY_SECONDS if domain == "pedalboard" else 0.2
        normalized_minimum = math.log(minimum_decay / 0.2) / math.log(40.0)
        normalized_decay = normalized_minimum + (1.0 - normalized_minimum) * float(values[0])
        effect = Reverb(
            0.2 * 40.0**normalized_decay,
            float(values[1]),
            float(values[2]) * 0.7,
        )
        dry = np.zeros(frames, dtype=np.float32)
        dry[0] = 1.0
        target = _render_target(dry, effect, domain)
        prediction = profile.render(dry, effect)
        target_tail = target - dry * (1.0 - effect.mix)
        prediction_tail = prediction - dry * (1.0 - effect.mix)
        model = _errors(prediction, target, prediction_tail, target_tail)
        baseline = _errors(dry, target, np.zeros_like(dry), target_tail)
        for name, value in model.items():
            totals[f"model_{name}"] += value
        for name, value in baseline.items():
            totals[f"baseline_{name}"] += value
        peak_ratios.append(
            float(np.max(np.abs(prediction)) / max(float(np.max(np.abs(target))), 1.0e-8))
        )
        finite = finite and bool(np.isfinite(prediction).all())
    metrics = {name: value / samples for name, value in totals.items()}
    for name in ("esr", "mae", "edc", "spectral"):
        metrics[f"{name}_improvement"] = 1.0 - metrics[f"model_{name}"] / max(
            metrics[f"baseline_{name}"], 1.0e-12
        )
    ordered = sorted(peak_ratios)
    metrics["peak_ratio_p95"] = ordered[max(0, math.ceil(0.95 * samples) - 1)]
    metrics["worst_peak_ratio"] = ordered[-1]
    metrics["examples"] = samples
    metrics["finite"] = finite
    metrics["passed"] = bool(
        finite
        and metrics["edc_improvement"] >= 0.5
        and metrics["spectral_improvement"] >= 0.25
        and metrics["mae_improvement"] >= 0.0
        and metrics["peak_ratio_p95"] <= 1.35
        and metrics["worst_peak_ratio"] <= 1.75
    )
    return metrics


def invariant_audit(profile: ReverbDeviceProfile) -> dict:
    rng = np.random.default_rng(20261130)
    dry = (rng.standard_normal(8_192) * 0.05).astype(np.float32)
    bypass = profile.render(dry, Reverb(1.0, 0.5, 0.0))
    silence = profile.render(np.zeros_like(dry), Reverb(8.0, 1.0, 0.7))
    return {
        "bypass_max_absolute_error": float(np.max(np.abs(bypass - dry))),
        "silence_max_absolute_output": float(np.max(np.abs(silence))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=RUN / "reverb-device-profile.npz")
    parser.add_argument("--output", type=Path, default=RUN / "validation.json")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--frames", type=int, default=120_000)
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--seed", type=int, default=20261131)
    args = parser.parse_args()
    domains = tuple(item.strip() for item in args.domains.split(",") if item.strip())
    if not domains or any(name not in (*DOMAINS, "challenge") for name in domains):
        raise ValueError("invalid Reverb validation domains")
    validation = {}
    for domain in domains:
        profile = (
            load_reverb_profile(args.profile)
            if domain == "reference"
            else fit_reverb_profile(domain)
        )
        validation[domain] = evaluate_profile(
            profile, domain, args.samples, args.frames, args.seed
        )
        print(json.dumps({"domain": domain, **validation[domain]}, sort_keys=True))
    canonical = load_reverb_profile(args.profile)
    invariants = invariant_audit(canonical)
    passed = bool(
        all(report["passed"] for report in validation.values())
        and invariants["bypass_max_absolute_error"] == 0.0
        and invariants["silence_max_absolute_output"] == 0.0
    )
    challenge = "challenge" in domains
    report = {
        "schema": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "profile": str(args.profile),
        "samples_per_domain": args.samples,
        "frames": args.frames,
        "sample_rate": FORWARD_RATE,
        "metrics_policy": "EDC and log spectrum are primary; waveform phase is diagnostic",
        "profile_fit_per_domain": True,
        "validation": validation,
        "invariants": invariants,
        "source_files_read_only": True,
        "runtime_automatic_normalization": False,
        "challenge_renderer_opened": challenge,
        "challenge_profile_fit_from_calibration_grid": challenge,
        "challenge_result_used_for_architecture_selection": False,
        "locked_tele_test_opened": False,
        "physical_audio_devices_used": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": passed}, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
