#!/usr/bin/env python3
"""End-to-end Reverb audit: inverse controls followed by capture-fitted forward rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from .data import RATE as INVERSE_RATE
from .data import _waveform, dry_sources
from .evaluate_forward_reverb import _errors
from .forward_drive import FORWARD_RATE, _latin_hypercube
from .forward_reverb import fit_reverb_profile
from .inference import BUNDLE, RemixerRuntime
from .pedalboard_renderer import MIN_REVERB_DECAY_SECONDS, render_pedalboard_chain
from .render import render_chain
from .spec import ChainSpec, Reverb
from .train import CORPUS


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/reverb-forward-phase1"
DOMAINS = ("reference", "alternate", "stress", "pedalboard")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forward_audio(audio: np.ndarray) -> np.ndarray:
    return resample_poly(audio, FORWARD_RATE, INVERSE_RATE).astype(np.float32)


def _decoded_reverb(report: dict) -> Reverb:
    effects = report["chain"]["effects"]
    if len(effects) != 1 or effects[0].get("kind") != "reverb":
        raise ValueError(f"inverse runtime returned an invalid Reverb chain: {effects}")
    effect = effects[0]
    result = Reverb(float(effect["decay_s"]), float(effect["damping"]), float(effect["mix"]))
    result.validate()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--output", type=Path, default=RUN / "closed-loop-validation.json")
    parser.add_argument("--samples-per-domain", type=int, default=12)
    parser.add_argument("--domains", default=",".join(DOMAINS))
    args = parser.parse_args()
    domains = tuple(item.strip() for item in args.domains.split(",") if item.strip())
    if not domains or any(name not in (*DOMAINS, "challenge") for name in domains):
        raise ValueError("invalid Reverb closed-loop domains")

    bundle_hash_before = _sha256(args.bundle)
    inverse = RemixerRuntime(args.bundle, target=torch.device("cpu"))
    sources = dry_sources(args.corpus, "valid")
    controls = _latin_hypercube(args.samples_per_domain * len(domains), 3, 20261210)
    reports = {}
    cursor = 0
    profile_hashes = {}
    for domain in domains:
        profile = fit_reverb_profile(domain)
        profile_hashes[domain] = profile.calibration_hash
        totals = defaultdict(float)
        parameter_errors = {"decay_s": [], "damping": [], "mix": []}
        peak_ratios = []
        for index in range(args.samples_per_domain):
            dry = _waveform(sources[(cursor + index) % len(sources)])
            peak = max(float(np.max(np.abs(dry))), 1.0e-5)
            dry = np.asarray(dry * (0.22 / peak), dtype=np.float32)
            values = controls[cursor + index]
            minimum_decay = MIN_REVERB_DECAY_SECONDS if domain == "pedalboard" else 0.2
            minimum_normalized = math.log(minimum_decay / 0.2) / math.log(40.0)
            decay_control = minimum_normalized + (1.0 - minimum_normalized) * float(values[0])
            truth = Reverb(0.2 * 40.0**decay_control, float(values[1]), float(values[2]) * 0.7)
            spec = ChainSpec((truth,))
            wet = (
                render_pedalboard_chain(dry, spec, INVERSE_RATE)
                if domain == "pedalboard"
                else render_chain(dry, spec, INVERSE_RATE, domain)
            )
            estimated = _decoded_reverb(inverse.infer(dry, wet, ("reverb",)))
            dry_forward = _forward_audio(dry)
            wet_forward = _forward_audio(wet)
            oracle = profile.render(dry_forward, truth)
            closed = profile.render(dry_forward, estimated)
            target_tail = wet_forward - dry_forward * (1.0 - truth.mix)
            oracle_tail = oracle - dry_forward * (1.0 - truth.mix)
            closed_tail = closed - dry_forward * (1.0 - estimated.mix)
            baseline = _errors(
                dry_forward, wet_forward, np.zeros_like(dry_forward), target_tail
            )
            oracle_errors = _errors(oracle, wet_forward, oracle_tail, target_tail)
            closed_errors = _errors(closed, wet_forward, closed_tail, target_tail)
            for name, value in baseline.items():
                totals[f"baseline_{name}"] += value
            for name, value in oracle_errors.items():
                totals[f"oracle_{name}"] += value
            for name, value in closed_errors.items():
                totals[f"closed_loop_{name}"] += value
            parameter_errors["decay_s"].append(abs(estimated.decay_s - truth.decay_s))
            parameter_errors["damping"].append(abs(estimated.damping - truth.damping))
            parameter_errors["mix"].append(abs(estimated.mix - truth.mix))
            peak_ratios.append(
                float(np.max(np.abs(closed)) / max(float(np.max(np.abs(wet_forward))), 1.0e-6))
            )
        cursor += args.samples_per_domain
        metrics = {name: value / args.samples_per_domain for name, value in totals.items()}
        for name in ("esr", "mae", "edc", "spectral"):
            metrics[f"oracle_{name}_improvement"] = 1.0 - metrics[f"oracle_{name}"] / max(
                metrics[f"baseline_{name}"], 1.0e-12
            )
            metrics[f"closed_loop_{name}_improvement"] = 1.0 - metrics[
                f"closed_loop_{name}"
            ] / max(metrics[f"baseline_{name}"], 1.0e-12)
        metrics["parameter_error"] = {
            name: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
            }
            for name, values in parameter_errors.items()
        }
        ordered = sorted(peak_ratios)
        metrics["peak_ratio_p95"] = ordered[
            max(0, math.ceil(0.95 * args.samples_per_domain) - 1)
        ]
        metrics["worst_peak_ratio"] = ordered[-1]
        metrics["passed"] = bool(
            metrics["oracle_edc_improvement"] >= 0.5
            and metrics["oracle_spectral_improvement"] >= 0.25
            and metrics["closed_loop_edc_improvement"] >= 0.5
            and metrics["closed_loop_spectral_improvement"] >= 0.25
            and metrics["peak_ratio_p95"] <= 1.4
            and metrics["worst_peak_ratio"] <= 1.75
        )
        reports[domain] = metrics
        print(json.dumps({"domain": domain, **metrics}, sort_keys=True))
    bundle_hash_after = _sha256(args.bundle)
    challenge = "challenge" in domains
    passed = all(report["passed"] for report in reports.values()) and (
        bundle_hash_before == bundle_hash_after
    )
    report = {
        "schema": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "samples_per_domain": args.samples_per_domain,
        "inverse_bundle": str(args.bundle),
        "inverse_bundle_sha256_before": bundle_hash_before,
        "inverse_bundle_sha256_after": bundle_hash_after,
        "model_artifacts_immutable_during_audit": bundle_hash_before == bundle_hash_after,
        "profile_calibration_hashes": profile_hashes,
        "metrics_policy": (
            "EDC and pooled wet-tail spectrum are acceptance metrics; sample-phase ESR and "
            "MAE are diagnostic across stochastic Reverb sample-rate realizations"
        ),
        "validation": reports,
        "source_files_read_only": True,
        "analysis_copy_resampling_only": True,
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
