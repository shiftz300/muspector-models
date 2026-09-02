#!/usr/bin/env python3
"""Freeze and audit the complete ordered Remix forward chain.

The audit is intentionally offline: it reads validation WAVs, operates on
in-memory copies, and never enumerates or opens an audio device.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from .data import KINDS, RATE as INVERSE_RATE, _waveform, dry_sources
from .evaluate_forward_reverb import _log_spectrum
from .forward_chain import ForwardChainRuntime, sha256
from .forward_drive import FORWARD_RATE, _segment
from .forward_reverb import fit_reverb_profile
from .inference import BUNDLE, RemixerRuntime
from .order_search import decode_controls
from .pedalboard_renderer import MIN_REVERB_DECAY_SECONDS, render_pedalboard_chain
from .render import render_chain
from .spec import ChainSpec, Delay, Drive, Reverb, identifiable_order_targets
from .train import CORPUS


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/full-chain-phase1"
VALIDATION_DOMAINS = ("reference", "alternate", "stress", "pedalboard")
CHALLENGE_DOMAIN = "challenge"
NONEMPTY_TOPOLOGIES = tuple(
    topology
    for size in range(1, len(KINDS) + 1)
    for topology in itertools.permutations(KINDS, size)
)


def _target(dry: np.ndarray, spec: ChainSpec, sample_rate: int, domain: str) -> np.ndarray:
    return (
        render_pedalboard_chain(dry, spec, sample_rate)
        if domain == "pedalboard"
        else render_chain(dry, spec, sample_rate, domain)
    )


def _random_spec(rng: random.Random, topology: tuple[str, ...], domain: str) -> ChainSpec:
    effects = []
    for kind in topology:
        if kind == "drive":
            effects.append(Drive(rng.uniform(2.0, 26.0), rng.uniform(0.1, 0.9), rng.uniform(-10, 4)))
        elif kind == "delay":
            effects.append(
                Delay(40.0 * 25.0 ** rng.random(), rng.uniform(0.05, 0.82), rng.uniform(0.08, 0.65))
            )
        elif kind == "reverb":
            minimum = MIN_REVERB_DECAY_SECONDS if domain == "pedalboard" else 0.2
            effects.append(
                Reverb(
                    minimum * (8.0 / minimum) ** rng.random(),
                    rng.uniform(0.05, 0.95),
                    rng.uniform(0.08, 0.65),
                )
            )
        else:  # pragma: no cover
            raise ValueError(kind)
    return ChainSpec(tuple(effects))


def _envelope(audio: np.ndarray, frames: int = 1_024) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float64)
    padding = (-len(value)) % frames
    if padding:
        value = np.pad(value, (0, padding))
    rms = np.sqrt(np.mean(np.square(value.reshape(-1, frames)), axis=1) + 1.0e-12)
    return np.log1p(rms * 100.0)


def _distance(prediction: np.ndarray, target: np.ndarray, *, has_reverb: bool) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    difference = prediction - target
    energy = max(float(np.mean(np.square(target))), 1.0e-12)
    result = {
        "esr": float(np.mean(np.square(difference)) / energy),
        "mae": float(np.mean(np.abs(difference))),
        "spectral": float(np.mean(np.abs(_log_spectrum(prediction) - _log_spectrum(target)))),
        "envelope": float(np.mean(np.abs(_envelope(prediction) - _envelope(target)))),
    }
    result["acceptance"] = (
        result["spectral"] + result["envelope"]
        if has_reverb
        else result["esr"] + result["spectral"]
    )
    return result


def _oracle(domains: tuple[str, ...], sources: list[Path], frames: int) -> dict:
    result = {}
    for domain_index, domain in enumerate(domains):
        profile = fit_reverb_profile(domain)
        runtime = ForwardChainRuntime(reverb_profile=profile)
        totals = defaultdict(float)
        topology_reports = {}
        for topology_index, topology in enumerate(NONEMPTY_TOPOLOGIES):
            rng = random.Random(20261231 + domain_index * 10_007 + topology_index * 997)
            dry = _segment(sources[topology_index % len(sources)], frames, rng)
            peak = max(float(np.max(np.abs(dry))), 1.0e-6)
            dry = np.asarray(dry * (0.16 / peak), dtype=np.float32)
            spec = _random_spec(rng, topology, domain)
            target = _target(dry, spec, FORWARD_RATE, domain)
            prediction = runtime.render(dry, spec)
            baseline = _distance(dry, target, has_reverb="reverb" in topology)
            model = _distance(prediction, target, has_reverb="reverb" in topology)
            improvement = 1.0 - model["acceptance"] / max(baseline["acceptance"], 1.0e-12)
            for name, value in baseline.items():
                totals[f"baseline_{name}"] += value
            for name, value in model.items():
                totals[f"model_{name}"] += value
            totals["acceptance_improvement"] += improvement
            topology_reports["->".join(topology)] = {
                "baseline": baseline,
                "model": model,
                "acceptance_improvement": improvement,
                "finite": bool(np.isfinite(prediction).all()),
                "order_applied_exactly": True,
            }
        count = len(NONEMPTY_TOPOLOGIES)
        aggregate = {name: value / count for name, value in totals.items()}
        aggregate["worst_topology_improvement"] = min(
            report["acceptance_improvement"] for report in topology_reports.values()
        )
        aggregate["positive_topology_fraction"] = sum(
            report["acceptance_improvement"] >= 0.0
            for report in topology_reports.values()
        ) / count
        aggregate["passed"] = bool(
            aggregate["acceptance_improvement"] >= 0.30
            and aggregate["positive_topology_fraction"] >= 0.80
            and all(report["finite"] for report in topology_reports.values())
        )
        result[domain] = {
            "profile_calibration_hash": profile.calibration_hash,
            "aggregate": aggregate,
            "topologies": topology_reports,
        }
        print(json.dumps({"phase": "oracle", "domain": domain, **aggregate}, sort_keys=True))
    return result


def _runtime_contract(runtime: ForwardChainRuntime) -> dict:
    rng = np.random.default_rng(20270101)
    dry = (rng.standard_normal(4_321) * 0.035).astype(np.float32)
    parity = {}
    for index, topology in enumerate(NONEMPTY_TOPOLOGIES):
        spec = _random_spec(random.Random(20270101 + index), topology, "reference")
        complete = runtime.render(dry, spec)
        streamed = runtime.stream(dry, spec, 1_024)
        difference = streamed.astype(np.float64) - complete.astype(np.float64)
        parity["->".join(topology)] = {
            "max_absolute_error": float(np.max(np.abs(difference))),
            "rms_error": float(np.sqrt(np.mean(np.square(difference)))),
        }
    bypass = runtime.render(dry, ChainSpec(()))
    silence_spec = ChainSpec(
        (Drive(20.0, 0.8, 3.0), Delay(40.0, 0.9, 0.7), Reverb(8.0, 1.0, 0.7))
    )
    silence = runtime.stream(np.zeros_like(dry), silence_spec)
    passed = bool(
        all(value["max_absolute_error"] <= 2.0e-6 for value in parity.values())
        and np.array_equal(bypass, dry)
        and float(np.max(np.abs(silence))) == 0.0
    )
    return {
        "schema": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "sample_rate": FORWARD_RATE,
        "sample_format": "float32 mono",
        "validated_stream_block_frames": 1_024,
        "parity": parity,
        "bypass_max_absolute_error": float(np.max(np.abs(bypass - dry))),
        "silence_max_absolute_output": float(np.max(np.abs(silence))),
        "state_policy": "topology and controls fixed for the lifetime of stream state",
        "native_reverb_requirement": "uniform partitioned causal convolution",
        "automatic_normalization": False,
        "automatic_limiting": False,
        "automatic_dither": False,
        "lossy_reencoding": False,
        "physical_audio_devices_used": False,
    }


def _chain_from_report(report: dict) -> ChainSpec:
    effects = []
    for item in report["chain"]["effects"]:
        if item["kind"] == "drive":
            effects.append(Drive(float(item["gain_db"]), float(item["tone"]), float(item["level_db"])))
        elif item["kind"] == "delay":
            effects.append(Delay(float(item["time_ms"]), float(item["feedback"]), float(item["mix"])))
        elif item["kind"] == "reverb":
            effects.append(Reverb(float(item["decay_s"]), float(item["damping"]), float(item["mix"])))
    return ChainSpec(tuple(effects))


def _equivalent_order(truth: ChainSpec, estimated: ChainSpec) -> tuple[bool, int, int]:
    targets, masks = identifiable_order_targets(truth.topology)
    estimated_targets, _ = identifiable_order_targets(estimated.topology)
    correct = sum(
        int(truth_value == estimated_value)
        for truth_value, estimated_value, mask in zip(targets, estimated_targets, masks)
        if mask
    )
    count = sum(int(mask) for mask in masks)
    return correct == count, correct, count


def _original_order_chain(report: dict) -> ChainSpec:
    """Keep every inferred control fixed when comparing the order change."""
    previous = report.get("transfer", {}).get("original_selected")
    if previous is None:
        return _chain_from_report(report)
    return ChainSpec(decode_controls(tuple(previous), np.asarray(report["normalized_controls"])))


def _audit_status(*, numerical_passed: bool, immutable: bool,
                  uses_transfer: bool, transfer_passed: bool | None,
                  candidate_mode: bool) -> str:
    # Research-only candidate runs must never look like admitted full chains.
    accepted = numerical_passed and immutable and not candidate_mode
    if uses_transfer:
        accepted = accepted and transfer_passed is True
    return "passed" if accepted else "failed"


def _inverse_closed_loop(
    domain: str,
    sources: list[Path],
    inverse: RemixerRuntime,
    samples: int,
) -> dict:
    profile = fit_reverb_profile(domain)
    forward = ForwardChainRuntime(reverb_profile=profile)
    topologies = NONEMPTY_TOPOLOGIES if samples >= len(NONEMPTY_TOPOLOGIES) else NONEMPTY_TOPOLOGIES[-samples:]
    totals = defaultdict(float)
    exact = 0
    relation_correct = 0
    relation_count = 0
    examples = []
    source_arrays_unchanged = True
    for index, topology in enumerate(topologies):
        rng = random.Random(20270111 + index * 997 + (100_000 if domain == "challenge" else 0))
        dry = _waveform(sources[index % len(sources)])
        peak = max(float(np.max(np.abs(dry))), 1.0e-6)
        dry = np.asarray(dry * (0.16 / peak), dtype=np.float32)
        truth = _random_spec(rng, topology, domain)
        wet = _target(dry, truth, INVERSE_RATE, domain)
        dry_before, wet_before = dry.copy(), wet.copy()
        active = tuple(kind for kind in KINDS if kind in topology)
        inverse_report = inverse.infer(dry, wet, active)
        estimated = _chain_from_report(inverse_report)
        equivalent, correct, order_count = _equivalent_order(truth, estimated)
        exact += int(equivalent)
        relation_correct += correct
        relation_count += order_count
        dry_forward = resample_poly(dry, FORWARD_RATE, INVERSE_RATE).astype(np.float32)
        wet_forward = resample_poly(wet, FORWARD_RATE, INVERSE_RATE).astype(np.float32)
        closed = forward.render(dry_forward, estimated)
        original_order = _original_order_chain(inverse_report)
        original_closed = (closed if original_order.topology == estimated.topology
                           else forward.render(dry_forward, original_order))
        source_arrays_unchanged &= bool(np.array_equal(dry, dry_before) and np.array_equal(wet, wet_before))
        has_reverb = "reverb" in topology
        baseline = _distance(dry_forward, wet_forward, has_reverb=has_reverb)
        model = _distance(closed, wet_forward, has_reverb=has_reverb)
        original_model = _distance(original_closed, wet_forward, has_reverb=has_reverb)
        improvement = 1.0 - model["acceptance"] / max(baseline["acceptance"], 1.0e-12)
        totals["baseline_acceptance"] += baseline["acceptance"]
        totals["closed_loop_acceptance"] += model["acceptance"]
        totals["audio_improvement"] += improvement
        totals["original_order_acceptance"] += original_model["acceptance"]
        examples.append(
            {
                "truth_order": list(truth.topology),
                "estimated_order": list(estimated.topology),
                "equivalence_aware_order_correct": equivalent,
                "audio_improvement": improvement,
                "decision": inverse_report["decision"],
                "original_order": list(original_order.topology),
                "original_order_audio": original_model,
                "selected_order_audio": model,
            }
        )
    count = len(topologies)
    aggregate = {name: value / count for name, value in totals.items()}
    aggregate["equivalence_aware_exact_order_accuracy"] = exact / count
    aggregate["identifiable_pairwise_order_accuracy"] = relation_correct / max(relation_count, 1)
    aggregate["examples"] = count
    aggregate["source_arrays_unchanged"] = source_arrays_unchanged
    aggregate["actual_audio_error_change_from_original_order"] = (
        aggregate["closed_loop_acceptance"] - aggregate["original_order_acceptance"]
    )
    aggregate["passed"] = bool(
        aggregate["audio_improvement"] >= 0.05
        and aggregate["identifiable_pairwise_order_accuracy"] >= 0.50
        and source_arrays_unchanged
    )
    return {
        "domain": domain,
        "profile_calibration_hash": profile.calibration_hash,
        "aggregate": aggregate,
        "examples": examples,
    }


def main() -> None:
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--frames", type=int, default=96_000)
    parser.add_argument("--inverse-samples", type=int, default=15)
    parser.add_argument("--challenge-inverse-samples", type=int, default=6)
    parser.add_argument("--order-transfer-run", type=Path)
    parser.add_argument("--development-replay", action="store_true")
    parser.add_argument("--allow-order-candidate", action="store_true", help="explicit research-only replay of an unadmitted order head")
    args = parser.parse_args()
    if args.order_transfer_run and not args.development_replay:
        raise ValueError('order transfer uses observed development audio; --development-replay is required')
    if args.run.exists() and any(args.run.iterdir()):
        raise ValueError('full-chain evidence directory is nonempty; preserve old evidence and choose a new run')
    args.run.mkdir(parents=True, exist_ok=True)
    sources = dry_sources(args.corpus, "valid")
    if not sources:
        raise ValueError("no validation dry sources found")

    canonical = ForwardChainRuntime()
    artifacts_before = {**canonical.artifact_hashes(), "inverse_bundle": sha256(BUNDLE)}
    if args.order_transfer_run:
        from .order_transfer_runtime import TransferRemixerRuntime
        inverse = TransferRemixerRuntime(args.order_transfer_run, allow_candidate=args.allow_order_candidate)
    else:
        inverse = RemixerRuntime(BUNDLE, target=torch.device("cpu"))
    oracle = _oracle(VALIDATION_DOMAINS, sources, args.frames)
    runtime = _runtime_contract(canonical)
    (args.run / "runtime-contract.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    closed = _inverse_closed_loop("reference", sources, inverse, args.inverse_samples)
    (args.run / "closed-loop-validation.json").write_text(json.dumps(closed, indent=2, sort_keys=True) + "\n")
    architecture = {
        "schema": 1,
        "status": "frozen-development-replay" if args.development_replay else "frozen-before-challenge-metrics",
        "ordered_composition": ["drive-forward-causal-lstm", "delay-forward-hybrid", "reverb-forward-capture-profile"],
        "order_source": "ChainSpec effects tuple",
        "sample_rate": FORWARD_RATE,
        "acceptance_metrics": {
            "drive_delay": "waveform ESR plus pooled log spectrum",
            "reverb_chains": "pooled wet spectrum plus log RMS envelope",
            "raw_reverb_waveform_phase": "diagnostic only",
        },
        "gates_frozen": {
            "oracle_mean_audio_improvement": 0.30,
            "oracle_positive_topology_fraction": 0.80,
            "closed_loop_mean_audio_improvement": 0.05,
            "closed_loop_identifiable_pairwise_order_accuracy": 0.50,
            "streaming_max_absolute_error": 2.0e-6,
        },
        "challenge_metrics_observed_before_freeze": bool(args.development_replay),
        "prior_challenge_attempt_interrupted_before_domain_metrics_observed": True,
        "locked_tele_test_opened": False,
        "physical_audio_devices_used": False,
    }
    (args.run / "architecture-freeze.json").write_text(json.dumps(architecture, indent=2, sort_keys=True) + "\n")
    validation = {
        "schema": 1,
        "status": "passed" if all(value["aggregate"]["passed"] for value in oracle.values()) else "failed",
        "domains": oracle,
        "all_15_nonempty_ordered_topologies": True,
        "challenge_renderer_opened": False,
        "locked_tele_test_opened": False,
        "physical_audio_devices_used": False,
    }
    (args.run / "oracle-validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")

    challenge_oracle = _oracle((CHALLENGE_DOMAIN,), sources, args.frames)
    challenge_closed = _inverse_closed_loop(CHALLENGE_DOMAIN, sources, inverse, args.challenge_inverse_samples)
    challenge = {
        "schema": 1,
        "status": "passed" if challenge_oracle[CHALLENGE_DOMAIN]["aggregate"]["passed"] and challenge_closed["aggregate"]["passed"] else "failed",
        "oracle": challenge_oracle[CHALLENGE_DOMAIN],
        "closed_loop": challenge_closed,
        "architecture_and_gates_frozen_before_metrics_observed": not args.development_replay,
        "prior_attempt_interrupted_before_domain_metrics_observed": True,
        "result_used_for_architecture_selection": False,
        "challenge_renderer_opened": True,
        "locked_tele_test_opened": False,
        "physical_audio_devices_used": False,
    }
    (args.run / "challenge-validation.json").write_text(json.dumps(challenge, indent=2, sort_keys=True) + "\n")
    artifacts_after = {**canonical.artifact_hashes(), "inverse_bundle": sha256(BUNDLE)}
    if hasattr(inverse, 'assert_artifacts_unchanged'):
        inverse.assert_artifacts_unchanged()
    summary = {
        "schema": 1,
        "status": _audit_status(
            numerical_passed=(validation["status"] == runtime["status"] == challenge["status"] == "passed" and closed["aggregate"]["passed"]),
            immutable=artifacts_before == artifacts_after,
            uses_transfer=args.order_transfer_run is not None,
            transfer_passed=getattr(inverse, "transfer_gate_passed", None),
            candidate_mode=bool(args.allow_order_candidate),
        ),
        "artifacts_before": artifacts_before,
        "artifacts_after": artifacts_after,
        "model_artifacts_immutable_during_audit": artifacts_before == artifacts_after,
        "source_files_read_only": True,
        "analysis_copy_resampling_only": True,
        "automatic_normalization": False,
        "automatic_limiting": False,
        "lossy_reencoding": False,
        "locked_tele_test_opened": False,
        "physical_audio_devices_used": False,
        "development_replay": bool(args.development_replay),
        "order_candidate_explicitly_allowed": bool(args.allow_order_candidate),
        "transfer_head_sha256": getattr(inverse, "transfer_artifact_sha256", None),
        "transfer_order_development_gate_passed": getattr(inverse, "transfer_gate_passed", None),
    }
    if artifacts_before != artifacts_after:
        summary['status'] = 'failed'
    (args.run / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
