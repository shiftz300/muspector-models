#!/usr/bin/env python3
"""Final source/control-disjoint full-chain audit for a Drive device adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

from .data import dry_sources
from .evaluate_forward_chain import NONEMPTY_TOPOLOGIES, _distance, _random_spec, _target
from .forward_chain import DRIVE_CHECKPOINT, ForwardChainRuntime
from .forward_drive import FORWARD_RATE, _segment
from .forward_reverb import fit_reverb_profile
from .spec import ChainSpec, Delay, Drive, Reverb
from .train import CORPUS


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/drive-adapter-pilot"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--adapter", type=Path, default=RUN / "drive-device-adapter.pt")
    parser.add_argument("--domain", default="final_challenge")
    parser.add_argument("--frames", type=int, default=96_000)
    args = parser.parse_args()
    training = json.loads((args.run / "metrics.json").read_text())
    adapter_payload = __import__("torch").load(
        args.adapter, map_location="cpu", weights_only=True
    )
    if not training["promotable"]:
        raise ValueError("Drive adapter did not pass its isolated validation")
    if adapter_payload["base_checkpoint_sha256"] != _sha256(DRIVE_CHECKPOINT):
        raise ValueError("Drive adapter was trained against another base checkpoint")
    sources = [
        path for path in dry_sources(args.corpus, "train") if path.name.startswith("les_")
    ]
    if not sources:
        raise ValueError("adapter chain-final Les Paul sources are missing")
    profile = fit_reverb_profile(args.domain)
    generic = ForwardChainRuntime(reverb_profile=profile)
    calibrated = ForwardChainRuntime(
        reverb_profile=profile,
        drive_adapter=args.adapter,
    )
    reports = {}
    improvements = []
    peak_ratios = []
    for index, topology in enumerate(NONEMPTY_TOPOLOGIES):
        rng = random.Random(20270210 + index * 104_729)
        dry = _segment(sources[index % len(sources)], args.frames, rng)
        peak = max(float(np.max(np.abs(dry))), 1.0e-6)
        dry = np.asarray(dry * (0.16 / peak), dtype=np.float32)
        spec = _random_spec(rng, topology, args.domain)
        target = _target(dry, spec, FORWARD_RATE, args.domain)
        before = generic.render(dry, spec)
        after = calibrated.render(dry, spec)
        has_reverb = "reverb" in topology
        before_error = _distance(before, target, has_reverb=has_reverb)
        after_error = _distance(after, target, has_reverb=has_reverb)
        improvement = 1.0 - after_error["acceptance"] / max(
            before_error["acceptance"], 1.0e-12
        )
        if "drive" in topology:
            improvements.append(improvement)
            peak_ratios.append(
                float(np.max(np.abs(after)) / max(float(np.max(np.abs(target))), 1.0e-8))
            )
        reports["->".join(topology)] = {
            "generic": before_error,
            "calibrated": after_error,
            "adapter_improvement": improvement,
            "adapter_active": "drive" in topology,
            "finite": bool(np.isfinite(after).all()),
        }
    positive_fraction = sum(value >= 0.0 for value in improvements) / len(improvements)
    ordered_peaks = sorted(peak_ratios)

    rng = np.random.default_rng(20270211)
    stream_source = (rng.standard_normal(4_321) * 0.035).astype(np.float32)
    parity = {}
    for index, topology in enumerate(NONEMPTY_TOPOLOGIES):
        spec = _random_spec(random.Random(20270211 + index), topology, args.domain)
        complete = calibrated.render(stream_source, spec)
        streamed = calibrated.stream(stream_source, spec, 1_024)
        parity["->".join(topology)] = float(np.max(np.abs(complete - streamed)))
    bypass = calibrated.render(stream_source, ChainSpec(()))
    silence_spec = ChainSpec(
        (Drive(24.0, 0.8, 3.0), Delay(40.0, 0.9, 0.7), Reverb(8.0, 1.0, 0.7))
    )
    silence = calibrated.stream(np.zeros_like(stream_source), silence_spec)
    aggregate = {
        "drive_topologies": len(improvements),
        "mean_adapter_improvement": float(np.mean(improvements)),
        "median_adapter_improvement": float(np.median(improvements)),
        "worst_adapter_improvement": min(improvements),
        "positive_topology_fraction": positive_fraction,
        "peak_ratio_p95": ordered_peaks[max(0, math.ceil(0.95 * len(ordered_peaks)) - 1)],
        "worst_peak_ratio": ordered_peaks[-1],
        "streaming_max_absolute_error": max(parity.values()),
        "bypass_max_absolute_error": float(np.max(np.abs(bypass - stream_source))),
        "silence_max_absolute_output": float(np.max(np.abs(silence))),
    }
    gates = {
        "isolated_adapter_validation_passed": bool(training["promotable"]),
        "mean_chain_improvement_at_least_20_percent": aggregate[
            "mean_adapter_improvement"
        ]
        >= 0.20,
        "at_least_80_percent_drive_topologies_improve": positive_fraction >= 0.80,
        "chain_peak_ratio_p95_within_1_35": aggregate["peak_ratio_p95"] <= 1.35,
        "chain_worst_peak_ratio_within_1_75": aggregate["worst_peak_ratio"] <= 1.75,
        "streaming_parity": aggregate["streaming_max_absolute_error"] <= 2.0e-6,
        "bypass_bit_exact": aggregate["bypass_max_absolute_error"] == 0.0,
        "silence_bit_exact": aggregate["silence_max_absolute_output"] == 0.0,
        "all_outputs_finite": all(value["finite"] for value in reports.values()),
    }
    accepted = all(gates.values())
    report = {
        "schema": 1,
        "status": "accepted-synthetic-device-adapter" if accepted else "rejected",
        "accepted": accepted,
        "scope": {
            "method_ready_for_aligned_capture-kit_trials": accepted,
            "real_hardware_fidelity_claim_allowed": False,
            "ready_for_ui_integration": False,
            "target_domain": args.domain,
        },
        "adapter": str(args.adapter),
        "adapter_sha256": _sha256(args.adapter),
        "base_checkpoint_sha256": _sha256(DRIVE_CHECKPOINT),
        "reverb_calibration_hash": profile.calibration_hash,
        "chain_final_split": "Les Paul sources; unseen by adapter training and selection",
        "aggregate": aggregate,
        "gates": gates,
        "topologies": reports,
        "streaming_parity": parity,
        "quality_policy": {
            "source_files_read_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "automatic_dither": False,
            "lossy_reencoding": False,
            "locked_tele_test_opened": False,
            "physical_audio_devices_used": False,
        },
    }
    (args.run / "chain-acceptance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"accepted": accepted, "aggregate": aggregate, "gates": gates}, sort_keys=True))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
