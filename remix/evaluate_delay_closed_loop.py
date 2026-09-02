#!/usr/bin/env python3
"""End-to-end Delay audit: inverse controls followed by forward rendering."""

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
from .data import SAMPLES as INVERSE_SAMPLES
from .data import _waveform, dry_sources
from .forward_delay import DelayForwardRenderer, normalized_delay_controls
from .forward_drive import FORWARD_RATE, _latin_hypercube
from .inference import BUNDLE, RemixerRuntime
from .pedalboard_renderer import render_pedalboard_chain
from .render import render_chain
from .spec import ChainSpec, Delay
from .train import CORPUS


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/delay-forward-pilot"
DOMAINS = ("reference", "alternate", "stress", "pedalboard")
ALL_DOMAINS = (*DOMAINS, "challenge")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forward_audio(audio: np.ndarray) -> np.ndarray:
    value = resample_poly(audio, FORWARD_RATE, INVERSE_RATE).astype(np.float32)
    expected = round(len(audio) * FORWARD_RATE / INVERSE_RATE)
    if len(value) < expected:
        value = np.pad(value, (0, expected - len(value)))
    return value[:expected]


def _errors(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    energy = max(float(np.mean(np.square(target, dtype=np.float64))), 1.0e-12)
    return float(np.mean(np.square(difference)) / energy), float(np.mean(np.abs(difference)))


def _decoded_delay(report: dict) -> Delay:
    effects = report["chain"]["effects"]
    if len(effects) != 1 or effects[0].get("kind") != "delay":
        raise ValueError(f"inverse runtime returned an invalid Delay chain: {effects}")
    effect = effects[0]
    result = Delay(float(effect["time_ms"]), float(effect["feedback"]), float(effect["mix"]))
    result.validate()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--delay-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=RUN / "delay-forward-candidate.pt")
    parser.add_argument("--output", type=Path, default=RUN / "closed-loop-validation.json")
    parser.add_argument("--samples-per-domain", type=int, default=12)
    parser.add_argument("--domains", default=",".join(DOMAINS))
    args = parser.parse_args()
    domains = tuple(name.strip() for name in args.domains.split(",") if name.strip())
    if not domains or any(name not in ALL_DOMAINS for name in domains):
        raise ValueError(f"closed-loop domains must be selected from {ALL_DOMAINS}")

    inverse_hash_before = _sha256(args.bundle)
    delay_hash_before = _sha256(args.delay_checkpoint) if args.delay_checkpoint else None
    forward_hash_before = _sha256(args.checkpoint)
    inverse = RemixerRuntime(args.bundle, target=torch.device("cpu"))
    if args.delay_checkpoint is not None:
        inverse.delay.load_state_dict(
            torch.load(args.delay_checkpoint, map_location="cpu", weights_only=True)
        )
        inverse.delay.eval()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    forward = DelayForwardRenderer(fir_taps=int(payload["fir_taps"]))
    forward.load_state_dict(payload["state_dict"], strict=True)
    forward.eval()
    sources = dry_sources(args.corpus, "valid")
    total_samples = args.samples_per_domain * len(domains)
    controls = _latin_hypercube(total_samples, 3, 20261101)
    reports = {}
    cursor = 0
    for domain in domains:
        totals = defaultdict(float)
        peak_ratios = []
        parameter_errors = {"time_ms": [], "feedback": [], "mix": []}
        examples = []
        for index in range(args.samples_per_domain):
            dry = _waveform(sources[(cursor + index) % len(sources)])
            peak = max(float(np.max(np.abs(dry))), 1.0e-5)
            dry = np.asarray(dry * (0.22 / peak), dtype=np.float32)
            normalized = controls[cursor + index]
            truth = Delay(
                40.0 * 25.0 ** float(normalized[0]),
                float(normalized[1]) * 0.9,
                float(normalized[2]) * 0.7,
            )
            spec = ChainSpec((truth,))
            wet = (
                render_pedalboard_chain(dry, spec, INVERSE_RATE)
                if domain == "pedalboard"
                else render_chain(dry, spec, INVERSE_RATE, domain)
            )
            inverse_report = inverse.infer(dry, wet, ("delay",))
            estimated = _decoded_delay(inverse_report)
            dry_forward = _forward_audio(dry)
            wet_forward = _forward_audio(wet)
            with torch.inference_mode():
                dry_tensor = torch.from_numpy(dry_forward).unsqueeze(0)
                oracle = forward(
                    dry_tensor,
                    torch.from_numpy(normalized_delay_controls(truth)).unsqueeze(0),
                ).squeeze(0).numpy()
                closed = forward(
                    dry_tensor,
                    torch.from_numpy(normalized_delay_controls(estimated)).unsqueeze(0),
                ).squeeze(0).numpy()
            baseline_esr, baseline_mae = _errors(dry_forward, wet_forward)
            oracle_esr, oracle_mae = _errors(oracle, wet_forward)
            closed_esr, closed_mae = _errors(closed, wet_forward)
            totals["baseline_esr"] += baseline_esr
            totals["baseline_mae"] += baseline_mae
            totals["oracle_esr"] += oracle_esr
            totals["oracle_mae"] += oracle_mae
            totals["closed_loop_esr"] += closed_esr
            totals["closed_loop_mae"] += closed_mae
            peak_ratios.append(
                float(np.max(np.abs(closed)) / max(float(np.max(np.abs(wet_forward))), 1.0e-6))
            )
            parameter_errors["time_ms"].append(abs(estimated.time_ms - truth.time_ms))
            parameter_errors["feedback"].append(abs(estimated.feedback - truth.feedback))
            parameter_errors["mix"].append(abs(estimated.mix - truth.mix))
            examples.append(
                {
                    "truth": {"time_ms": truth.time_ms, "feedback": truth.feedback, "mix": truth.mix},
                    "estimated": {
                        "time_ms": estimated.time_ms,
                        "feedback": estimated.feedback,
                        "mix": estimated.mix,
                    },
                    "baseline_esr": baseline_esr,
                    "oracle_esr": oracle_esr,
                    "closed_loop_esr": closed_esr,
                }
            )
        cursor += args.samples_per_domain
        count = args.samples_per_domain
        metrics = {name: value / count for name, value in totals.items()}
        metrics["oracle_esr_improvement"] = 1.0 - metrics["oracle_esr"] / max(
            metrics["baseline_esr"], 1.0e-12
        )
        metrics["closed_loop_esr_improvement"] = 1.0 - metrics["closed_loop_esr"] / max(
            metrics["baseline_esr"], 1.0e-12
        )
        metrics["closed_loop_mae_improvement"] = 1.0 - metrics["closed_loop_mae"] / max(
            metrics["baseline_mae"], 1.0e-12
        )
        ordered_peaks = sorted(peak_ratios)
        metrics["peak_ratio_p95"] = ordered_peaks[max(0, math.ceil(0.95 * count) - 1)]
        metrics["worst_peak_ratio"] = ordered_peaks[-1]
        metrics["parameter_error"] = {
            name: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
            }
            for name, values in parameter_errors.items()
        }
        metrics["passed"] = bool(
            metrics["oracle_esr"] < 0.9 * metrics["baseline_esr"]
            and metrics["closed_loop_esr"] < 0.9 * metrics["baseline_esr"]
            and metrics["closed_loop_mae"] < 0.9 * metrics["baseline_mae"]
            and metrics["peak_ratio_p95"] <= 1.25
            and metrics["worst_peak_ratio"] <= 1.75
        )
        reports[domain] = {**metrics, "examples": examples}
        print(json.dumps({"domain": domain, **metrics}, sort_keys=True))
    inverse_hash_after = _sha256(args.bundle)
    delay_hash_after = _sha256(args.delay_checkpoint) if args.delay_checkpoint else None
    forward_hash_after = _sha256(args.checkpoint)
    artifacts_immutable = bool(
        inverse_hash_before == inverse_hash_after
        and delay_hash_before == delay_hash_after
        and forward_hash_before == forward_hash_after
    )
    passed = all(values["passed"] for values in reports.values()) and artifacts_immutable
    report = {
        "schema": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "samples_per_domain": args.samples_per_domain,
        "inverse_bundle": str(args.bundle),
        "inverse_bundle_sha256_before": inverse_hash_before,
        "inverse_bundle_sha256_after": inverse_hash_after,
        "delay_checkpoint_override": str(args.delay_checkpoint) if args.delay_checkpoint else None,
        "delay_checkpoint_override_sha256_before": delay_hash_before,
        "delay_checkpoint_override_sha256_after": delay_hash_after,
        "forward_checkpoint": str(args.checkpoint),
        "forward_checkpoint_sha256_before": forward_hash_before,
        "forward_checkpoint_sha256_after": forward_hash_after,
        "model_artifacts_immutable_during_audit": artifacts_immutable,
        "inverse_sample_rate": INVERSE_RATE,
        "forward_sample_rate": FORWARD_RATE,
        "analysis_frames": INVERSE_SAMPLES,
        "analysis_copy_resampling_only": True,
        "runtime_automatic_normalization": False,
        "source_files_read_only": True,
        "physical_audio_devices_used": False,
        "locked_tele_test_opened": False,
        "challenge_renderer_opened": "challenge" in domains,
        "challenge_result_used_for_training": False,
        "validation": reports,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": passed}, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
