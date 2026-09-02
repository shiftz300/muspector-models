#!/usr/bin/env python3
"""Freeze the evidence and architecture decision for the ASRNN RAT pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing ASRNN evidence: {path}")
    return json.loads(path.read_text())


def summarize(workspace: Path) -> dict:
    runs = workspace / "remix/runs"
    audit = _read(runs / "asrnn-rat-adapter-pilot/data-audit.json")
    adapter = _read(runs / "asrnn-rat-adapter-pilot-tuning-256-e12/metrics.json")
    direct = _read(runs / "asrnn-rat-direct-pilot-full-e6/metrics.json")
    official = _read(runs / "asrnn-rat-official-baseline/metrics.json")
    stable = _read(runs / "asrnn-rat-official-stable-baseline/metrics.json")
    stable_pilot = _read(runs / "asrnn-rat-stable-pilot/metrics.json")
    runtime = _read(runs / "asrnn-rat-stable-pilot/runtime-benchmark.json")
    stable_best = min(
        stable["checkpoints"], key=lambda row: row["strict_mean_per_file_esr"]
    )
    return {
        "schema": 1,
        "phase": "phase-2-real-hardware-rat-architecture-pilot",
        "status": "accepted-internal-noncommercial-pilot",
        "phase_2_complete_within_scope": True,
        "dataset": {
            "files": audit["files"],
            "duration_hours": audit["duration_hours"],
            "license": audit["license"],
            "exact_dry_duplicates_across_official_splits": audit[
                "official_train_eval_exact_dry_duplicates"
            ],
        },
        "local_adapter": {
            "accepted": adapter["accepted"],
            "official_eval": adapter["official_eval"],
            "decision": "reject residual adapter; generic Drive level semantics dominate",
        },
        "local_direct_model": {
            "accepted": direct["accepted"],
            "official_eval": direct["official_eval"],
            "runtime": direct["runtime"],
            "decision": "retain as training scaffold, not as a promoted checkpoint",
        },
        "official_unregularized_upper_bound": {
            "best_global_esr": official["best_official_global_esr"],
            "decision": "accuracy reference only; rejected for nonzero silence output",
        },
        "official_stable_upper_bound": stable_best,
        "canonical_stable_pilot": {
            "accepted": stable_pilot["accepted"],
            "checkpoint": stable_pilot["checkpoint"],
            "checkpoint_sha256": stable_pilot["checkpoint_sha256"],
            "official_eval": stable_pilot["official_eval"],
            "runtime_benchmark": runtime,
            "decision": "promote for internal non-commercial offline model work only",
        },
        "architecture_decision": {
            "runtime": "independent standard-operator stable checkpoint runtime",
            "shape": "4 layers x 8 states",
            "training_origin": "public GFB-trained stable checkpoint under CC-BY-NC-4.0",
            "reason": (
                "the converted checkpoint reproduces the stable reference metrics, exact "
                "static and dynamic-control silence, and stateful streaming"
            ),
            "copy_official_gpl_code_into_apache_project": False,
        },
        "more_public_audio_needed_now": False,
        "next_data_gate": (
            "collect a separately licensed, session-disjoint Capture Pack and independently "
            "train a release-compatible model with calibrate/valid/locked-final evidence"
        ),
        "quality_contract": {
            "source_files_read_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
        },
        "ui_integration_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("remix/runs/asrnn-rat-phase2-summary.json")
    )
    args = parser.parse_args()
    report = summarize(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
