#!/usr/bin/env python3
"""Freeze the Phase-3 RAT inverse-control evidence and decision boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONTROL_GATES = {
    "volume": {"mae_normalized": 0.06, "p95_normalized": 0.18},
    "distortion": {"mae_normalized": 0.08, "p95_normalized": 0.22},
    "tone": {"mae_normalized": 0.08, "p95_normalized": 0.22},
}


def _read(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing ASRNN inverse evidence: {path}")
    return json.loads(path.read_text())


def control_decisions(report: dict) -> dict:
    """Evaluate semantic controls without treating quiet upstream knobs as labels."""
    official_eval = report["official_eval"]
    decisions = {}
    for control in ("volume", "distortion", "tone"):
        if control == "volume":
            metrics = official_eval["per_control"][control]
            population = "all official-eval examples"
        else:
            metrics = official_eval.get("audible_distortion_tone", {}).get(control)
            population = "audible official-eval examples only"
        gate = CONTROL_GATES[control]
        passed = bool(
            metrics
            and metrics["mae_normalized"] <= gate["mae_normalized"]
            and metrics["p95_normalized"] <= gate["p95_normalized"]
        )
        decisions[control] = {
            "passed": passed,
            "population": population,
            "gate": gate,
            "metrics": metrics,
        }
    return decisions


def _candidate(name: str, path: Path, report: dict) -> dict:
    decisions = control_decisions(report)
    semantic_error = sum(
        decision["metrics"]["mae_normalized"]
        for decision in decisions.values()
        if decision["passed"]
    )
    return {
        "name": name,
        "report": str(path),
        "accepted_controls": [
            name for name, decision in decisions.items() if decision["passed"]
        ],
        "semantic_control_decisions": decisions,
        "official_eval_macro_mae_normalized": report["official_eval"][
            "macro_mae_normalized"
        ],
        "closed_loop_mean_recovered_esr": report["closed_loop"][
            "mean_recovered_control_esr"
        ],
        "closed_loop_mae_improvement_vs_bypass": report["closed_loop"][
            "recovered_vs_bypass_mae_improvement"
        ],
        "selection_key": [
            -sum(decision["passed"] for decision in decisions.values()),
            semantic_error,
            report["closed_loop"]["mean_recovered_control_esr"],
        ],
    }


def summarize(workspace: Path) -> dict:
    runs = workspace / "remix/runs"
    paths = {
        "learned-feature-inverse": runs / "asrnn-rat-inverse-pilot/metrics.json",
        "hybrid-refinement-4096": runs
        / "asrnn-rat-inverse-pilot/refinement.json",
        "hybrid-refinement-8192": runs
        / "asrnn-rat-inverse-pilot/refinement-8192.json",
        "spectral-feature-inverse": runs
        / "asrnn-rat-spectral-inverse-pilot/metrics.json",
    }
    candidates = [
        _candidate(name, path.relative_to(workspace), _read(path))
        for name, path in paths.items()
    ]
    selected = min(candidates, key=lambda row: row["selection_key"])
    decisions = selected["semantic_control_decisions"]
    precise = [name for name, decision in decisions.items() if decision["passed"]]
    unresolved = [name for name, decision in decisions.items() if not decision["passed"]]
    selected_report = _read(workspace / selected["report"])
    quiet_examples = selected_report["official_eval"].get(
        "unidentifiable_quiet_examples", 0
    )
    return {
        "schema": 1,
        "phase": "phase-3-real-hardware-rat-inverse-control-pilot",
        "status": "partial-accepted-tone-unresolved",
        "phase_3_evaluation_complete": True,
        "phase_3_exact_all_controls": False,
        "selected_candidate": selected,
        "candidate_comparison": candidates,
        "precise_controls": precise,
        "unresolved_controls": unresolved,
        "identifiability": {
            "quiet_wet_peak_threshold": 1.0e-3,
            "quiet_official_eval_examples": quiet_examples,
            "official_eval_examples": selected_report["official_eval"]["examples"],
            "policy": (
                "Volume remains observable at quiet output, but upstream Distortion and "
                "Tone are not claimed when Wet is below the identifiability threshold."
            ),
        },
        "closed_loop_interpretation": (
            "Low reconstruction error proves that the recovered setting is useful for "
            "render matching. It does not prove semantic knob truth because controls can "
            "compensate for forward-model error."
        ),
        "decision": {
            "volume": "accept precise recovery on all official-eval examples",
            "distortion": "accept precise recovery only for audible/identifiable Wet",
            "tone": "reject exact recovery and freeze as unresolved",
            "next_model_work": (
                "retain the selected hybrid path; do not continue blind tuning on this "
                "dataset or connect it to UI"
            ),
        },
        "additional_public_audio_needed_now": False,
        "tone_data_needed_for_resolution": (
            "a separately licensed, session-disjoint capture pack with controlled Tone "
            "sweeps, multiple source programs, and deliberately audible output levels"
        ),
        "quality_contract": {
            "source_files_read_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
        },
        "license_scope": "internal non-commercial development only (CC-BY-NC-4.0 data)",
        "locked_final_opened": False,
        "ui_integration_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("remix/runs/asrnn-rat-inverse-summary.json"),
    )
    args = parser.parse_args()
    report = summarize(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
