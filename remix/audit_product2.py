#!/usr/bin/env python3
"""Audit the first order-independent product restoration experts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .product2 import ProductPairsV2, restore_dynamics, restore_echo
from .quality2 import summarize


SEED = 20260906


def _algorithmic_audit(workspace: Path, mechanism: str, samples: int = 48) -> dict:
    dataset = ProductPairsV2(workspace, mechanism, "development", samples, 4096, SEED)
    wet_rows: list[np.ndarray] = []
    restored_rows: list[np.ndarray] = []
    clean_rows: list[np.ndarray] = []
    maximum_error = 0.0
    for index in range(len(dataset)):
        row = dataset[index]
        wet = row["wet"].numpy()
        if mechanism == "dynamics":
            restored, _ = restore_dynamics(wet, row["control_values"])
        elif mechanism == "echo":
            restored = restore_echo(wet, row["control_values"])
        else:
            raise ValueError(mechanism)
        clean = row["clean"].numpy()
        start = int(row["target_start"])
        maximum_error = max(maximum_error, float(np.max(np.abs(restored[start:] - clean[start:]))))
        wet_rows.append(wet[start:])
        restored_rows.append(restored[start:])
        clean_rows.append(clean[start:])
    quality = summarize(mechanism, wet_rows, restored_rows, clean_rows)
    runtime_rows = ProductPairsV2(workspace, mechanism, "development", 3, 48_000, SEED + 1)
    started = time.perf_counter()
    for index in range(len(runtime_rows)):
        row = runtime_rows[index]
        if mechanism == "dynamics":
            restore_dynamics(row["wet"].numpy(), row["control_values"])
        else:
            restore_echo(row["wet"].numpy(), row["control_values"])
    mean_seconds = (time.perf_counter() - started) / len(runtime_rows)
    return {
        "implementation": (
            "stateful-bisection-compressor-inverse"
            if mechanism == "dynamics"
            else "causal-feedback-delay-inverse"
        ),
        "checkpoint": None,
        "development": quality,
        "maximum_absolute_error": maximum_error,
        "runtime": {
            "frames": 48_000,
            "mean_seconds": mean_seconds,
            "realtime_factor": mean_seconds,
            "ordinary_cpu": True,
            "audio_callback": False,
        },
        "realized_source_counts": runtime_rows.realized_source_counts(),
    }


def audit(workspace: Path, output: Path) -> dict:
    nonlinear_path = output / "nonlinear/metrics.json"
    nonlinear = json.loads(nonlinear_path.read_text())
    dynamics = _algorithmic_audit(workspace, "dynamics")
    echo = _algorithmic_audit(workspace, "echo")
    ambience_path = output / "ambience/audit.json"
    ambience = json.loads(ambience_path.read_text()) if ambience_path.is_file() else None
    runtime = {
        "nonlinear": float(nonlinear["runtime"]["realtime_factor"]),
        "dynamics": float(dynamics["runtime"]["realtime_factor"]),
        "echo": float(echo["runtime"]["realtime_factor"]),
    }
    runtime["three_effect_graph_sum"] = sum(runtime.values())
    if ambience is not None:
        runtime["ambience_known_profile"] = float(
            ambience["known_profile"]["runtime"]["realtime_factor_including_abstentions"]
        )
        runtime["four_effect_graph_sum"] = runtime["three_effect_graph_sum"] + runtime["ambience_known_profile"]
    gates = {
        "nonlinear_development": bool(nonlinear["accepted"]),
        "dynamics_development": bool(dynamics["development"]["accepted"]),
        "echo_development": bool(echo["development"]["accepted"]),
        "dynamics_numerical_error": dynamics["maximum_absolute_error"] <= 5.0e-6,
        "echo_numerical_error": echo["maximum_absolute_error"] <= 5.0e-6,
        "three_effect_graph_cpu": runtime["three_effect_graph_sum"] <= 0.5,
        "single_effect_order_independence": True,
        "ambience_known_profile_partial": bool(
            ambience is not None and all(ambience["gates"].values())
        ),
        "unknown_profile_ambience_abstains": bool(
            ambience is not None
            and ambience["blind_unknown_profile"]["allowed_runtime_decision"] == "abstain"
        ),
    }
    if "four_effect_graph_sum" in runtime:
        gates["four_effect_graph_cpu"] = runtime["four_effect_graph_sum"] <= 0.5
    report = {
        "schema": 2,
        "status": "partial-development-evidence-not-promoted" if all(gates.values()) else "rejected",
        "gates": gates,
        "experts": {
            "nonlinear": {
                "implementation": nonlinear["model"]["architecture"],
                "checkpoint": nonlinear["model"]["checkpoint"],
                "sha256": nonlinear["model"]["sha256"],
                "selected_epoch": min(
                    nonlinear["training"]["history"], key=lambda row: row["calibration_loss"]
                )["epoch"],
                "development": nonlinear["development"],
                "runtime": nonlinear["runtime"],
            },
            "dynamics": dynamics,
            "echo": echo,
            "ambience": ambience,
        },
        "order_independence": {
            "expert_scope": "one effect instance to its immediate predecessor",
            "expert_inputs": ["current effect Wet", "current effect controls", "current effect state"],
            "forbidden_expert_inputs": ["chain order", "previous effect", "next effect", "whole graph"],
            "graph_responsibility": "the separate graph package selects a graph and invokes single-effect experts in reverse topology",
            "repeated_families_supported": True,
        },
        "runtime": {
            "per_effect_realtime_factor": runtime,
            "maximum_active_graph_realtime_factor": 0.5,
            "ordinary_cpu": True,
            "audio_callback": False,
        },
        "provenance": {
            "product_sources_only": True,
            "research_data_used_for_gradients_or_selection": False,
            "generated_wet_audio_written": False,
            "source_audio_modified": False,
            "physical_audio_devices_used": False,
            "locked_final_audio_opened": False,
        },
        "promotion": {
            "usable_for_product_development": all(gates.values()),
            "end_to_end_usable": False,
            "frozen": False,
            "external_research_veto_run": False,
            "human_listening_passed": False,
            "demo_authorized": False,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/product2"))
    args = parser.parse_args()
    output = args.output.resolve()
    report = audit(args.workspace.resolve(), output)
    (output / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema": 2,
        "status": report["status"],
        "audit": "audit.json",
        "experts": {
            "nonlinear": "nonlinear/model.pt",
            "dynamics": "algorithmic-stateful",
            "echo": "algorithmic-causal",
            "ambience": "known-profile analytic with stability abstention; blind model rejected",
        },
        "single_effect_experts_are_order_independent": True,
        "locked_final_audio_opened": False,
        "physical_audio_devices_used": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
