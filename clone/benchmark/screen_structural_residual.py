"""Test one tiny residual after the bounded structured clone selection.

This is deliberately not a capacity sweep.  A single zero-initialized,
bounded GRU residual is trained on the fit rows with MPS, then accepted only
when calibration mean and p99 both improve over its frozen structured base.
OOD rows are created only after those decisions are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..evaluation.evaluate import benchmark_runtime, one_example, summarize
from ..models.parallel_residual import ParallelPlusResidual2k
from ..sysid.estimators import (
    BoundedParallelHammersteinEstimator,
    ParallelHammersteinEstimator,
)
from .screen_bounded_parallel import _select as select_base
from .synthetic_dut import (
    FRAMES,
    TOPOLOGIES,
    TRAIN_SOURCES,
    make_history_pairs,
    make_observations,
    write_manifest,
)


BASE_CANDIDATES = ("parallel_hammerstein", "bounded_parallel_hammerstein")
FIT_SOURCES = TRAIN_SOURCES[:-2]
CALIBRATION_SOURCES = TRAIN_SOURCES[-2:]
OOD_SPLITS = ("source_ood", "level_ood", "control_ood")


def _git_revision(workspace: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _public(rows) -> list[dict]:
    return [row.public() for row in rows]


def _fit_base(name: str, rows: list[dict]):
    if name == "parallel_hammerstein":
        return ParallelHammersteinEstimator().fit(rows)
    if name == "bounded_parallel_hammerstein":
        return BoundedParallelHammersteinEstimator().fit(rows)
    raise ValueError(name)


def _history_report() -> dict:
    rows = []
    for pair in make_history_pairs():
        quiet = pair.y_quiet[pair.payload_start:]
        excited = pair.y_excited[pair.payload_start:]
        delta = np.asarray(excited - quiet, dtype=np.float64)
        rows.append({
            "topology": pair.topology,
            "relative_output_delta": float(
                np.mean(np.abs(delta)) / max(float(np.mean(np.abs(quiet))), 1.0e-8)
            ),
        })
    return {
        "by_topology": {
            topology: float(np.mean([
                row["relative_output_delta"] for row in rows
                if row["topology"] == topology
            ]))
            for topology in TOPOLOGIES
        },
        "interpretation": "diagnostic only; not used for selection",
    }


def _summary(model, rows: list) -> dict:
    return summarize([
        one_example(model.predict(row.x, row.control), row.y)
        for row in rows
    ])


def run(args: argparse.Namespace) -> dict:
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace benchmark output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(output / "dataset-manifest.json")
    manifest.update({
        "residual_fit_sources": list(FIT_SOURCES),
        "residual_calibration_sources": list(CALIBRATION_SOURCES),
        "base_candidates": list(BASE_CANDIDATES),
        "residual_architecture": "causal GRU(2,24) plus bounded 0.35*tanh output",
        "residual_epochs": args.epochs,
    })
    (output / "dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    fit_grid = make_observations("fit")
    choices = {}
    final_models = {}
    for topology in TOPOLOGIES:
        fit_rows = [
            row for row in fit_grid
            if row.topology == topology and row.source_id in FIT_SOURCES
        ]
        calibration_rows = [
            row for row in fit_grid
            if row.topology == topology and row.source_id in CALIBRATION_SOURCES
        ]
        bases = {}
        fit_records = {}
        for name in BASE_CANDIDATES:
            base = _fit_base(name, _public(fit_rows))
            bases[name] = base
            fit_records[name] = {
                "parameters": int(base.parameters),
                "fit_seconds": float(base.fit_seconds),
                "fit_rows": len(fit_rows),
            }
        base_calibration = {
            name: _summary(model, calibration_rows)
            for name, model in bases.items()
        }
        base_choice = select_base(base_calibration, "tail_guard")
        selected_base_name = base_choice["selected_model"]
        selected_base = bases[selected_base_name]
        residual = ParallelPlusResidual2k(
            selected_base, device=args.device, epochs=args.epochs,
        ).fit(_public(fit_rows))
        residual_calibration = _summary(residual, calibration_rows)
        base_calibration_selected = base_calibration[selected_base_name]
        residual_mean = residual_calibration["absolute_esr"]["mean"]
        base_mean = base_calibration_selected["absolute_esr"]["mean"]
        residual_p99 = residual_calibration["absolute_esr"]["p99"]
        base_p99 = base_calibration_selected["absolute_esr"]["p99"]
        residual_selected = bool(
            residual_mean < base_mean and residual_p99 <= base_p99
        )
        final_models[topology] = residual if residual_selected else selected_base
        choices[topology] = {
            "base_fit": fit_records,
            "base_calibration": base_calibration,
            "base_selection": base_choice,
            "selected_base": selected_base_name,
            "residual": {
                "parameters": int(residual.residual_parameters),
                "total_parameters": int(residual.parameters),
                "device": args.device,
                "epochs": args.epochs,
                "fit_seconds": float(residual.fit_seconds),
                "fit_loss": float(residual.residual_fit_loss),
                "calibration": residual_calibration,
            },
            "final_selection": {
                "selected_model": "parallel_plus_residual_2k" if residual_selected else selected_base_name,
                "residual_selected": residual_selected,
                "rule": "accept residual only when calibration absolute ESR mean strictly improves and p99 does not worsen",
                "base_absolute_esr_mean": base_mean,
                "residual_absolute_esr_mean": residual_mean,
                "base_absolute_esr_p99": base_p99,
                "residual_absolute_esr_p99": residual_p99,
            },
        }
        print(json.dumps({
            "stage": "select_residual",
            "topology": topology,
            "base": selected_base_name,
            "residual_selected": residual_selected,
            "base_mean": base_mean,
            "residual_mean": residual_mean,
            "base_p99": base_p99,
            "residual_p99": residual_p99,
            "residual_fit_seconds": residual.fit_seconds,
        }), flush=True)

    # OOD rows are materialized only after all residual decisions are frozen.
    splits = {
        split: make_observations(split)
        for split in OOD_SPLITS + ("topology_ood",)
    }
    same_dut = {}
    for split in OOD_SPLITS:
        aggregate = []
        by_topology = defaultdict(list)
        for row in splits[split]:
            metrics = one_example(
                final_models[row.topology].predict(row.x, row.control), row.y,
            )
            aggregate.append(metrics)
            by_topology[row.topology].append(metrics)
        same_dut[split] = {
            "aggregate": summarize(aggregate),
            "by_topology": {
                topology: summarize(values)
                for topology, values in sorted(by_topology.items())
            },
        }
    transfer = []
    transfer_by_topology = defaultdict(list)
    model_a = final_models["A"]
    for row in splits["topology_ood"]:
        if row.topology not in ("D", "E"):
            continue
        metrics = one_example(model_a.predict(row.x, row.control), row.y)
        transfer.append(metrics)
        transfer_by_topology[row.topology].append(metrics)
    selected_evaluation = {
        "same_dut_ood": same_dut,
        "topology_transfer": {
            "fit_topology": "A",
            "evaluated_hidden_topologies": ["D", "E"],
            "aggregate": summarize(transfer),
            "by_topology": {
                topology: summarize(values)
                for topology, values in sorted(transfer_by_topology.items())
            },
        },
        "cpu_rtf_fit_on_A": float(benchmark_runtime(model_a, FRAMES, manifest["sample_rate"])),
    }
    print(json.dumps({
        "stage": "evaluate_selected",
        "models": {
            topology: value["final_selection"]["selected_model"]
            for topology, value in choices.items()
        },
        "same_dut_esr": {
            split: value["aggregate"]["absolute_esr"]["mean"]
            for split, value in same_dut.items()
        },
        "topology_transfer_esr": selected_evaluation["topology_transfer"]["aggregate"]["absolute_esr"]["mean"],
        "cpu_rtf_fit_on_A": selected_evaluation["cpu_rtf_fit_on_A"],
    }, indent=2), flush=True)

    payload = {
        "schema": 1,
        "status": "diagnostic-structural-tiny-residual",
        "benchmark": manifest,
        "residual_contract": {
            "estimator_input": ["x", "y for fit only", "control"],
            "topology_input": False,
            "analytic_inverse_used": False,
            "physical_audio_devices_used": False,
            "fit_sources": list(FIT_SOURCES),
            "calibration_sources": list(CALIBRATION_SOURCES),
            "ood_created_only_after_selection": True,
            "single_architecture_no_capacity_sweep": True,
            "residual_zero_initialized": True,
            "residual_output_bounded": True,
            "mps_training_requested": args.device == "mps",
        },
        "fit": {
            "git_revision": _git_revision(workspace),
            "by_topology": choices,
        },
        "selected_evaluation": selected_evaluation,
        "posthoc_diagnostics": {
            "history_dependence": _history_report(),
            "interpretation": "a residual is retained only when it improves calibration mean without worsening calibration p99; otherwise the structured base remains active",
        },
        "provenance": {
            "product_audio": False,
            "external_audio": False,
            "license": "repository-owned synthetic benchmark; no external audio",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "synthetic hidden DUTs are a structural stress test, not a physical-device claim",
            "the residual is not an inverse model and no product checkpoint is produced",
            "topology transfer remains a hidden-structure stress test rather than a blind classifier",
        ],
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    registry_path = workspace / "experiments.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    registry.append({
        "id": output.name,
        "git_commit": payload["fit"]["git_revision"],
        "data_manifest_sha256": hashlib.sha256((output / "dataset-manifest.json").read_bytes()).hexdigest(),
        "algorithm": "tail-aware bounded PH plus one zero-initialized 2k residual",
        "fit_split": "fit sources 0-5",
        "calibration_split": "fit sources 6-7",
        "validation_split": list(OOD_SPLITS) + ["topology_ood"],
        "locked": False,
        "status": "diagnostic",
        "metrics": str(metrics_path),
    })
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "output": str(output),
        "registry": str(registry_path),
    }, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/foundation/clone-structural-ood-residual-v1"),
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    run(args)


if __name__ == "__main__":
    main()
