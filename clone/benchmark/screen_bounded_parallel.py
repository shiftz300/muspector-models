"""Screen a bounded PH replacement against polynomial PH on Structural-OOD.

The experiment targets one diagnosed failure only: polynomial branches can
explode when level-OOD amplitudes exceed the fit range.  Both candidates have
the same seven 64-tap branches and are selected from calibration before any
OOD row is evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..evaluation.evaluate import benchmark_runtime, one_example, summarize
from ..sysid.estimators import (
    BoundedParallelHammersteinEstimator,
    ParallelHammersteinEstimator,
)
from .synthetic_dut import (
    FRAMES,
    TOPOLOGIES,
    TRAIN_SOURCES,
    make_history_pairs,
    make_observations,
    write_manifest,
)


CANDIDATES = ("parallel_hammerstein", "bounded_parallel_hammerstein")
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


def _fit(name: str, rows: list[dict]):
    if name == "parallel_hammerstein":
        return ParallelHammersteinEstimator().fit(rows)
    if name == "bounded_parallel_hammerstein":
        return BoundedParallelHammersteinEstimator().fit(rows)
    raise ValueError(name)


def _public(rows) -> list[dict]:
    return [row.public() for row in rows]


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
        "interpretation": "diagnostic only; not used for model selection",
    }


def _select(
    calibration: dict[str, dict],
    objective: str,
    mean_tolerance: float = 0.05,
) -> dict:
    scores = {
        name: float(report["absolute_esr"]["mean"])
        for name, report in calibration.items()
    }
    if objective == "mean":
        selected = min(scores, key=scores.get)
        eligible = sorted(scores)
        rule = "lowest calibration absolute ESR; equal parameter budget"
    elif objective == "tail_guard":
        best_mean = min(scores.values())
        threshold = best_mean * (1.0 + mean_tolerance)
        eligible = sorted(name for name, score in scores.items() if score <= threshold)
        selected = min(
            eligible,
            key=lambda name: (
                float(calibration[name]["absolute_esr"]["p99"]),
                scores[name],
                name,
            ),
        )
        rule = "lowest calibration absolute-ESR p99 among candidates within 5% of best mean"
    else:
        raise ValueError(f"unknown selection objective: {objective}")
    return {
        "metric": "absolute_esr.mean" if objective == "mean" else "absolute_esr.p99",
        "scores": scores,
        "selected_model": selected,
        "eligible_models": eligible,
        "selection_rule": rule,
        "mean_tolerance": mean_tolerance if objective == "tail_guard" else None,
        "parameter_budget": 448,
    }


def run(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace benchmark output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(output / "dataset-manifest.json")
    manifest.update({
        "screen_fit_sources": list(FIT_SOURCES),
        "screen_calibration_sources": list(CALIBRATION_SOURCES),
        "screen_candidates": list(CANDIDATES),
        "screen_hypothesis": "bounded nonlinear branches reduce level-OOD blow-up from polynomial PH",
        "screen_selection_objective": args.selection_objective,
    })
    (output / "dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    fit_grid = make_observations("fit")
    fit_models = {}
    calibration_models = {}
    fit_records = {}
    selection = {}
    for topology in TOPOLOGIES:
        fit_rows = [
            row for row in fit_grid
            if row.topology == topology and row.source_id in FIT_SOURCES
        ]
        calibration_rows = [
            row for row in fit_grid
            if row.topology == topology and row.source_id in CALIBRATION_SOURCES
        ]
        fit_models[topology] = {}
        fit_records[topology] = {}
        for name in CANDIDATES:
            started = time.perf_counter()
            model = _fit(name, _public(fit_rows))
            fit_models[topology][name] = model
            fit_records[topology][name] = {
                "parameters": int(model.parameters),
                "fit_seconds": float(time.perf_counter() - started),
                "fit_rows": len(fit_rows),
            }
            print(json.dumps({
                "stage": "fit",
                "topology": topology,
                "model": name,
                **fit_records[topology][name],
            }), flush=True)
        calibration = {
            name: summarize([
                one_example(model.predict(row.x, row.control), row.y)
                for row in calibration_rows
            ])
            for name, model in fit_models[topology].items()
        }
        choice = _select(calibration, args.selection_objective)
        selected_name = choice["selected_model"]
        calibration_models[topology] = fit_models[topology][selected_name]
        selection[topology] = {
            "fit": fit_records[topology],
            "calibration": calibration,
            "selection": choice,
        }
        print(json.dumps({
            "stage": "select",
            "topology": topology,
            **choice,
        }), flush=True)

    # OOD rows are materialized only after all five choices are frozen.
    splits = {
        split: make_observations(split)
        for split in OOD_SPLITS + ("topology_ood",)
    }
    same_dut = {}
    for split in OOD_SPLITS:
        aggregate = []
        by_topology = defaultdict(list)
        for row in splits[split]:
            prediction = calibration_models[row.topology].predict(row.x, row.control)
            metrics = one_example(prediction, row.y)
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
    model_a = calibration_models["A"]
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
            topology: value["selection"]["selected_model"]
            for topology, value in selection.items()
        },
        "same_dut_esr": {
            split: value["aggregate"]["absolute_esr"]["mean"]
            for split, value in same_dut.items()
        },
        "topology_transfer_esr": selected_evaluation["topology_transfer"]["aggregate"]["absolute_esr"]["mean"],
    }, indent=2), flush=True)

    payload = {
        "schema": 1,
        "status": "diagnostic-bounded-structural-screen",
        "benchmark": manifest,
        "screen_contract": {
            "estimator_input": ["x", "y for fit only", "control"],
            "topology_input": False,
            "analytic_inverse_used": False,
            "physical_audio_devices_used": False,
            "fit_sources": list(FIT_SOURCES),
            "calibration_sources": list(CALIBRATION_SOURCES),
            "ood_created_only_after_selection": True,
            "bla_or_history_used_for_selection": False,
            "equal_parameter_budget": True,
            "selection_objective": args.selection_objective,
        },
        "fit": {
            "git_revision": _git_revision(workspace),
            "candidate_models": list(CANDIDATES),
            "by_topology": selection,
        },
        "selected_evaluation": selected_evaluation,
        "posthoc_diagnostics": {
            "hypothesis": "replace unbounded polynomial branches with bounded asymmetric saturating branches",
            "history_dependence": _history_report(),
        },
        "provenance": {
            "product_audio": False,
            "external_audio": False,
            "license": "repository-owned synthetic benchmark; no external audio",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "synthetic hidden DUTs are a structural stress test, not a physical-device claim",
            "the bounded basis still has an explicit linear branch and is not a universal extrapolation guarantee",
            "this is forward-clone evidence and does not train or promote a Wet-to-Clean expert",
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
        "algorithm": list(CANDIDATES),
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
        default=Path("runs/foundation/clone-structural-ood-bounded-ph-v1"),
    )
    parser.add_argument(
        "--selection-objective",
        choices=("mean", "tail_guard"),
        default="mean",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
