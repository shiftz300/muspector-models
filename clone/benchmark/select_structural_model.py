"""Select the simplest structured clone on calibration, then test Structural-OOD.

This is the complexity-controlled escalation stage of the black-box forward
benchmark.  Candidate models see only public input/output observations and the
effect control.  The hidden topology is used by the benchmark harness to fit
separate known-DUT experts and for post-hoc reporting; it is never passed to an
estimator.

Development/OOD rows are intentionally not touched until calibration has
chosen one candidate per known topology.  Selection uses the mean calibration
absolute ESR and keeps the lowest-parameter candidate within a transparent
relative tolerance of the best score.
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
import torch

from ..evaluation.evaluate import benchmark_runtime, one_example, summarize
from ..sysid.estimators import (
    LinearEstimator,
    ParallelHammersteinEstimator,
    WienerHammersteinEstimator,
)
from ..sysid.state_bank import DynamicGrayBoxEstimator
from .synthetic_dut import (
    FRAMES,
    OOD_SOURCES,
    TOPOLOGIES,
    TRAIN_SOURCES,
    make_history_pairs,
    make_history_training_observations,
    make_observations,
    write_manifest,
)


CANDIDATES = (
    "linear",
    "parallel_hammerstein",
    "wiener_hammerstein",
    "dynamic_gray_box",
)
SELECTION_FIT_SOURCES = TRAIN_SOURCES[:-2]
SELECTION_CALIBRATION_SOURCES = TRAIN_SOURCES[-2:]
OOD_SPLITS = ("source_ood", "level_ood", "control_ood")


def _git_revision(workspace: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _public(rows) -> list[dict]:
    return [row.public() for row in rows]


def _fit_model(name: str, rows: list[dict], device: str):
    if name == "linear":
        return LinearEstimator().fit(rows)
    if name == "parallel_hammerstein":
        return ParallelHammersteinEstimator().fit(rows)
    if name == "wiener_hammerstein":
        return WienerHammersteinEstimator(device=device).fit(rows)
    if name == "dynamic_gray_box":
        return DynamicGrayBoxEstimator().fit(rows)
    raise ValueError(name)


def _rows_for_sources(rows, topology: str, source_ids) -> list:
    source_ids = set(source_ids)
    return [
        row for row in rows
        if row.topology == topology and row.source_id in source_ids
    ]


def _fit_records(
    fit_rows,
    topology: str,
    device: str,
) -> tuple[dict[str, object], dict[str, dict]]:
    source_rows = _rows_for_sources(fit_rows, topology, SELECTION_FIT_SOURCES)
    if not source_rows:
        raise ValueError(f"no fit rows for topology {topology}")
    estimators: dict[str, object] = {}
    records: dict[str, dict] = {}
    for name in CANDIDATES:
        public_rows = _public(source_rows)
        if name == "dynamic_gray_box":
            history_rows = [
                row for row in make_history_training_observations((topology,))
                if row.source_id in SELECTION_FIT_SOURCES
            ]
            public_rows.extend(_public(history_rows))
        started = time.perf_counter()
        estimator = _fit_model(name, public_rows, device)
        fit_seconds = float(time.perf_counter() - started)
        estimators[name] = estimator
        records[name] = {
            "parameters": int(estimator.parameters),
            "fit_seconds": fit_seconds,
            "fit_rows": len(public_rows),
        }
        print(json.dumps({
            "stage": "fit",
            "topology": topology,
            "model": name,
            "parameters": int(estimator.parameters),
            "fit_seconds": fit_seconds,
            "fit_rows": len(public_rows),
        }), flush=True)
    return estimators, records


def _calibration_report(estimators: dict[str, object], rows: list) -> dict[str, dict]:
    reports = {}
    for name, estimator in estimators.items():
        metrics = [
            one_example(estimator.predict(row.x, row.control), row.y)
            for row in rows
        ]
        reports[name] = summarize(metrics)
    return reports


def _choose(
    records: dict[str, dict],
    calibration: dict[str, dict],
    tolerance: float,
) -> dict:
    scores = {
        name: float(report["absolute_esr"]["mean"])
        for name, report in calibration.items()
    }
    best_name = min(scores, key=scores.get)
    best_score = scores[best_name]
    threshold = best_score * (1.0 + tolerance)
    eligible = [name for name, score in scores.items() if score <= threshold]
    selected = min(
        eligible,
        key=lambda name: (int(records[name]["parameters"]), scores[name], name),
    )
    return {
        "metric": "absolute_esr.mean",
        "complexity_tolerance": tolerance,
        "best_calibration_model": best_name,
        "best_calibration_score": best_score,
        "eligibility_threshold": threshold,
        "eligible_models": sorted(eligible),
        "selected_model": selected,
        "selected_parameters": int(records[selected]["parameters"]),
        "scores": scores,
    }


def _history_report() -> dict:
    rows = []
    for pair in make_history_pairs():
        quiet = pair.y_quiet[pair.payload_start:]
        excited = pair.y_excited[pair.payload_start:]
        delta = np.asarray(excited - quiet, dtype=np.float64)
        rows.append({
            "topology": pair.topology,
            "mean_absolute_output_delta": float(np.mean(np.abs(delta))),
            "peak_output_delta": float(np.max(np.abs(delta))),
            "relative_output_delta": float(
                np.mean(np.abs(delta)) / max(float(np.mean(np.abs(quiet))), 1.0e-8)
            ),
        })
    by_topology = defaultdict(list)
    for row in rows:
        by_topology[row["topology"]].append(row)
    return {
        "pairs": rows,
        "by_topology": {
            topology: {
                "mean_absolute_output_delta": float(np.mean([
                    row["mean_absolute_output_delta"] for row in values
                ])),
                "relative_output_delta": float(np.mean([
                    row["relative_output_delta"] for row in values
                ])),
            }
            for topology, values in sorted(by_topology.items())
        },
        "interpretation": "same payload after different prefixes measures hidden DUT history dependence",
    }


def _evaluate_selected(
    selected_estimators: dict[str, object],
    model_a: object,
    splits: dict[str, list],
    sample_rate: int,
) -> dict:
    same_dut = {}
    for split in OOD_SPLITS:
        aggregate = []
        by_topology = defaultdict(list)
        for row in splits[split]:
            estimator = selected_estimators[row.topology]
            metrics = one_example(estimator.predict(row.x, row.control), row.y)
            aggregate.append(metrics)
            by_topology[row.topology].append(metrics)
        same_dut[split] = {
            "aggregate": summarize(aggregate),
            "by_topology": {
                topology: summarize(values)
                for topology, values in sorted(by_topology.items())
            },
        }

    topology_transfer_rows = []
    transfer_by_topology = defaultdict(list)
    for row in splits["topology_ood"]:
        if row.topology not in ("D", "E"):
            continue
        metrics = one_example(model_a.predict(row.x, row.control), row.y)
        topology_transfer_rows.append(metrics)
        transfer_by_topology[row.topology].append(metrics)
    return {
        "same_dut_ood": same_dut,
        "topology_transfer": {
            "fit_topology": "A",
            "evaluated_hidden_topologies": ["D", "E"],
            "aggregate": summarize(topology_transfer_rows),
            "by_topology": {
                topology: summarize(values)
                for topology, values in sorted(transfer_by_topology.items())
            },
        },
        "cpu_rtf_fit_on_A": float(benchmark_runtime(
            model_a, FRAMES, sample_rate,
        )),
    }


def run(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace benchmark output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(output / "dataset-manifest.json")
    manifest.update({
        "selection_fit_sources": list(SELECTION_FIT_SOURCES),
        "selection_calibration_sources": list(SELECTION_CALIBRATION_SOURCES),
        "selection_rule": "lowest parameter count within tolerance of best calibration absolute ESR",
        "selection_complexity_tolerance": args.complexity_tolerance,
    })
    (output / "dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    # Only the declared fit/calibration portions of the fit grid are made
    # before model selection.  OOD rows are materialized after selection.
    fit_grid = make_observations("fit")
    calibration_rows = [
        row for row in fit_grid
        if row.source_id in SELECTION_CALIBRATION_SOURCES
    ]
    selected_estimators: dict[str, object] = {}
    topology_records = {}
    for topology in TOPOLOGIES:
        estimators, records = _fit_records(fit_grid, topology, args.device)
        topology_calibration_rows = [
            row for row in calibration_rows if row.topology == topology
        ]
        calibration = _calibration_report(estimators, topology_calibration_rows)
        choice = _choose(records, calibration, args.complexity_tolerance)
        selected_name = choice["selected_model"]
        selected_estimators[topology] = estimators[selected_name]
        topology_records[topology] = {
            "candidates": records,
            "calibration": calibration,
            "selection": choice,
        }
        print(json.dumps({
            "stage": "select",
            "topology": topology,
            "selected_model": selected_name,
            "selected_parameters": choice["selected_parameters"],
            "calibration_scores": choice["scores"],
        }), flush=True)

    # No OOD row is read until every known-topology choice is frozen.
    splits = {split: make_observations(split) for split in OOD_SPLITS + ("topology_ood",)}
    selected = _evaluate_selected(
        selected_estimators,
        selected_estimators["A"],
        splits,
        manifest["sample_rate"],
    )
    print(json.dumps({
        "stage": "evaluate_selected",
        "models": {
            topology: record["selection"]["selected_model"]
            for topology, record in topology_records.items()
        },
        "same_dut_esr": {
            split: report["aggregate"]["absolute_esr"]["mean"]
            for split, report in selected["same_dut_ood"].items()
        },
        "topology_transfer_esr": selected["topology_transfer"]["aggregate"]["absolute_esr"]["mean"],
        "cpu_rtf_fit_on_A": selected["cpu_rtf_fit_on_A"],
    }, indent=2), flush=True)

    payload = {
        "schema": 1,
        "status": "diagnostic-structural-complexity-selection",
        "benchmark": manifest,
        "selection_contract": {
            "estimator_input": ["x", "y for fit only", "control"],
            "topology_input": False,
            "renderer_identity_input": False,
            "analytic_inverse_used": False,
            "physical_audio_devices_used": False,
            "absolute_metrics": True,
            "fit_sources": list(SELECTION_FIT_SOURCES),
            "calibration_sources": list(SELECTION_CALIBRATION_SOURCES),
            "ood_created_only_after_selection": True,
            "bla_report_used_for_selection": False,
            "selection_rule": "lowest parameter count within tolerance of best calibration absolute ESR",
            "complexity_tolerance": args.complexity_tolerance,
        },
        "fit": {
            "device": args.device,
            "git_revision": _git_revision(workspace),
            "candidate_models": list(CANDIDATES),
            "by_topology": topology_records,
        },
        "selected_evaluation": selected,
        "posthoc_diagnostics": {
            "bla_report": "runs/foundation/clone-structural-ood-bla-v1/metrics.json",
            "bla_used_for_selection": False,
            "interpretation": "BLA and history evidence explain why static, nonlinear, or stateful model families may be needed; calibration metrics alone freeze the selection.",
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
            "the selected known-topology experts are not a universal blind topology classifier",
            "topology transfer is intentionally fit on A and evaluated on hidden D/E",
            "metrics are diagnostic and cannot promote a product model or produce a usable inverse",
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
        default=Path("runs/foundation/clone-structural-ood-complexity-v1"),
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--complexity-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    if not 0.0 <= args.complexity_tolerance:
        raise ValueError("complexity tolerance must be non-negative")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    run(args)


if __name__ == "__main__":
    main()
