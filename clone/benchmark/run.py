"""Run the first Structural-OOD estimator comparison."""

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
from ..models.causal_lstm import CausalLSTM48
from ..models.structured_residual import StructuredPlusResidual2k
from ..sysid.estimators import LinearEstimator, ParallelHammersteinEstimator, WienerHammersteinEstimator
from ..sysid.state_bank import DynamicGrayBoxEstimator
from .synthetic_dut import (
    FRAMES,
    TOPOLOGIES,
    make_history_pairs,
    make_history_training_observations,
    make_observations,
    write_manifest,
)


def _git_revision(workspace: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _fit_model(name: str, rows: list[dict], device: str):
    if name == "linear":
        return LinearEstimator().fit(rows)
    if name == "parallel_hammerstein":
        return ParallelHammersteinEstimator().fit(rows)
    if name == "wiener_hammerstein":
        return WienerHammersteinEstimator(device=device).fit(rows)
    if name == "dynamic_gray_box":
        return DynamicGrayBoxEstimator().fit(rows)
    if name == "structured_plus_residual_2k":
        return StructuredPlusResidual2k(device=device).fit(rows)
    if name == "causal_lstm_4.8k":
        return CausalLSTM48(device=device).fit(rows)
    raise ValueError(name)


def _public(rows) -> list[dict]:
    return [row.public() for row in rows]


def run(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace benchmark output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(output / "dataset-manifest.json")
    splits = {name: make_observations(name) for name in manifest["splits"]}
    models = (
        "linear", "parallel_hammerstein", "wiener_hammerstein",
        "dynamic_gray_box", "causal_lstm_4.8k",
    )
    all_results: dict[str, dict] = {}
    topology_models: dict[tuple[str, str], object] = {}
    fit_records = {}

    for model_name in models:
        print(json.dumps({"stage": "fit", "model": model_name}), flush=True)
        fit_started = time.perf_counter()
        for topology in TOPOLOGIES:
            fit_rows = [row for row in splits["fit"] if row.topology == topology]
            public_rows = _public(fit_rows)
            if model_name == "dynamic_gray_box":
                public_rows.extend(_public(make_history_training_observations((topology,))))
            estimator = _fit_model(model_name, public_rows, args.device)
            topology_models[(model_name, topology)] = estimator
            print(json.dumps({
                "model": model_name, "topology": topology,
                "parameters": estimator.parameters,
                "residual_parameters": getattr(estimator, "residual_parameters", None),
                "fit_seconds": estimator.fit_seconds,
            }), flush=True)
        fit_records[model_name] = {
            "fit_seconds_all_topologies": float(time.perf_counter() - fit_started),
            "parameters": int(topology_models[(model_name, "A")].parameters),
        }

    # Same-DUT held-out reports.
    for model_name in models:
        category_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
        for split in ("source_ood", "level_ood", "control_ood"):
            for row in splits[split]:
                estimator = topology_models[(model_name, row.topology)]
                category_rows[split].append(one_example(
                    estimator.predict(row.x, row.control), row.y,
                ))
        # Structural topology transfer is intentionally fit on A only, then
        # evaluated on unseen D/E.  No model receives the hidden topology ID.
        for row in splits["topology_ood"]:
            if row.topology not in ("D", "E"):
                continue
            estimator = topology_models[(model_name, "A")]
            category_rows["topology_ood"].append(one_example(
                estimator.predict(row.x, row.control), row.y,
            ))
        runtime = benchmark_runtime(topology_models[(model_name, "A")], FRAMES, manifest["sample_rate"])
        all_results[model_name] = {
            "parameters": fit_records[model_name]["parameters"],
            "fit_seconds_all_topologies": fit_records[model_name]["fit_seconds_all_topologies"],
            "cpu_rtf": runtime,
            "categories": {name: summarize(rows) for name, rows in sorted(category_rows.items())},
        }
        print(json.dumps({
            "stage": "evaluate", "model": model_name,
            "cpu_rtf": runtime,
            "categories": {name: value["absolute_esr"] for name, value in all_results[model_name]["categories"].items()},
        }), flush=True)

    history_rows = []
    for pair in make_history_pairs():
        quiet = pair.y_quiet[pair.payload_start:]
        excited = pair.y_excited[pair.payload_start:]
        delta = np.asarray(excited - quiet, dtype=np.float64)
        history_rows.append({
            "topology": pair.topology,
            "mean_absolute_output_delta": float(np.mean(np.abs(delta))),
            "peak_output_delta": float(np.max(np.abs(delta))),
            "relative_output_delta": float(
                np.mean(np.abs(delta)) / max(float(np.mean(np.abs(quiet))), 1.0e-8)
            ),
        })

    payload = {
        "schema": 1,
        "status": "diagnostic-benchmark",
        "benchmark": manifest,
        "estimator_contract": {
            "topology_input": False,
            "renderer_identity_input": False,
            "analytic_inverse_used": False,
            "physical_audio_devices_used": False,
            "absolute_metrics": True,
        },
        "fit": {
            "device": args.device,
            "git_revision": _git_revision(workspace),
            "models": all_results,
        },
        "results": all_results,
        "history_dependence": {
            "pairs": history_rows,
            "by_topology": {
                topology: {
                    "mean_absolute_output_delta": float(np.mean([
                        row["mean_absolute_output_delta"]
                        for row in history_rows if row["topology"] == topology
                    ])),
                    "relative_output_delta": float(np.mean([
                        row["relative_output_delta"]
                        for row in history_rows if row["topology"] == topology
                    ])),
                }
                for topology in TOPOLOGIES
            },
            "interpretation": "nonzero payload delta after identical payload proves history dependence of the hidden DUT",
        },
        "provenance": {
            "product_audio": False,
            "external_audio": False,
            "license": "repository-owned synthetic benchmark; no external audio",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "first round uses synthetic hidden DUTs rather than an external hardware capture",
            "topology transfer is intentionally a model-fit-on-A to D/E stress test",
            "the residual is intentionally limited to the approximately 2k-parameter budget",
            "metrics are not a product promotion gate",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    registry_path = workspace / "experiments.json"
    registry = []
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
    experiment_id = output.name
    registry.append({
        "id": experiment_id,
        "git_commit": payload["fit"]["git_revision"],
        "data_manifest_sha256": hashlib.sha256((output / "dataset-manifest.json").read_bytes()).hexdigest(),
        "algorithm": list(models),
        "fit_split": "fit",
        "calibration_split": None,
        "validation_split": ["source_ood", "level_ood", "control_ood", "topology_ood"],
        "locked": False,
        "status": "diagnostic",
        "metrics": str(output / "metrics.json"),
    })
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(output), "registry": str(registry_path)}, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/clone-structural-ood-v1"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    run(args)


if __name__ == "__main__":
    main()
