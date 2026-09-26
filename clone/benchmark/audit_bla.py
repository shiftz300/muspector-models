"""Run the fit-only multi-level BLA and history-dependence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..sysid.bla import grouped_bla
from .synthetic_dut import make_history_pairs, make_observations, write_manifest


def _git_revision(workspace: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _history_report() -> dict:
    rows = []
    for pair in make_history_pairs():
        quiet = pair.y_quiet[pair.payload_start:]
        excited = pair.y_excited[pair.payload_start:]
        delta = np.asarray(excited - quiet, dtype=np.float64)
        rows.append({
            "topology": pair.topology,
            "source_id": pair.topology + ":held-out-history",
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
        "interpretation": "nonzero payload delta after identical payload proves hidden DUT history dependence",
    }


def run(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace BLA audit output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(output / "dataset-manifest.json")
    fit_rows = make_observations("fit")
    bla = grouped_bla(fit_rows, fft_size=args.fft_size)
    history = _history_report()
    payload = {
        "schema": 1,
        "status": "diagnostic-bla-audit",
        "benchmark": manifest,
        "audit_contract": {
            "fit_split_only": True,
            "topology_used_only_for_posthoc_grouping": True,
            "renderer_identity_used_by_estimator": False,
            "analytic_inverse_used": False,
            "physical_audio_devices_used": False,
            "external_audio": False,
            "fft_size": args.fft_size,
        },
        "bla": bla,
        "history_dependence": history,
        "interpretation": {
            "level_shape_drift_is_diagnostic": "large drift means a fixed linear dynamic block is insufficient across input levels",
            "control_shape_drift_is_diagnostic": "large drift means control changes more than a scalar gain",
            "history_delta_is_diagnostic": "large nonzero delta requires an explicit state or dynamic model",
            "no_thresholds_used_for_model_selection": True,
        },
        "fit": {
            "split": "fit",
            "rows": len(fit_rows),
            "git_revision": _git_revision(workspace),
        },
        "provenance": {
            "product_audio": False,
            "external_audio": False,
            "license": "repository-owned synthetic benchmark; no external audio",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    registry_path = workspace / "experiments.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    registry.append({
        "id": output.name,
        "git_commit": payload["fit"]["git_revision"],
        "data_manifest_sha256": hashlib.sha256((output / "dataset-manifest.json").read_bytes()).hexdigest(),
        "algorithm": "fit-only multi-level/control BLA and hidden-DUT history audit",
        "fit_split": "fit",
        "calibration_split": None,
        "validation_split": None,
        "locked": False,
        "status": "diagnostic",
        "metrics": str(metrics_path),
    })
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "output": str(output),
        "topologies": sorted(bla),
        "history_relative_delta": {
            topology: row["relative_output_delta"]
            for topology, row in history["by_topology"].items()
        },
    }, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/clone-structural-ood-bla-v1"))
    parser.add_argument("--fft-size", type=int, default=1024)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
