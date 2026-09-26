"""Audit observable profile replay with an explicit out-of-domain guard.

Each synthetic DUT is treated as an unknown black-box profile.  The profile
fitter receives only public ``x``, ``y`` and ``control`` rows; the benchmark
harness keeps topology outside that API solely to construct separate profiles
and score hidden-DUT transfer.  A fixed tail-aware base selection is followed
by the one previously screened tiny residual.  At replay time, inputs beyond
the observed fit/calibration peak are abstained to an identity/input
passthrough instead of being extrapolated without a support contract.  This
is a forward-clone boundary, not a restoration-quality claim.
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
    make_observations,
    write_manifest,
)


BASE_CANDIDATES = ("parallel_hammerstein", "bounded_parallel_hammerstein")
FIT_SOURCES = TRAIN_SOURCES[:-2]
CALIBRATION_SOURCES = TRAIN_SOURCES[-2:]
OOD_SPLITS = ("source_ood", "level_ood", "control_ood")
SUPPORT_MARGIN = 0.05


def _git_revision(workspace: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _public(rows) -> list[dict]:
    """Strip topology, source, level and split metadata before profile fitting."""

    return [row.public() for row in rows]


def _fit_base(name: str, rows: list[dict]):
    if name == "parallel_hammerstein":
        return ParallelHammersteinEstimator().fit(rows)
    if name == "bounded_parallel_hammerstein":
        return BoundedParallelHammersteinEstimator().fit(rows)
    raise ValueError(name)


def _summary(model, rows: list[dict]) -> dict:
    return summarize([
        one_example(model.predict(row["x"], row["control"]), row["y"])
        for row in rows
    ])


def _fit_profile(
    fit_rows: list[dict],
    calibration_rows: list[dict],
    *,
    device: str,
    epochs: int,
) -> tuple[dict, object]:
    """Fit and select one profile without any hidden topology input."""

    bases = {}
    base_fit = {}
    for name in BASE_CANDIDATES:
        base = _fit_base(name, fit_rows)
        bases[name] = base
        base_fit[name] = {
            "parameters": int(base.parameters),
            "fit_seconds": float(base.fit_seconds),
            "fit_rows": len(fit_rows),
        }
    base_calibration = {
        name: _summary(model, calibration_rows)
        for name, model in bases.items()
    }
    base_choice = select_base(base_calibration, "tail_guard")
    base_name = base_choice["selected_model"]
    base = bases[base_name]

    residual = ParallelPlusResidual2k(base, device=device, epochs=epochs).fit(fit_rows)
    residual_calibration = _summary(residual, calibration_rows)
    base_calibration_selected = base_calibration[base_name]
    base_mean = base_calibration_selected["absolute_esr"]["mean"]
    residual_mean = residual_calibration["absolute_esr"]["mean"]
    base_p99 = base_calibration_selected["absolute_esr"]["p99"]
    residual_p99 = residual_calibration["absolute_esr"]["p99"]
    residual_selected = bool(
        residual_mean < base_mean and residual_p99 <= base_p99
    )
    selected_model = residual if residual_selected else base
    observed_peak = max(
        float(np.max(np.abs(row["x"])))
        for row in fit_rows + calibration_rows
    )
    profile = {
        "base_fit": base_fit,
        "base_calibration": base_calibration,
        "base_selection": base_choice,
        "selected_base": base_name,
        "residual": {
            "parameters": int(residual.residual_parameters),
            "total_parameters": int(residual.parameters),
            "device": device,
            "epochs": epochs,
            "fit_seconds": float(residual.fit_seconds),
            "fit_loss": float(residual.residual_fit_loss),
            "calibration": residual_calibration,
        },
        "final_selection": {
            "selected_model": "parallel_plus_residual_2k" if residual_selected else base_name,
            "residual_selected": residual_selected,
            "rule": "accept residual only when calibration absolute ESR mean strictly improves and p99 does not worsen",
            "base_absolute_esr_mean": base_mean,
            "residual_absolute_esr_mean": residual_mean,
            "base_absolute_esr_p99": base_p99,
            "residual_absolute_esr_p99": residual_p99,
        },
        "support_guard": {
            "observed_peak": observed_peak,
            "margin": SUPPORT_MARGIN,
            "supported_peak": observed_peak * (1.0 + SUPPORT_MARGIN),
            "fallback": "identity/input passthrough when peak exceeds supported_peak",
        },
    }
    return profile, selected_model


def _guarded_predict(profile: dict, model, row) -> tuple[np.ndarray, bool]:
    peak = float(np.max(np.abs(row.x)))
    if peak > profile["support_guard"]["supported_peak"]:
        return np.asarray(row.x, dtype=np.float32), False
    return model.predict(row.x, row.control), True


def _optional_summary(rows: list[dict]) -> dict | None:
    return summarize(rows) if rows else None


def _evaluate_rows(profiles: dict[str, tuple[dict, object]], rows: list) -> dict:
    all_metrics = []
    admitted_metrics = []
    passthrough_metrics = []
    by_topology = defaultdict(lambda: {
        "all": [], "admitted": [], "passthrough": [],
    })
    for row in rows:
        profile, model = profiles[row.topology]
        prediction, admitted = _guarded_predict(profile, model, row)
        metrics = one_example(prediction, row.y)
        all_metrics.append(metrics)
        group = by_topology[row.topology]
        group["all"].append(metrics)
        if admitted:
            admitted_metrics.append(metrics)
            group["admitted"].append(metrics)
        else:
            passthrough_metrics.append(metrics)
            group["passthrough"].append(metrics)
    return {
        "examples": len(all_metrics),
        "coverage": float(len(admitted_metrics) / max(len(all_metrics), 1)),
        "all_with_passthrough": summarize(all_metrics),
        "admitted": _optional_summary(admitted_metrics),
        "passthrough": _optional_summary(passthrough_metrics),
        "by_topology": {
            topology: {
                "examples": len(values["all"]),
                "coverage": float(len(values["admitted"]) / max(len(values["all"]), 1)),
                "all_with_passthrough": summarize(values["all"]),
                "admitted": _optional_summary(values["admitted"]),
                "passthrough": _optional_summary(values["passthrough"]),
            }
            for topology, values in sorted(by_topology.items())
        },
    }


def _evaluate_transfer(profile: tuple[dict, object], rows: list) -> dict:
    profile_data, model = profile
    all_metrics = []
    admitted_metrics = []
    passthrough_metrics = []
    by_topology = defaultdict(lambda: {"all": [], "admitted": [], "passthrough": []})
    for row in rows:
        if row.topology not in ("D", "E"):
            continue
        prediction, admitted = _guarded_predict(profile_data, model, row)
        metrics = one_example(prediction, row.y)
        all_metrics.append(metrics)
        group = by_topology[row.topology]
        group["all"].append(metrics)
        if admitted:
            admitted_metrics.append(metrics)
            group["admitted"].append(metrics)
        else:
            passthrough_metrics.append(metrics)
            group["passthrough"].append(metrics)
    return {
        "fit_profile": "A",
        "evaluated_hidden_topologies": ["D", "E"],
        "examples": len(all_metrics),
        "coverage": float(len(admitted_metrics) / max(len(all_metrics), 1)),
        "all_with_passthrough": summarize(all_metrics),
        "admitted": _optional_summary(admitted_metrics),
        "passthrough": _optional_summary(passthrough_metrics),
        "by_topology": {
            topology: {
                "examples": len(values["all"]),
                "coverage": float(len(values["admitted"]) / max(len(values["all"]), 1)),
                "all_with_passthrough": summarize(values["all"]),
                "admitted": _optional_summary(values["admitted"]),
                "passthrough": _optional_summary(values["passthrough"]),
            }
            for topology, values in sorted(by_topology.items())
        },
    }


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
        "profile_fit_sources": list(FIT_SOURCES),
        "profile_calibration_sources": list(CALIBRATION_SOURCES),
        "profile_input_contract": ["x", "y for fit/calibration only", "control"],
        "support_guard_margin": SUPPORT_MARGIN,
    })
    (output / "dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    fit_grid = make_observations("fit")
    profiles: dict[str, tuple[dict, object]] = {}
    profile_records = {}
    for topology in TOPOLOGIES:
        # Topology is used here only by the benchmark harness to assemble a
        # separate unknown profile.  It is stripped before _fit_profile runs.
        fit_rows = [
            row for row in fit_grid
            if row.topology == topology and row.source_id in FIT_SOURCES
        ]
        calibration_rows = [
            row for row in fit_grid
            if row.topology == topology and row.source_id in CALIBRATION_SOURCES
        ]
        profile, model = _fit_profile(
            _public(fit_rows),
            _public(calibration_rows),
            device=args.device,
            epochs=args.epochs,
        )
        profiles[topology] = (profile, model)
        profile_records[topology] = profile
        print(json.dumps({
            "stage": "profile_select",
            "profile": topology,
            "selected_model": profile["final_selection"]["selected_model"],
            "residual_selected": profile["final_selection"]["residual_selected"],
            "supported_peak": profile["support_guard"]["supported_peak"],
        }), flush=True)

    # OOD rows are created only after every profile has frozen its selection and
    # support boundary.
    splits = {
        split: make_observations(split)
        for split in OOD_SPLITS + ("topology_ood",)
    }
    same_profile = {
        split: _evaluate_rows(profiles, splits[split])
        for split in OOD_SPLITS
    }
    transfer = _evaluate_transfer(profiles["A"], splits["topology_ood"])
    selected_evaluation = {
        "same_profile_ood": same_profile,
        "topology_transfer": transfer,
        "cpu_rtf_fit_on_A": float(benchmark_runtime(profiles["A"][1], FRAMES, manifest["sample_rate"])),
    }
    print(json.dumps({
        "stage": "evaluate_profile_replay",
        "models": {
            topology: value["final_selection"]["selected_model"]
            for topology, value in profile_records.items()
        },
        "coverage": {
            split: value["coverage"]
            for split, value in same_profile.items()
        },
        "admitted_esr": {
            split: None if value["admitted"] is None else value["admitted"]["absolute_esr"]["mean"]
            for split, value in same_profile.items()
        },
        "topology_transfer_coverage": transfer["coverage"],
        "topology_transfer_admitted_esr": None if transfer["admitted"] is None else transfer["admitted"]["absolute_esr"]["mean"],
        "cpu_rtf_fit_on_A": selected_evaluation["cpu_rtf_fit_on_A"],
    }, indent=2), flush=True)

    payload = {
        "schema": 1,
        "status": "diagnostic-profile-replay-guard",
        "benchmark": manifest,
        "profile_contract": {
            "profile_fit_input": ["x", "y for fit only", "control"],
            "profile_calibration_input": ["x", "y for calibration only", "control"],
            "topology_input_to_fitter": False,
            "source_level_split_input_to_fitter": False,
            "renderer_identity_input": False,
            "analytic_inverse_used": False,
            "physical_audio_devices_used": False,
            "ood_created_only_after_selection": True,
            "selection_rule": "tail-aware bounded PH base, then accept one fixed tiny residual only if calibration mean improves and p99 does not worsen",
            "support_guard_rule": "abstain to identity/input passthrough above observed fit/calibration peak plus 5%",
            "mps_training_requested": args.device == "mps",
        },
        "fit": {
            "git_revision": _git_revision(workspace),
            "profiles": profile_records,
        },
        "selected_evaluation": selected_evaluation,
        "provenance": {
            "product_audio": False,
            "external_audio": False,
            "license": "repository-owned synthetic benchmark; no external audio",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "profile replay assumes product-safe paired calibration observations for each profile",
            "the support guard trades amplitude-OOD coverage for an explicit identity fallback; it is not a restoration-quality guarantee",
            "hidden topology transfer remains a structural stress test, not a universal topology classifier",
            "this is forward-clone evidence and does not produce an inverse/product checkpoint",
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
        "algorithm": "observable profile replay with tail-aware bounded PH, tiny residual and support guard",
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
        default=Path("runs/foundation/clone-profile-replay-v1"),
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    run(args)


if __name__ == "__main__":
    main()
