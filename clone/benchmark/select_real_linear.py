"""Calibration-only selection of a small causal FIR for Guitar-TECHS."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..evaluation.evaluate import benchmark_runtime, one_example, summarize
from ..sysid.estimators import LinearEstimator
from .run_real import CONTROL, PROFILE_IDS, RATE, TARGET_FRAMES, _load_rows


# This is a deliberately small, pre-registered grid.  The calibration split
# chooses one common configuration; development is never consulted until the
# final report is produced.
CANDIDATES = (
    ("linear-t32-r1e-5", 32, 1.0e-5),
    ("linear-t64-r1e-5", 64, 1.0e-5),
    ("linear-t64-r1e-3", 64, 1.0e-3),
    ("linear-t64-r1e-1", 64, 1.0e-1),
    ("linear-t96-r1e-3", 96, 1.0e-3),
    ("linear-t128-r1e-2", 128, 1.0e-2),
    ("linear-t192-r1e-2", 192, 1.0e-2),
    ("linear-t257-r1e-2", 257, 1.0e-2),
    ("linear-t257-r1e-1", 257, 1.0e-1),
)


def _git_revision(workspace: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _evaluate(models: dict[str, LinearEstimator], rows) -> dict:
    category_rows = defaultdict(list)
    for row in rows:
        prediction = models[row.profile_id].predict(row.x, CONTROL)
        category_rows[row.profile_id].append(one_example(
            prediction[row.target_start:], row.y[row.target_start:],
        ))
    return {
        "profiles": {
            profile_id: summarize(values)
            for profile_id, values in sorted(category_rows.items())
        },
        "aggregate": summarize([
            value for profile_values in category_rows.values() for value in profile_values
        ]),
    }


def run(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace real selection output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    from remix.license_gate import require_product_weights

    authorization = require_product_weights(
        workspace / "remix/data_sources.json", ("guitar-techs",),
    )
    fit_rows = _load_rows(workspace, "fit", args.fit_per_profile, 20260941)
    calibration_rows = _load_rows(workspace, "calibration", args.calibration_per_profile, 20260942)

    candidates = []
    for name, taps, ridge in CANDIDATES:
        print(json.dumps({"stage": "fit", "candidate": name, "taps": taps, "ridge": ridge}), flush=True)
        models = {}
        fit_details = {}
        for profile_id in PROFILE_IDS:
            profile_fit_rows = [row for row in fit_rows if row.profile_id == profile_id]
            estimator = LinearEstimator(taps=taps, ridge=ridge).fit(
                [row.public() for row in profile_fit_rows],
            )
            models[profile_id] = estimator
            fit_details[profile_id] = {
                "parameters": estimator.parameters,
                "fit_seconds": estimator.fit_seconds,
            }
        calibration = _evaluate(models, calibration_rows)
        record = {
            "name": name,
            "taps": taps,
            "ridge": ridge,
            "parameters": taps,
            "fit_seconds": sum(row["fit_seconds"] for row in fit_details.values()),
            "fit_details": fit_details,
            "calibration": calibration,
        }
        candidates.append(record)
        print(json.dumps({
            "stage": "calibration",
            "candidate": name,
            "absolute_esr": calibration["aggregate"]["absolute_esr"],
        }), flush=True)

    selected = min(
        candidates,
        key=lambda row: (
            row["calibration"]["aggregate"]["absolute_esr"]["mean"],
            row["parameters"],
        ),
    )
    print(json.dumps({
        "stage": "select",
        "candidate": selected["name"],
        "calibration_absolute_esr": selected["calibration"]["aggregate"]["absolute_esr"],
    }), flush=True)

    # Only now, after calibration has selected one candidate, evaluate the
    # untouched development split for the final acceptance report.
    development_rows = _load_rows(workspace, "development", args.development_per_profile, 20260943)
    selected_models = {}
    selected_fit_details = {}
    for profile_id in PROFILE_IDS:
        profile_fit_rows = [row for row in fit_rows if row.profile_id == profile_id]
        estimator = LinearEstimator(
            taps=selected["taps"], ridge=selected["ridge"],
        ).fit([row.public() for row in profile_fit_rows])
        selected_models[profile_id] = estimator
        selected_fit_details[profile_id] = {
            "parameters": estimator.parameters,
            "fit_seconds": estimator.fit_seconds,
        }
    selected_development = _evaluate(selected_models, development_rows)

    manifest = {
        "schema": 1,
        "kind": "black-box-forward-system-identification-real-linear-selection",
        "source_id": "guitar-techs",
        "profiles": list(PROFILE_IDS),
        "sample_rate": RATE,
        "target_frames": TARGET_FRAMES,
        "history_frames": 8_192,
        "fit_split": "Guitar-TECHS fit time interval",
        "calibration_split": "Guitar-TECHS calibration time interval",
        "development_split": "Guitar-TECHS development time interval",
        "fit_examples_per_profile": args.fit_per_profile,
        "calibration_examples_per_profile": args.calibration_per_profile,
        "development_examples_per_profile": args.development_per_profile,
        "control": "fixed profile; no knob interpolation claim",
        "direction": "Guitar-TECHS direct-input -> Amp+cab+mic",
        "p3": "excluded by AmpCabPairs and not decoded",
        "selection_scope": "one common FIR configuration across P1/P2",
    }
    payload = {
        "schema": 1,
        "status": "diagnostic-real-linear-selection",
        "manifest": manifest,
        "selection": {
            "criterion": "mean aggregate absolute ESR on calibration only",
            "development_used_for_selection": False,
            "selected": {
                key: selected[key] for key in ("name", "taps", "ridge", "parameters")
            },
            "selected_calibration": selected["calibration"],
            "selected_development": selected_development,
            "candidate_grid": [
                {key: row[key] for key in ("name", "taps", "ridge")}
                for row in candidates
            ],
        },
        "candidates": candidates,
        "estimator_contract": {
            "topology_input": False,
            "renderer_identity_input": False,
            "analytic_inverse_used": False,
            "physical_audio_devices_used": False,
            "profile_interpolation": False,
            "absolute_metrics": True,
        },
        "authorization": authorization,
        "fit": {
            "device": "cpu",
            "mps_used": False,
            "reason": "closed-form FIR least squares; no neural optimizer in this experiment",
            "git_revision": _git_revision(workspace),
        },
        "selected_fit_details": selected_fit_details,
        "provenance": {
            "source_id": "guitar-techs",
            "license": "CC BY 4.0; attribution is in remix/data_sources.json",
            "p3_audio_opened": False,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "two fixed named Amp+cab+mic profiles only",
            "single published performance family with time-disjoint validation",
            "not a generic Amp or cabinet model and not a product promotion gate",
            "no listening acceptance or physical-chain veto",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    registry_path = workspace / "experiments.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    registry.append({
        "id": output.name,
        "git_commit": payload["fit"]["git_revision"],
        "data_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "algorithm": "calibration-only FIR selection",
        "fit_split": manifest["fit_split"],
        "calibration_split": manifest["calibration_split"],
        "validation_split": manifest["development_split"],
        "locked": False,
        "status": "diagnostic",
        "metrics": str(output / "metrics.json"),
    })
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "output": str(output),
        "selected": payload["selection"]["selected"],
        "development_absolute_esr": payload["selection"]["selected_development"]["aggregate"]["absolute_esr"],
    }, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/clone-real-guitar-techs-v5"))
    parser.add_argument("--fit-per-profile", type=int, default=24)
    parser.add_argument("--calibration-per-profile", type=int, default=24)
    parser.add_argument("--development-per-profile", type=int, default=24)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
