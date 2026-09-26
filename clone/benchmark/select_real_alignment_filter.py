"""Calibration-only selection of a fit-window alignment filter."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..evaluation.evaluate import one_example, summarize
from ..sysid.estimators import LinearEstimator
from .audit_real_alignment import _estimate_residual_lag
from .run_real import CONTROL, PROFILE_IDS, _load_rows


# None is the unfiltered reference.  The remaining thresholds are fixed
# before looking at calibration/development quality; calibration chooses only
# among these data-curation policies.
FILTERS = (
    ("all-fit-windows", None),
    ("abs-residual-lag-le-8", 8),
    ("abs-residual-lag-le-16", 16),
    ("abs-residual-lag-le-32", 32),
    ("abs-residual-lag-le-64", 64),
    ("abs-residual-lag-le-128", 128),
)


def _git_revision(workspace: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _annotate_fit(rows):
    annotated = []
    for row in rows:
        lag, correlation = _estimate_residual_lag(
            row.x[row.target_start:], row.y[row.target_start:],
        )
        annotated.append((row, lag, correlation))
    return annotated


def _evaluate(models, rows):
    grouped = defaultdict(list)
    for row in rows:
        prediction = models[row.profile_id].predict(row.x, CONTROL)
        grouped[row.profile_id].append(one_example(
            prediction[row.target_start:], row.y[row.target_start:],
        ))
    return {
        "profiles": {
            profile_id: summarize(values)
            for profile_id, values in sorted(grouped.items())
        },
        "aggregate": summarize([
            value for profile_values in grouped.values() for value in profile_values
        ]),
    }


def _fit_for_filter(annotated, threshold):
    models = {}
    counts = {}
    for profile_id in PROFILE_IDS:
        selected = [
            row for row, lag, _ in annotated
            if row.profile_id == profile_id
            and (threshold is None or abs(lag) <= threshold)
        ]
        if len(selected) < 2:
            raise ValueError(f"alignment filter leaves too few rows for {profile_id}")
        models[profile_id] = LinearEstimator().fit([row.public() for row in selected])
        counts[profile_id] = len(selected)
    return models, counts


def run(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace alignment-filter output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    from remix.license_gate import require_product_weights

    authorization = require_product_weights(
        workspace / "remix/data_sources.json", ("guitar-techs",),
    )
    fit_rows = _load_rows(workspace, "fit", args.fit_per_profile, 20260951)
    calibration_rows = _load_rows(workspace, "calibration", args.calibration_per_profile, 20260952)
    annotated = _annotate_fit(fit_rows)

    candidates = []
    for name, threshold in FILTERS:
        models, counts = _fit_for_filter(annotated, threshold)
        calibration = _evaluate(models, calibration_rows)
        record = {
            "name": name,
            "max_abs_residual_lag_frames": threshold,
            "fit_rows_per_profile": counts,
            "parameters": 64,
            "fit_seconds": sum(model.fit_seconds for model in models.values()),
            "calibration": calibration,
        }
        candidates.append(record)
        print(json.dumps({
            "stage": "calibration",
            "candidate": name,
            "fit_rows_per_profile": counts,
            "absolute_esr": calibration["aggregate"]["absolute_esr"],
        }), flush=True)

    selected = min(
        candidates,
        key=lambda row: (
            row["calibration"]["aggregate"]["absolute_esr"]["mean"],
            -sum(row["fit_rows_per_profile"].values()),
        ),
    )
    print(json.dumps({
        "stage": "select",
        "candidate": selected["name"],
        "calibration_absolute_esr": selected["calibration"]["aggregate"]["absolute_esr"],
    }), flush=True)

    # Development is loaded only after the fit-only curation policy has been
    # selected from calibration.
    development_rows = _load_rows(workspace, "development", args.development_per_profile, 20260953)
    selected_models, selected_counts = _fit_for_filter(annotated, selected["max_abs_residual_lag_frames"])
    development = _evaluate(selected_models, development_rows)
    reference_models, reference_counts = _fit_for_filter(annotated, None)
    reference_development = _evaluate(reference_models, development_rows)

    manifest = {
        "schema": 1,
        "kind": "black-box-forward-system-identification-real-alignment-filter-selection",
        "source_id": "guitar-techs",
        "profiles": list(PROFILE_IDS),
        "fit_split": "Guitar-TECHS fit time interval",
        "calibration_split": "Guitar-TECHS calibration time interval",
        "development_split": "Guitar-TECHS development time interval",
        "fit_examples_per_profile": args.fit_per_profile,
        "calibration_examples_per_profile": args.calibration_per_profile,
        "development_examples_per_profile": args.development_per_profile,
        "filter_signal": "differentiated local cross-correlation residual lag",
        "estimator": "64-tap causal FIR",
        "p3": "excluded by AmpCabPairs and not decoded",
    }
    payload = {
        "schema": 1,
        "status": "diagnostic-real-alignment-filter-selection",
        "manifest": manifest,
        "selection": {
            "criterion": "mean aggregate absolute ESR on calibration only",
            "development_used_for_selection": False,
            "selected": {
                key: selected[key]
                for key in ("name", "max_abs_residual_lag_frames", "fit_rows_per_profile", "parameters")
            },
            "selected_calibration": selected["calibration"],
            "selected_development": development,
            "all_fit_reference": {
                "fit_rows_per_profile": reference_counts,
                "development": reference_development,
            },
            "candidate_grid": [
                {key: row[key] for key in ("name", "max_abs_residual_lag_frames", "fit_rows_per_profile")}
                for row in candidates
            ],
        },
        "fit": {
            "git_revision": _git_revision(workspace),
            "selected_fit_rows_per_profile": selected_counts,
        },
        "authorization": authorization,
        "estimator_contract": {
            "topology_input": False,
            "renderer_identity_input": False,
            "analytic_inverse_used": False,
            "physical_audio_devices_used": False,
            "profile_interpolation": False,
            "absolute_metrics": True,
        },
        "provenance": {
            "source_id": "guitar-techs",
            "license": "CC BY 4.0; attribution is in remix/data_sources.json",
            "p3_audio_opened": False,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "fit curation is based on a local correlation proxy, not a physical delay measurement",
            "two fixed named Amp+cab+mic profiles only",
            "not a model quality or listening acceptance result",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    registry_path = workspace / "experiments.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    registry.append({
        "id": output.name,
        "git_commit": payload["fit"]["git_revision"],
        "data_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "algorithm": "calibration-only fit alignment filter selection plus 64-tap FIR",
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
        "development_absolute_esr": development["aggregate"]["absolute_esr"],
    }, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/clone-real-guitar-techs-alignment-filter-v1"))
    parser.add_argument("--fit-per-profile", type=int, default=24)
    parser.add_argument("--calibration-per-profile", type=int, default=24)
    parser.add_argument("--development-per-profile", type=int, default=24)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
