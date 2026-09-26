"""Train and validate a state-bank gray-box Wet-to-DI inverse diagnostic.

This is deliberately separate from the neural inverse experiments.  The
estimator is a causal, low-order nonlinear state bank followed by a fitted
short FIR bank.  It receives only one effect instance's Wet signal and its
fixed profile control; graph order and neighbouring effects are not inputs.

The small candidate grid is registered before calibration.  Calibration is
used for the common state-bank configuration choice; the development split is
read only after that choice and is never used to tune the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..evaluation.evaluate import benchmark_runtime, one_example, summarize as summarize_error
from ..sysid.estimators import LinearEstimator
from ..sysid.state_bank import DynamicGrayBoxEstimator, TIME_CONSTANTS
from .run_real import CONTROL, PROFILE_IDS, RATE, TARGET_FRAMES, _load_rows


# A pre-declared mechanism grid, not an open-ended capacity sweep.  The
# nonlinear/state basis is fixed; only FIR memory and numerical regularization
# are compared.  The common candidate is selected on calibration only.
CANDIDATES = (
    ("graybox-t12-r1e-3", 12, 1.0e-3),
    ("graybox-t24-r1e-3", 24, 1.0e-3),
    ("graybox-t24-r1e-2", 24, 1.0e-2),
    ("graybox-t24-r1e-1", 24, 1.0e-1),
    ("graybox-t32-r1e-2", 32, 1.0e-2),
)


def _git_revision(workspace: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _inverse_public(row) -> dict[str, np.ndarray | float]:
    """Expose the real pair in the required immediate Wet -> predecessor direction."""

    return {"x": row.y, "y": row.x, "control": CONTROL}


def _fit_models(fit_rows, taps: int, ridge: float):
    models = {}
    details = {}
    for profile_id in PROFILE_IDS:
        profile_rows = [row for row in fit_rows if row.profile_id == profile_id]
        model = DynamicGrayBoxEstimator(taps=taps, ridge=ridge).fit(
            [_inverse_public(row) for row in profile_rows],
        )
        models[profile_id] = model
        details[profile_id] = {
            "parameters": model.parameters,
            "branch_count": model.branch_count,
            "taps": taps,
            "ridge": ridge,
            "time_constants_seconds": list(TIME_CONSTANTS),
            "fit_seconds": model.fit_seconds,
            "fit_examples": len(profile_rows),
        }
        print(json.dumps({
            "stage": "fit",
            "mechanism": "dynamic-gray-box-state-bank",
            "direction": "wet-to-di",
            "profile": profile_id,
            **details[profile_id],
        }), flush=True)
    return models, details


def _fit_linear_models(fit_rows):
    models = {}
    details = {}
    for profile_id in PROFILE_IDS:
        profile_rows = [row for row in fit_rows if row.profile_id == profile_id]
        model = LinearEstimator().fit([_inverse_public(row) for row in profile_rows])
        models[profile_id] = model
        details[profile_id] = {
            "parameters": model.parameters,
            "taps": model.taps,
            "ridge": model.ridge,
            "fit_seconds": model.fit_seconds,
            "fit_examples": len(profile_rows),
        }
    return models, details


def _evaluate(models, rows) -> dict:
    grouped_quality = defaultdict(list)
    grouped_error = defaultdict(list)
    for row in rows:
        public = _inverse_public(row)
        restored = models[row.profile_id].predict(public["x"], CONTROL)
        wet = public["x"][row.target_start:]
        clean = public["y"][row.target_start:]
        restored = restored[row.target_start:]
        grouped_quality[row.profile_id].append((wet, restored, clean))
        grouped_error[row.profile_id].append(one_example(restored, clean))

    from remix.quality2 import summarize as summarize_quality

    profiles = {}
    for profile_id in PROFILE_IDS:
        wet, restored, clean = zip(*grouped_quality[profile_id], strict=True)
        profiles[profile_id] = {
            "quality_gate": summarize_quality("amp", list(wet), list(restored), list(clean)),
            "absolute_error": summarize_error(grouped_error[profile_id]),
        }
    all_quality = [item for values in grouped_quality.values() for item in values]
    all_error = [item for values in grouped_error.values() for item in values]
    wet, restored, clean = zip(*all_quality, strict=True)
    return {
        "profiles": profiles,
        "aggregate": {
            "quality_gate": summarize_quality("amp", list(wet), list(restored), list(clean)),
            "absolute_error": summarize_error(all_error),
        },
    }


def _selection_key(record: dict) -> tuple:
    quality = record["calibration"]["aggregate"]["quality_gate"]
    gates = quality["gates"]
    passed_gates = sum(bool(value) for value in gates.values())
    return (
        0 if quality["accepted"] else 1,
        -passed_gates,
        -quality["pass_fraction"],
        record["calibration"]["aggregate"]["absolute_error"]["absolute_esr"]["mean"],
        record["parameters"],
    )


def _load_checkpoint(checkpoint: Path):
    payload = json.loads((checkpoint.parent / "checkpoint.json").read_text())
    with np.load(checkpoint, allow_pickle=False) as archive:
        archive_schema = int(np.asarray(archive["schema"]).reshape(-1)[0])
        if archive_schema != int(payload["schema"]):
            raise ValueError(f"checkpoint schema mismatch: {archive_schema} != {payload['schema']}")
        archive_direction = str(np.asarray(archive["direction"]).reshape(-1)[0])
        if archive_direction != payload["direction"]:
            raise ValueError("checkpoint direction mismatch")
        archive_architecture = str(np.asarray(archive["architecture"]).reshape(-1)[0])
        if archive_architecture != payload["architecture"]:
            raise ValueError("checkpoint architecture mismatch")
        coefficients_by_profile = {
            profile_id: np.asarray(
                archive[f"{profile_id}.coefficients"], dtype=np.float64,
            )
            for profile_id in PROFILE_IDS
        }
    models = {}
    for profile_id in PROFILE_IDS:
        profile = payload["profiles"].get(profile_id)
        if profile is None:
            raise ValueError(f"checkpoint missing profile: {profile_id}")
        model = DynamicGrayBoxEstimator(
            taps=int(profile["taps"]), ridge=float(profile["ridge"]),
        )
        model.coefficients = coefficients_by_profile[profile_id]
        model.branch_count = int(profile["branch_count"])
        expected_width = model.branch_count * model.taps
        if model.coefficients.shape != (expected_width,):
            raise ValueError(
                f"checkpoint coefficient shape mismatch for {profile_id}: "
                f"{model.coefficients.shape} != {(expected_width,)}"
            )
        model.fit_seconds = 0.0
        models[profile_id] = model
    return models, payload


def run(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace inverse gray-box output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    from remix.license_gate import require_product_weights

    authorization = require_product_weights(
        workspace / "remix/data_sources.json", ("guitar-techs",),
    )
    fit_rows = _load_rows(workspace, "fit", args.fit_per_profile, 20260981)
    calibration_rows = _load_rows(workspace, "calibration", args.calibration_per_profile, 20260982)

    candidates = []
    models_by_name = {}
    for name, taps, ridge in CANDIDATES:
        print(json.dumps({
            "stage": "fit-candidate",
            "mechanism": "dynamic-gray-box-state-bank",
            "candidate": name,
            "taps": taps,
            "ridge": ridge,
            "mps_used": False,
            "mps_reason": "state-bank basis and closed-form least squares are NumPy/SciPy CPU operations",
        }), flush=True)
        models, fit_details = _fit_models(fit_rows, taps, ridge)
        calibration = _evaluate(models, calibration_rows)
        record = {
            "name": name,
            "mechanism": "dynamic-gray-box-state-bank",
            "direction": "wet-to-di",
            "taps": taps,
            "ridge": ridge,
            "branch_count": 6 + 3 * len(TIME_CONSTANTS),
            "parameters": (6 + 3 * len(TIME_CONSTANTS)) * taps,
            "time_constants_seconds": list(TIME_CONSTANTS),
            "fit_seconds": sum(row["fit_seconds"] for row in fit_details.values()),
            "fit_details": fit_details,
            "calibration": calibration,
        }
        candidates.append(record)
        models_by_name[name] = models
        print(json.dumps({
            "stage": "calibration",
            "mechanism": "dynamic-gray-box-state-bank",
            "candidate": name,
            "quality_accepted": calibration["aggregate"]["quality_gate"]["accepted"],
            "quality_gates": calibration["aggregate"]["quality_gate"]["gates"],
            "absolute_esr": calibration["aggregate"]["absolute_error"]["absolute_esr"],
        }), flush=True)

    selected = min(candidates, key=_selection_key)
    selected_models = models_by_name[selected["name"]]
    print(json.dumps({
        "stage": "select",
        "mechanism": "dynamic-gray-box-state-bank",
        "direction": "wet-to-di",
        "candidate": selected["name"],
        "selection_key": list(_selection_key(selected)),
        "development_used_for_selection": False,
    }), flush=True)

    # Development is loaded only after the calibration-only candidate choice.
    development_rows = _load_rows(workspace, "development", args.development_per_profile, 20260983)
    development = _evaluate(selected_models, development_rows)
    runtime = benchmark_runtime(selected_models[PROFILE_IDS[0]], TARGET_FRAMES, RATE)

    linear_models, linear_details = _fit_linear_models(fit_rows)
    linear_development = _evaluate(linear_models, development_rows)

    checkpoint = output / "inverse-graybox-diagnostic.npz"
    checkpoint_payload = {
        "schema": 1,
        "direction": "Guitar-TECHS Amp+cab+mic -> direct-input",
        "architecture": "DynamicGrayBoxEstimator",
        "mechanism": "fixed envelope state bank + causal FIR readout",
        "sample_rate": RATE,
        "selected_candidate": selected["name"],
        "profiles": {
            profile_id: {
                "taps": selected["taps"],
                "ridge": selected["ridge"],
                "branch_count": selected["branch_count"],
                "coefficients": selected_models[profile_id].coefficients,
            }
            for profile_id in PROFILE_IDS
        },
    }
    np.savez_compressed(
        checkpoint,
        schema=np.asarray([checkpoint_payload["schema"]], dtype=np.int64),
        direction=np.asarray([checkpoint_payload["direction"]]),
        architecture=np.asarray([checkpoint_payload["architecture"]]),
        mechanism=np.asarray([checkpoint_payload["mechanism"]]),
        sample_rate=np.asarray([RATE], dtype=np.int64),
        selected_candidate=np.asarray([selected["name"]]),
        **{
            f"{profile_id}.coefficients": selected_models[profile_id].coefficients
            for profile_id in PROFILE_IDS
        },
    )
    # JSON sidecar keeps the exact configuration human-auditable while the
    # compressed numeric file preserves the fitted coefficients without lossy
    # text conversion.
    (output / "checkpoint.json").write_text(json.dumps({
        **{key: value for key, value in checkpoint_payload.items() if key != "profiles"},
        "profiles": {
            profile_id: {
                "taps": value["taps"],
                "ridge": value["ridge"],
                "branch_count": value["branch_count"],
                "coefficients": value["coefficients"].tolist(),
            }
            for profile_id, value in checkpoint_payload["profiles"].items()
        },
    }, indent=2, sort_keys=True) + "\n")

    reloaded_models, reloaded_payload = _load_checkpoint(checkpoint)
    reloaded_development = _evaluate(reloaded_models, development_rows)
    reload_esr = reloaded_development["aggregate"]["absolute_error"]["absolute_esr"]["mean"]
    selected_esr = development["aggregate"]["absolute_error"]["absolute_esr"]["mean"]
    if not np.isclose(reload_esr, selected_esr, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"gray-box checkpoint reload changed development ESR: {reload_esr} != {selected_esr}")

    from remix.quality2 import REQUIRED_IMPROVEMENTS, THRESHOLDS

    gate_policy = {
        "mechanism": "amp",
        "required_improvements": sorted(REQUIRED_IMPROVEMENTS["amp"]),
        "thresholds": THRESHOLDS,
        "pass_fraction_minimum": 0.50,
        "added_clipping_fraction_maximum": 0.01,
        "source": "remix.quality2 frozen policy",
    }
    manifest = {
        "schema": 1,
        "kind": "gray-box-wet-to-clean-real-inverse",
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
        "direction": "Guitar-TECHS Amp+cab+mic -> direct-input",
        "order_independent_contract": "one effect Wet plus fixed profile control only; no graph-order or neighbouring-effect input",
        "p3": "excluded by AmpCabPairs and not decoded",
        "state_bank_time_constants_seconds": list(TIME_CONSTANTS),
        "mps_used": False,
        "mps_reason": "the gray-box fit is a NumPy/SciPy closed-form CPU estimator; MPS is not involved in this algorithm",
    }
    payload = {
        "schema": 1,
        "status": "diagnostic-real-inverse-gray-box",
        "manifest": manifest,
        "gate_policy": gate_policy,
        "selection": {
            "criterion": "first satisfy frozen quality gate, then maximize calibration gate count/pass fraction, then minimize calibration absolute ESR",
            "development_used_for_selection": False,
            "selected": {
                "name": selected["name"],
                "taps": selected["taps"],
                "ridge": selected["ridge"],
                "branch_count": selected["branch_count"],
                "parameters": selected["parameters"],
            },
            "selected_calibration": selected["calibration"],
            "selected_development": development,
            "reloaded_development": reloaded_development,
            "checkpoint_reload_matches": True,
            "paired_linear_reference": {
                "model": "64-tap causal FIR",
                "fit_details": linear_details,
                "development": linear_development,
            },
            "candidate_grid": [
                {
                    "name": row["name"],
                    "taps": row["taps"],
                    "ridge": row["ridge"],
                    "branch_count": row["branch_count"],
                    "parameters": row["parameters"],
                    "fit_seconds": row["fit_seconds"],
                    "calibration_quality": row["calibration"]["aggregate"]["quality_gate"],
                    "calibration_absolute_error": row["calibration"]["aggregate"]["absolute_error"],
                }
                for row in candidates
            ],
        },
        "runtime": {
            "cpu_rtf": runtime,
            "deployment_device": "ordinary CPU measurement",
        },
        "fit": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "checkpoint_config_sha256": hashlib.sha256((output / "checkpoint.json").read_bytes()).hexdigest(),
            "git_revision": _git_revision(workspace),
            "authorization": authorization,
        },
        "provenance": {
            "source_id": "guitar-techs",
            "license": "CC BY 4.0; attribution is in remix/data_sources.json",
            "p3_audio_opened": False,
            "physical_audio_devices_used": False,
            "demo_generated": False,
            "locked_final_opened": False,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "frozen automatic gates do not replace Wet/Candidate/Original listening",
            "two fixed named Amp+cab+mic profiles only",
            "no physical-chain veto or locked-final evaluation",
            "fixed state-bank time constants and fixed profile control; no arbitrary knob or cabinet interpolation claim",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    registry_path = workspace / "experiments.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    registry.append({
        "id": output.name,
        "git_commit": payload["fit"]["git_revision"],
        "data_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "algorithm": "dynamic gray-box state bank inverse with calibration-selected FIR memory",
        "fit_split": "Guitar-TECHS fit time interval",
        "calibration_split": "Guitar-TECHS calibration time interval",
        "validation_split": "Guitar-TECHS development time interval",
        "locked": False,
        "status": "diagnostic",
        "metrics": str(output / "metrics.json"),
    })
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "output": str(output),
        "selected_candidate": selected["name"],
        "calibration_accepted": selected["calibration"]["aggregate"]["quality_gate"]["accepted"],
        "development_accepted": development["aggregate"]["quality_gate"]["accepted"],
        "development_absolute_esr": development["aggregate"]["absolute_error"]["absolute_esr"],
        "paired_linear_development_absolute_esr": linear_development["aggregate"]["absolute_error"]["absolute_esr"],
        "checkpoint_reload_matches": True,
    }, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/clone-real-guitar-techs-inverse-graybox-v1"))
    parser.add_argument("--fit-per-profile", type=int, default=64)
    parser.add_argument("--calibration-per-profile", type=int, default=32)
    parser.add_argument("--development-per-profile", type=int, default=64)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
