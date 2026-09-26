"""Train a spectral-prefiltered dynamic gray-box Amp inverse diagnostic.

The original state-bank inverse was fit directly on the Mic/Wet waveform.  An
Amp+cab+mic profile has a long, mostly linear cabinet/microphone response, so
this controlled follow-up first estimates a fit-only profile inverse FIR and
then fits the existing causal envelope state bank on the prefiltered signal.
This is a mechanism ablation, not an open-ended model sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import convolve

from ..evaluation.evaluate import benchmark_runtime, one_example, summarize as summarize_error
from ..sysid.state_bank import DynamicGrayBoxEstimator, TIME_CONSTANTS
from .run_real import CONTROL, PROFILE_IDS, RATE, TARGET_FRAMES, _load_rows


# The first entry is the structural ablation.  The remaining four vary only
# the short readout memory and ridge within a small pre-registered grid.
CANDIDATES = (
    ("spectral-inverse-fir-only", None, 0.0),
    ("spectral-graybox-t12-r1e-1", 12, 1.0e-1),
    ("spectral-graybox-t24-r1e-1", 24, 1.0e-1),
    ("spectral-graybox-t24-r1", 24, 1.0),
    ("spectral-graybox-t32-r1", 32, 1.0),
)


@dataclass
class _ProfileModel:
    prefilter: np.ndarray
    state_model: DynamicGrayBoxEstimator | None

    @property
    def parameters(self) -> int:
        return int(len(self.prefilter) + (0 if self.state_model is None else self.state_model.parameters))

    @property
    def fit_seconds(self) -> float:
        return 0.0 if self.state_model is None else self.state_model.fit_seconds

    def predict(self, value: np.ndarray, control: float) -> np.ndarray:
        prefiltered = np.asarray(
            convolve(np.asarray(value, dtype=np.float64), self.prefilter, mode="same", method="fft"),
            dtype=np.float32,
        )
        if self.state_model is None:
            return prefiltered
        return self.state_model.predict(prefiltered, control)


def _git_revision(workspace: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _inverse_public(row, prefilter: np.ndarray) -> dict[str, np.ndarray | float]:
    return {
        "x": np.asarray(
            convolve(np.asarray(row.y, dtype=np.float64), prefilter, mode="same", method="fft"),
            dtype=np.float32,
        ),
        "y": row.x,
        "control": CONTROL,
    }


def _fit_profile_firs(workspace: Path) -> tuple[dict[str, np.ndarray], dict]:
    # This helper reads only the licensed fit split.  The source implementation
    # uses P1/P2 only and records the fit-only spectral construction metadata.
    from remix.amp_cab_spectral import estimate_profile_firs

    filters, metadata = estimate_profile_firs(workspace, samples=400)
    if len(filters) != len(PROFILE_IDS):
        raise ValueError(f"unexpected profile FIR count: {len(filters)}")
    return {
        profile_id: np.asarray(filters[index].numpy(), dtype=np.float64)
        for index, profile_id in enumerate(PROFILE_IDS)
    }, metadata


def _fit_models(fit_rows, profile_firs: dict[str, np.ndarray], taps: int | None, ridge: float):
    models = {}
    details = {}
    for profile_id in PROFILE_IDS:
        profile_rows = [row for row in fit_rows if row.profile_id == profile_id]
        state_model = None
        if taps is not None:
            state_model = DynamicGrayBoxEstimator(taps=taps, ridge=ridge).fit([
                _inverse_public(row, profile_firs[profile_id]) for row in profile_rows
            ])
        model = _ProfileModel(profile_firs[profile_id], state_model)
        models[profile_id] = model
        details[profile_id] = {
            "parameters": model.parameters,
            "prefilter_fir_taps": len(model.prefilter),
            "state_bank_parameters": 0 if state_model is None else state_model.parameters,
            "branch_count": 0 if state_model is None else state_model.branch_count,
            "taps": taps,
            "ridge": ridge if taps is not None else None,
            "time_constants_seconds": list(TIME_CONSTANTS) if state_model is not None else [],
            "fit_seconds": model.fit_seconds,
            "fit_examples": len(profile_rows),
        }
        print(json.dumps({
            "stage": "fit",
            "mechanism": "spectral-prefiltered-dynamic-gray-box",
            "direction": "wet-to-di",
            "profile": profile_id,
            **details[profile_id],
        }), flush=True)
    return models, details


def _evaluate(models: dict[str, _ProfileModel], rows) -> dict:
    grouped_quality = defaultdict(list)
    grouped_error = defaultdict(list)
    for row in rows:
        model = models[row.profile_id]
        restored = model.predict(row.y, CONTROL)
        wet = row.y[row.target_start:]
        clean = row.x[row.target_start:]
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
    passed_gates = sum(bool(value) for value in quality["gates"].values())
    return (
        0 if quality["accepted"] else 1,
        -passed_gates,
        -quality["pass_fraction"],
        record["calibration"]["aggregate"]["absolute_error"]["absolute_esr"]["mean"],
        record["parameters"],
    )


def _save_checkpoint(output: Path, selected: dict, models: dict[str, _ProfileModel]) -> tuple[Path, Path]:
    checkpoint = output / "inverse-graybox-prefilter-diagnostic.npz"
    np.savez_compressed(
        checkpoint,
        schema=np.asarray([1], dtype=np.int64),
        direction=np.asarray(["Guitar-TECHS Amp+cab+mic -> direct-input"]),
        architecture=np.asarray(["SpectralPrefilterDynamicGrayBox"]),
        mechanism=np.asarray(["fit-only profile inverse FIR + envelope state bank"]),
        sample_rate=np.asarray([RATE], dtype=np.int64),
        selected_candidate=np.asarray([selected["name"]]),
        **{
            f"{profile_id}.prefilter": models[profile_id].prefilter
            for profile_id in PROFILE_IDS
        },
        **{
            f"{profile_id}.state_coefficients": (
                np.asarray([], dtype=np.float64)
                if models[profile_id].state_model is None
                else models[profile_id].state_model.coefficients
            )
            for profile_id in PROFILE_IDS
        },
    )
    config = {
        "schema": 1,
        "direction": "Guitar-TECHS Amp+cab+mic -> direct-input",
        "architecture": "SpectralPrefilterDynamicGrayBox",
        "mechanism": "fit-only profile inverse FIR + envelope state bank",
        "sample_rate": RATE,
        "selected_candidate": selected["name"],
        "profiles": {
            profile_id: {
                "prefilter_fir_taps": len(models[profile_id].prefilter),
                "state_bank_parameters": 0 if models[profile_id].state_model is None else models[profile_id].state_model.parameters,
                "branch_count": 0 if models[profile_id].state_model is None else models[profile_id].state_model.branch_count,
                "taps": selected["taps"],
                "ridge": selected["ridge"] if selected["taps"] is not None else None,
            }
            for profile_id in PROFILE_IDS
        },
    }
    config_path = output / "checkpoint.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return checkpoint, config_path


def _load_checkpoint(checkpoint: Path):
    config = json.loads((checkpoint.parent / "checkpoint.json").read_text())
    with np.load(checkpoint, allow_pickle=False) as archive:
        if int(np.asarray(archive["schema"]).reshape(-1)[0]) != int(config["schema"]):
            raise ValueError("gray-box prefilter checkpoint schema mismatch")
        if str(np.asarray(archive["direction"]).reshape(-1)[0]) != config["direction"]:
            raise ValueError("gray-box prefilter checkpoint direction mismatch")
        filters = {
            profile_id: np.asarray(archive[f"{profile_id}.prefilter"], dtype=np.float64)
            for profile_id in PROFILE_IDS
        }
        state_coefficients = {
            profile_id: np.asarray(archive[f"{profile_id}.state_coefficients"], dtype=np.float64)
            for profile_id in PROFILE_IDS
        }
    models = {}
    for profile_id in PROFILE_IDS:
        profile = config["profiles"][profile_id]
        taps = profile["taps"]
        state_model = None
        if taps is not None:
            state_model = DynamicGrayBoxEstimator(taps=int(taps), ridge=float(profile["ridge"]))
            state_model.coefficients = state_coefficients[profile_id]
            state_model.branch_count = int(profile["branch_count"])
            state_model.fit_seconds = 0.0
            expected = state_model.parameters
            if state_model.coefficients.shape != (expected,):
                raise ValueError(f"state coefficient shape mismatch for {profile_id}")
        models[profile_id] = _ProfileModel(filters[profile_id], state_model)
    return models, config


def run(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace inverse gray-box prefilter output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    from remix.license_gate import require_product_weights

    authorization = require_product_weights(
        workspace / "remix/data_sources.json", ("guitar-techs",),
    )
    profile_firs, spectral_initialization = _fit_profile_firs(workspace)
    fit_rows = _load_rows(workspace, "fit", args.fit_per_profile, 20260981)
    calibration_rows = _load_rows(workspace, "calibration", args.calibration_per_profile, 20260982)

    candidates = []
    models_by_name = {}
    for name, taps, ridge in CANDIDATES:
        print(json.dumps({
            "stage": "fit-candidate",
            "mechanism": "spectral-prefiltered-dynamic-gray-box",
            "candidate": name,
            "taps": taps,
            "ridge": ridge if taps is not None else None,
            "mps_used": False,
            "mps_reason": "profile FIR and state-bank least squares are fit-only NumPy/SciPy operations",
        }), flush=True)
        models, fit_details = _fit_models(fit_rows, profile_firs, taps, ridge)
        calibration = _evaluate(models, calibration_rows)
        parameters = fit_details[PROFILE_IDS[0]]["parameters"]
        record = {
            "name": name,
            "mechanism": "spectral-prefiltered-dynamic-gray-box",
            "direction": "wet-to-di",
            "taps": taps,
            "ridge": ridge if taps is not None else None,
            "branch_count": 0 if taps is None else 6 + 3 * len(TIME_CONSTANTS),
            "parameters": parameters,
            "prefilter_fir_taps": len(profile_firs[PROFILE_IDS[0]]),
            "time_constants_seconds": list(TIME_CONSTANTS) if taps is not None else [],
            "fit_seconds": sum(row["fit_seconds"] for row in fit_details.values()),
            "fit_details": fit_details,
            "calibration": calibration,
        }
        candidates.append(record)
        models_by_name[name] = models
        print(json.dumps({
            "stage": "calibration",
            "mechanism": "spectral-prefiltered-dynamic-gray-box",
            "candidate": name,
            "quality_accepted": calibration["aggregate"]["quality_gate"]["accepted"],
            "quality_gates": calibration["aggregate"]["quality_gate"]["gates"],
            "absolute_esr": calibration["aggregate"]["absolute_error"]["absolute_esr"],
        }), flush=True)

    selected = min(candidates, key=_selection_key)
    selected_models = models_by_name[selected["name"]]
    print(json.dumps({
        "stage": "select",
        "mechanism": "spectral-prefiltered-dynamic-gray-box",
        "direction": "wet-to-di",
        "candidate": selected["name"],
        "selection_key": list(_selection_key(selected)),
        "development_used_for_selection": False,
    }), flush=True)

    # The development interval is first read after the common configuration
    # choice; it cannot influence fit or selection.
    development_rows = _load_rows(workspace, "development", args.development_per_profile, 20260983)
    development = _evaluate(selected_models, development_rows)
    runtime = benchmark_runtime(selected_models[PROFILE_IDS[0]], TARGET_FRAMES, RATE)

    checkpoint, config_path = _save_checkpoint(output, selected, selected_models)
    reloaded_models, reloaded_config = _load_checkpoint(checkpoint)
    reloaded_development = _evaluate(reloaded_models, development_rows)
    selected_esr = development["aggregate"]["absolute_error"]["absolute_esr"]["mean"]
    reloaded_esr = reloaded_development["aggregate"]["absolute_error"]["absolute_esr"]["mean"]
    if not np.isclose(selected_esr, reloaded_esr, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"gray-box prefilter checkpoint reload changed ESR: {selected_esr} != {reloaded_esr}")

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
        "kind": "gray-box-spectral-prefilter-wet-to-clean-real-inverse",
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
        "spectral_initialization": spectral_initialization,
        "mps_used": False,
        "mps_reason": "the prefilter and gray-box fit are closed-form NumPy/SciPy operations",
    }
    payload = {
        "schema": 1,
        "status": "diagnostic-real-inverse-gray-box-spectral-prefilter",
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
            "candidate_grid": [
                {
                    "name": row["name"],
                    "taps": row["taps"],
                    "ridge": row["ridge"],
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
            "checkpoint_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
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
            "fit-only fixed spectral prefilter; no arbitrary cabinet or knob interpolation claim",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    registry_path = workspace / "experiments.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    registry.append({
        "id": output.name,
        "git_commit": payload["fit"]["git_revision"],
        "data_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "algorithm": "fit-only spectral inverse FIR plus calibration-selected dynamic gray-box state bank",
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
        "checkpoint_reload_matches": True,
    }, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/clone-real-guitar-techs-inverse-graybox-prefilter-v1"))
    parser.add_argument("--fit-per-profile", type=int, default=64)
    parser.add_argument("--calibration-per-profile", type=int, default=32)
    parser.add_argument("--development-per-profile", type=int, default=64)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
