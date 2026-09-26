"""Train a restoration-loss Wet-to-DI inverse diagnostic on Guitar-TECHS."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..evaluation.evaluate import benchmark_runtime, one_example, summarize as summarize_error
from ..models.causal_lstm import CausalLSTM48
from ..sysid.estimators import LinearEstimator
from .run_real import CONTROL, PROFILE_IDS, RATE, TARGET_FRAMES, _load_rows


EPOCHS = (10, 18, 26)


def _git_revision(workspace: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _inverse_public(row) -> dict[str, np.ndarray | float]:
    return {"x": row.y, "y": row.x, "control": CONTROL}


def _scale(rows: list[dict]) -> float:
    return max(
        float(np.quantile(np.abs(np.concatenate([
            np.concatenate([row["x"] for row in rows]),
            np.concatenate([row["y"] for row in rows]),
        ])), 0.995)),
        1.0e-3,
    )


def _fit_epoch_models(fit_rows, epochs: int, device: str, residual: bool):
    models = {}
    details = {}
    for profile_id in PROFILE_IDS:
        profile_rows = [row for row in fit_rows if row.profile_id == profile_id]
        public_rows = [_inverse_public(row) for row in profile_rows]
        scale = _scale(public_rows)
        torch.manual_seed(20260980)
        model = CausalLSTM48(
            device=device,
            epochs=epochs,
            amplitude_scale=scale,
            residual=residual,
        ).fit(public_rows, objective="restoration")
        models[profile_id] = model
        details[profile_id] = {
            "parameters": model.parameters,
            "fit_seconds": model.fit_seconds,
            "amplitude_scale": model.amplitude_scale,
            "fit_examples": len(public_rows),
        }
        print(json.dumps({
            "stage": "fit",
            "objective": "restoration",
            "residual": residual,
            "direction": "wet-to-di",
            "epochs": epochs,
            "profile": profile_id,
            **details[profile_id],
        }), flush=True)
    return models, details


def _evaluate(models, rows):
    grouped_quality = defaultdict(list)
    grouped_error = defaultdict(list)
    for row in rows:
        public = _inverse_public(row)
        restored = models[row.profile_id].predict(public["x"], CONTROL)
        wet = public["x"][row.target_start:]
        clean = public["y"][row.target_start:]
        grouped_quality[row.profile_id].append((wet, restored[row.target_start:], clean))
        grouped_error[row.profile_id].append(one_example(restored[row.target_start:], clean))

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
    )


def _fit_linear_models(fit_rows):
    models = {}
    details = {}
    for profile_id in PROFILE_IDS:
        profile_rows = [row for row in fit_rows if row.profile_id == profile_id]
        model = LinearEstimator().fit([_inverse_public(row) for row in profile_rows])
        models[profile_id] = model
        details[profile_id] = {
            "parameters": model.parameters,
            "fit_seconds": model.fit_seconds,
            "fit_examples": len(profile_rows),
        }
    return models, details


def run(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace restoration-loss output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    from remix.license_gate import require_product_weights

    authorization = require_product_weights(
        workspace / "remix/data_sources.json", ("guitar-techs",),
    )
    fit_rows = _load_rows(workspace, "fit", args.fit_per_profile, 20260981)
    calibration_rows = _load_rows(workspace, "calibration", args.calibration_per_profile, 20260982)
    residual = args.residual

    candidates = []
    models_by_epoch = {}
    details_by_epoch = {}
    for epochs in EPOCHS:
        models, details = _fit_epoch_models(fit_rows, epochs, args.device, residual)
        calibration = _evaluate(models, calibration_rows)
        prefix = "inverse-causal-residual-lstm" if residual else "inverse-causal-lstm"
        name = f"{prefix}-4.8k-restoration-loss-{epochs}epochs"
        record = {
            "name": name,
            "objective": "restoration",
            "residual": residual,
            "epochs": epochs,
            "parameters": details[PROFILE_IDS[0]]["parameters"],
            "fit_seconds": sum(row["fit_seconds"] for row in details.values()),
            "fit_details": details,
            "calibration": calibration,
        }
        candidates.append(record)
        models_by_epoch[epochs] = models
        details_by_epoch[epochs] = details
        print(json.dumps({
            "stage": "calibration",
            "objective": "restoration",
            "residual": residual,
            "candidate": name,
            "quality_accepted": calibration["aggregate"]["quality_gate"]["accepted"],
            "quality_gates": calibration["aggregate"]["quality_gate"]["gates"],
            "absolute_esr": calibration["aggregate"]["absolute_error"]["absolute_esr"],
        }), flush=True)

    selected = min(candidates, key=_selection_key)
    selected_epoch = selected["epochs"]
    print(json.dumps({
        "stage": "select",
        "objective": "restoration",
        "direction": "wet-to-di",
        "candidate": selected["name"],
        "selection_key": list(_selection_key(selected)),
    }), flush=True)

    # Development is loaded only after the calibration-only model choice.
    development_rows = _load_rows(workspace, "development", args.development_per_profile, 20260983)
    selected_models = models_by_epoch[selected_epoch]
    development = _evaluate(selected_models, development_rows)
    runtime = benchmark_runtime(selected_models[PROFILE_IDS[0]], TARGET_FRAMES, RATE)

    linear_models, linear_details = _fit_linear_models(fit_rows)
    linear_development = _evaluate(linear_models, development_rows)

    checkpoint = output / "inverse-restoration-loss-diagnostic.pt"
    torch.save({
        "schema": 1,
        "direction": "Guitar-TECHS Amp+cab+mic -> direct-input",
        "sample_rate": RATE,
        "architecture": "CausalLSTM48",
        "objective": "restoration",
        "residual": residual,
        "selected_epochs": selected_epoch,
        "profiles": {
            profile_id: {
                "state_dict": selected_models[profile_id].network.state_dict(),
                "amplitude_scale": selected_models[profile_id].amplitude_scale,
            }
            for profile_id in PROFILE_IDS
        },
    }, checkpoint)

    manifest = {
        "schema": 1,
        "kind": "black-box-wet-to-clean-real-inverse-restoration-loss",
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
        "p3": "excluded by AmpCabPairs and not decoded",
        "objective": "waveform L1 + 0.5 transient L1 + 0.25 envelope L1 + 0.15 long-shape L1",
        "residual_path": residual,
        "epoch_selection": "quality-gate-aware common epoch selected on calibration only",
    }
    payload = {
        "schema": 1,
        "status": "diagnostic-real-inverse-residual-restoration-loss" if residual else "diagnostic-real-inverse-restoration-loss",
        "manifest": manifest,
        "selection": {
            "criterion": "first satisfy frozen quality gate, then maximize calibration gate count/pass fraction, then minimize absolute ESR",
            "development_used_for_selection": False,
            "selected": {
                "name": selected["name"],
                "epochs": selected_epoch,
                "parameters": selected["parameters"],
                "residual": residual,
            },
            "selected_calibration": selected["calibration"],
            "selected_development": development,
            "paired_linear_reference": {
                "model": "64-tap causal FIR",
                "fit_details": linear_details,
                "development": linear_development,
            },
            "candidate_grid": [
                {
                    "name": row["name"],
                    "epochs": row["epochs"],
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
            "device": args.device,
            "mps_used": args.device == "mps",
            "git_revision": _git_revision(workspace),
            "selected_fit_details": details_by_epoch[selected_epoch],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "residual": residual,
        },
        "estimator_contract": {
            "topology_input": False,
            "renderer_identity_input": False,
            "analytic_inverse_used": False,
            "physical_audio_devices_used": False,
            "profile_interpolation": False,
            "absolute_metrics": True,
        },
        "authorization": authorization,
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
            "causal inverse may not recover nonminimum-phase cabinet coloration",
            "no physical-chain veto or locked-final evaluation",
            "diagnostic inverse, not a promoted product model",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    registry_path = workspace / "experiments.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    registry.append({
        "id": output.name,
        "git_commit": payload["fit"]["git_revision"],
        "data_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "algorithm": "4.8k causal residual LSTM inverse with restoration-aware objective and frozen Amp gate" if residual else "4.8k causal LSTM inverse with restoration-aware objective and frozen Amp gate",
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
        "checkpoint": str(checkpoint),
        "selected": payload["selection"]["selected"],
        "calibration_accepted": selected["calibration"]["aggregate"]["quality_gate"]["accepted"],
        "development_accepted": development["aggregate"]["quality_gate"]["accepted"],
        "development_absolute_esr": development["aggregate"]["absolute_error"]["absolute_esr"],
    }, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/clone-real-guitar-techs-inverse-restoration-loss-v1"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--fit-per-profile", type=int, default=64)
    parser.add_argument("--calibration-per-profile", type=int, default=32)
    parser.add_argument("--development-per-profile", type=int, default=64)
    parser.add_argument("--residual", action="store_true")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    run(args)


if __name__ == "__main__":
    main()
