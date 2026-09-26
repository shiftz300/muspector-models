"""Train one higher-coverage causal LSTM forward diagnostic on Guitar-TECHS."""

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
from ..models.causal_lstm import CausalLSTM48
from ..sysid.estimators import LinearEstimator
from .run_real import CONTROL, PROFILE_IDS, RATE, TARGET_FRAMES, _load_rows


# A small epoch grid is selected on calibration.  The architecture and
# optimizer remain fixed; this is coverage/early-stopping validation, not an
# architecture sweep.
EPOCHS = (10, 18, 26)


def _git_revision(workspace: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _scale(rows: list[dict]) -> float:
    return max(
        float(np.quantile(np.abs(np.concatenate([
            np.concatenate([row["x"] for row in rows]),
            np.concatenate([row["y"] for row in rows]),
        ])), 0.995)),
        1.0e-3,
    )


def _fit_epoch_models(fit_rows, epochs: int, device: str):
    models = {}
    details = {}
    for profile_id in PROFILE_IDS:
        profile_rows = [row for row in fit_rows if row.profile_id == profile_id]
        public_rows = [row.public() for row in profile_rows]
        scale = _scale(public_rows)
        # Reset before construction so every epoch candidate starts from the
        # same initialization; fit() itself remains deterministic as well.
        torch.manual_seed(20260960)
        model = CausalLSTM48(
            device=device,
            epochs=epochs,
            amplitude_scale=scale,
        ).fit(public_rows)
        models[profile_id] = model
        details[profile_id] = {
            "parameters": model.parameters,
            "fit_seconds": model.fit_seconds,
            "amplitude_scale": model.amplitude_scale,
            "fit_examples": len(public_rows),
        }
        print(json.dumps({
            "stage": "fit",
            "epochs": epochs,
            "profile": profile_id,
            **details[profile_id],
        }), flush=True)
    return models, details


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


def run(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace LSTM coverage output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    from remix.license_gate import require_product_weights

    authorization = require_product_weights(
        workspace / "remix/data_sources.json", ("guitar-techs",),
    )
    fit_rows = _load_rows(workspace, "fit", args.fit_per_profile, 20260961)
    calibration_rows = _load_rows(workspace, "calibration", args.calibration_per_profile, 20260962)

    candidates = []
    models_by_epoch = {}
    details_by_epoch = {}
    for epochs in EPOCHS:
        models, details = _fit_epoch_models(fit_rows, epochs, args.device)
        calibration = _evaluate(models, calibration_rows)
        name = f"causal-lstm-4.8k-{epochs}epochs"
        record = {
            "name": name,
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
            "candidate": name,
            "absolute_esr": calibration["aggregate"]["absolute_esr"],
        }), flush=True)

    selected = min(
        candidates,
        key=lambda row: row["calibration"]["aggregate"]["absolute_esr"]["mean"],
    )
    selected_epoch = selected["epochs"]
    print(json.dumps({
        "stage": "select",
        "candidate": selected["name"],
        "calibration_absolute_esr": selected["calibration"]["aggregate"]["absolute_esr"],
    }), flush=True)

    # Do not load or evaluate development until the epoch is fixed.
    development_rows = _load_rows(workspace, "development", args.development_per_profile, 20260963)
    selected_models = models_by_epoch[selected_epoch]
    development = _evaluate(selected_models, development_rows)
    runtime = benchmark_runtime(selected_models[PROFILE_IDS[0]], TARGET_FRAMES, RATE)
    linear_models = {}
    linear_details = {}
    for profile_id in PROFILE_IDS:
        profile_rows = [row for row in fit_rows if row.profile_id == profile_id]
        linear = LinearEstimator().fit([row.public() for row in profile_rows])
        linear_models[profile_id] = linear
        linear_details[profile_id] = {
            "parameters": linear.parameters,
            "fit_seconds": linear.fit_seconds,
            "fit_examples": len(profile_rows),
        }
    linear_development = _evaluate(linear_models, development_rows)

    checkpoint = output / "forward-diagnostic.pt"
    torch.save({
        "schema": 1,
        "direction": "Guitar-TECHS direct-input -> Amp+cab+mic",
        "sample_rate": RATE,
        "architecture": "CausalLSTM48",
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
        "kind": "black-box-forward-system-identification-real-lstm-coverage",
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
        "epoch_selection": "common epoch selected on calibration only",
    }
    payload = {
        "schema": 1,
        "status": "diagnostic-real-lstm-coverage",
        "manifest": manifest,
        "selection": {
            "criterion": "mean aggregate absolute ESR on calibration only",
            "development_used_for_selection": False,
            "selected": {
                "name": selected["name"],
                "epochs": selected_epoch,
                "parameters": selected["parameters"],
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
                    "calibration": row["calibration"],
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
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "two fixed named Amp+cab+mic profiles only",
            "single published performance family with time-disjoint validation",
            "forward diagnostic, not a Wet-to-Clean inverse product",
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
        "algorithm": "4.8k causal LSTM with calibration-selected epoch and expanded fit coverage",
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
        "development_absolute_esr": development["aggregate"]["absolute_esr"],
    }, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/clone-real-guitar-techs-lstm-coverage-v1"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--fit-per-profile", type=int, default=64)
    parser.add_argument("--calibration-per-profile", type=int, default=32)
    parser.add_argument("--development-per-profile", type=int, default=64)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    run(args)


if __name__ == "__main__":
    main()
