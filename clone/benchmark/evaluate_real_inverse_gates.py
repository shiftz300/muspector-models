"""Run the frozen mechanism-aware Amp gates on the inverse diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..evaluation.evaluate import one_example, summarize as summarize_error
from ..models.causal_lstm import CausalLSTM48
from .run_real import CONTROL, PROFILE_IDS, _load_rows


INVERSE_RUN = Path("runs/foundation/clone-real-guitar-techs-inverse-lstm-v1")


def _git_revision(workspace: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_models(checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("direction") != "Guitar-TECHS Amp+cab+mic -> direct-input":
        raise ValueError("checkpoint is not the expected forward direction")
    models = {}
    for profile_id in PROFILE_IDS:
        profile = payload["profiles"].get(profile_id)
        if profile is None:
            raise ValueError(f"checkpoint missing profile: {profile_id}")
        model = CausalLSTM48(
            device="cpu",
            epochs=1,
            amplitude_scale=float(profile["amplitude_scale"]),
            residual=bool(payload.get("residual", False)),
        )
        model.network.load_state_dict(profile["state_dict"])
        model.network.eval()
        models[profile_id] = model
    return models, payload


def _evaluate(models, rows):
    grouped_gates = defaultdict(list)
    grouped_error = defaultdict(list)
    for row in rows:
        wet = row.y[row.target_start:]
        clean = row.x[row.target_start:]
        restored = models[row.profile_id].predict(row.y, CONTROL)[row.target_start:]
        grouped_gates[row.profile_id].append((wet, restored, clean))
        grouped_error[row.profile_id].append(one_example(restored, clean))

    from remix.quality2 import summarize as summarize_quality

    profiles = {}
    for profile_id in PROFILE_IDS:
        wet, restored, clean = zip(*grouped_gates[profile_id], strict=True)
        quality = summarize_quality("amp", list(wet), list(restored), list(clean))
        error = summarize_error(grouped_error[profile_id])
        profiles[profile_id] = {
            "quality_gate": quality,
            "absolute_error": error,
        }
    all_triplets = [item for profile_values in grouped_gates.values() for item in profile_values]
    all_errors = [item for profile_values in grouped_error.values() for item in profile_values]
    wet, restored, clean = zip(*all_triplets, strict=True)
    return {
        "profiles": profiles,
        "aggregate": {
            "quality_gate": summarize_quality("amp", list(wet), list(restored), list(clean)),
            "absolute_error": summarize_error(all_errors),
        },
    }


def run(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace inverse gate output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = (workspace / args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    models, checkpoint_payload = _load_models(checkpoint)
    calibration_rows = _load_rows(
        workspace, "calibration", args.calibration_per_profile, args.calibration_seed,
    )
    calibration = _evaluate(models, calibration_rows)

    # The frozen gate is recorded before the development split is read.  No
    # threshold or model choice is fitted to development.
    from remix.quality2 import REQUIRED_IMPROVEMENTS, THRESHOLDS

    gate_policy = {
        "mechanism": "amp",
        "required_improvements": sorted(REQUIRED_IMPROVEMENTS["amp"]),
        "thresholds": THRESHOLDS,
        "pass_fraction_minimum": 0.50,
        "added_clipping_fraction_maximum": 0.01,
        "source": "remix.quality2 frozen policy",
    }
    development_rows = _load_rows(
        workspace, "development", args.development_per_profile, args.development_seed,
    )
    development = _evaluate(models, development_rows)

    manifest = {
        "schema": 1,
        "kind": "black-box-wet-to-clean-real-inverse-gate",
        "source_id": "guitar-techs",
        "profiles": list(PROFILE_IDS),
        "fit_examples_per_profile": checkpoint_payload.get("fit_examples_per_profile", 64),
        "calibration_examples_per_profile": args.calibration_per_profile,
        "development_examples_per_profile": args.development_per_profile,
        "calibration_seed": args.calibration_seed,
        "development_seed": args.development_seed,
        "direction": "Guitar-TECHS Amp+cab+mic -> direct-input",
        "p3": "excluded by AmpCabPairs and not decoded",
    }
    payload = {
        "schema": 1,
        "status": "diagnostic-real-inverse-gates",
        "manifest": manifest,
        "gate_policy": gate_policy,
        "selection_used_development": False,
        "calibration": calibration,
        "development": development,
        "fit": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "git_revision": _git_revision(workspace),
        },
        "authorization": checkpoint_payload.get("authorization"),
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
        ],
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    registry_path = workspace / "experiments.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    registry.append({
        "id": output.name,
        "git_commit": payload["fit"]["git_revision"],
        "data_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "algorithm": "frozen quality2 Amp gate on inverse diagnostic",
        "fit_split": "checkpoint fit split",
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
        "calibration_accepted": calibration["aggregate"]["quality_gate"]["accepted"],
        "development_accepted": development["aggregate"]["quality_gate"]["accepted"],
        "development_absolute_esr": development["aggregate"]["absolute_error"]["absolute_esr"],
    }, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, default=INVERSE_RUN / "inverse-diagnostic.pt")
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/clone-real-guitar-techs-inverse-gates-v1"))
    parser.add_argument("--calibration-per-profile", type=int, default=32)
    parser.add_argument("--development-per-profile", type=int, default=64)
    parser.add_argument("--calibration-seed", type=int, default=20260962)
    parser.add_argument("--development-seed", type=int, default=20260963)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
