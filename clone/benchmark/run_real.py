"""Black-box estimator comparison on the licensed Guitar-TECHS P1/P2 pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..evaluation.evaluate import benchmark_runtime, one_example, summarize
from ..models.causal_lstm import CausalLSTM48
from ..sysid.estimators import LinearEstimator, ParallelHammersteinEstimator, WienerHammersteinEstimator
from ..sysid.state_bank import DynamicGrayBoxEstimator
from .synthetic_dut import RATE


PROFILE_IDS = ("P1-orange-cr60-sm57", "P2-yamaha-yb15-at2020")
TARGET_FRAMES = 4_096
CONTROL = 0.5


@dataclass(frozen=True)
class RealObservation:
    x: np.ndarray
    y: np.ndarray
    profile_id: str
    target_start: int
    index: int

    def public(self) -> dict[str, np.ndarray | float]:
        # A fixed profile is fitted independently.  The scalar is deliberately
        # constant; it is not a fabricated knob interpretation for hardware.
        return {"x": self.x, "y": self.y, "control": CONTROL}


def _git_revision(workspace: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_rows(workspace: Path, split: str, per_profile: int, seed: int) -> list[RealObservation]:
    # AmpCabPairs enforces the product license gate and excludes P3.  Request
    # only P1/P2 rows and never inspect any other profile directory.
    from remix.guitar_techs_amp_data import AmpCabPairs

    dataset = AmpCabPairs(
        workspace, split, per_profile * len(PROFILE_IDS), TARGET_FRAMES, seed,
    )
    rows = []
    for index in range(len(dataset)):
        value = dataset[index]
        if value["profile_id"] not in PROFILE_IDS:
            raise PermissionError(f"unexpected Guitar-TECHS profile: {value['profile_id']}")
        rows.append(RealObservation(
            value["clean"].numpy().astype(np.float32),
            value["wet"].numpy().astype(np.float32),
            value["profile_id"],
            int(value["target_start"]),
            index,
        ))
    return rows


def _fit_model(name: str, rows: list[dict], device: str):
    if name == "linear":
        return LinearEstimator().fit(rows)
    if name == "linear_257":
        return LinearEstimator(taps=257).fit(rows)
    if name == "parallel_hammerstein":
        return ParallelHammersteinEstimator().fit(rows)
    if name == "wiener_hammerstein":
        return WienerHammersteinEstimator(device=device).fit(rows)
    if name == "dynamic_gray_box":
        return DynamicGrayBoxEstimator().fit(rows)
    if name == "causal_lstm_4.8k":
        scale = max(
            float(np.quantile(np.abs(np.concatenate([
                np.concatenate([row["x"] for row in rows]),
                np.concatenate([row["y"] for row in rows]),
            ])), 0.995)),
            1.0e-3,
        )
        return CausalLSTM48(device=device, amplitude_scale=scale).fit(rows)
    raise ValueError(name)


def run(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace real benchmark output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    from remix.license_gate import require_product_weights

    authorization = require_product_weights(
        workspace / "remix/data_sources.json", ("guitar-techs",),
    )
    fit_rows = _load_rows(workspace, "fit", args.fit_per_profile, 20260931)
    development_rows = _load_rows(workspace, "development", args.development_per_profile, 20260932)
    models = (
        "linear", "linear_257", "parallel_hammerstein", "wiener_hammerstein",
        "dynamic_gray_box", "causal_lstm_4.8k",
    )
    results = {}
    for model_name in models:
        print(json.dumps({"stage": "fit", "model": model_name}), flush=True)
        profile_models = {}
        profile_fit = {}
        for profile_id in PROFILE_IDS:
            profile_fit_rows = [row for row in fit_rows if row.profile_id == profile_id]
            estimator = _fit_model(model_name, [row.public() for row in profile_fit_rows], args.device)
            profile_models[profile_id] = estimator
            profile_fit[profile_id] = {
                "parameters": estimator.parameters,
                "fit_seconds": estimator.fit_seconds,
                "residual_parameters": getattr(estimator, "residual_parameters", None),
                "amplitude_scale": getattr(estimator, "amplitude_scale", None),
            }
            print(json.dumps({"model": model_name, "profile": profile_id, **profile_fit[profile_id]}), flush=True)
        category_rows = defaultdict(list)
        for row in development_rows:
            estimator = profile_models[row.profile_id]
            prediction = estimator.predict(row.x, CONTROL)
            category_rows[row.profile_id].append(one_example(
                prediction[row.target_start:], row.y[row.target_start:],
            ))
        runtime = benchmark_runtime(profile_models[PROFILE_IDS[0]], TARGET_FRAMES, RATE)
        results[model_name] = {
            "parameters": profile_fit[PROFILE_IDS[0]]["parameters"],
            "fit_seconds": sum(value["fit_seconds"] for value in profile_fit.values()),
            "cpu_rtf": runtime,
            "profiles": {
                profile_id: summarize(rows) for profile_id, rows in sorted(category_rows.items())
            },
            "aggregate": summarize([row for rows in category_rows.values() for row in rows]),
        }
        print(json.dumps({
            "stage": "evaluate", "model": model_name,
            "cpu_rtf": runtime,
            "absolute_esr": results[model_name]["aggregate"]["absolute_esr"],
        }), flush=True)

    manifest = {
        "schema": 1,
        "kind": "black-box-forward-system-identification-real-aligned",
        "source_id": "guitar-techs",
        "profiles": list(PROFILE_IDS),
        "sample_rate": RATE,
        "target_frames": TARGET_FRAMES,
        "history_frames": 8_192,
        "fit_split": "Guitar-TECHS fit time interval",
        "development_split": "Guitar-TECHS development time interval",
        "fit_examples_per_profile": args.fit_per_profile,
        "development_examples_per_profile": args.development_per_profile,
        "control": "fixed profile; no knob interpolation claim",
        "direction": "Guitar-TECHS direct-input -> Amp+cab+mic",
        "p3": "excluded by AmpCabPairs and not decoded",
    }
    payload = {
        "schema": 1,
        "status": "diagnostic-real-aligned",
        "manifest": manifest,
        "estimator_contract": {
            "topology_input": False,
            "renderer_identity_input": False,
            "analytic_inverse_used": False,
            "physical_audio_devices_used": False,
            "profile_interpolation": False,
            "absolute_metrics": True,
        },
        "authorization": authorization,
        "fit": {"device": args.device, "git_revision": _git_revision(workspace)},
        "results": results,
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
        "algorithm": list(models),
        "fit_split": manifest["fit_split"],
        "calibration_split": None,
        "validation_split": manifest["development_split"],
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
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/clone-real-guitar-techs-v1"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--fit-per-profile", type=int, default=24)
    parser.add_argument("--development-per-profile", type=int, default=24)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    run(args)


if __name__ == "__main__":
    main()
