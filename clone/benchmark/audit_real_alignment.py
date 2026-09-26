"""Audit local residual alignment for the licensed Guitar-TECHS P1/P2 pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import correlate, correlation_lags

from .run_real import PROFILE_IDS, _load_rows


SPLITS = ("fit", "calibration", "development")
RESIDUAL_LIMIT = 512
TOLERANCES = (8, 16, 32, 64, 128)


def _git_revision(workspace: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _estimate_residual_lag(x: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    # Differentiate before correlation so the diagnostic emphasizes note
    # edges rather than treating the cabinet's broad spectral shape as delay.
    x = np.diff(x.astype(np.float64), prepend=float(x[0]))
    y = np.diff(y.astype(np.float64), prepend=float(y[0]))
    x -= x.mean()
    y -= y.mean()
    values = correlate(y, x, mode="full", method="fft")
    lags = correlation_lags(len(y), len(x), mode="full")
    selected = np.abs(lags) <= RESIDUAL_LIMIT
    values = values[selected]
    lags = lags[selected]
    index = int(np.argmax(np.abs(values)))
    scale = max(float(np.linalg.norm(x) * np.linalg.norm(y)), 1.0e-12)
    return int(lags[index]), float(abs(values[index]) / scale)


def _summarize(rows: list[dict]) -> dict:
    lags = np.asarray([row["residual_lag_frames"] for row in rows], dtype=np.float64)
    absolute = np.abs(lags)
    correlations = np.asarray([row["absolute_correlation"] for row in rows], dtype=np.float64)
    return {
        "examples": len(rows),
        "signed_lag_frames": {
            "mean": float(np.mean(lags)),
            "p50": float(np.quantile(lags, 0.50)),
            "p90": float(np.quantile(lags, 0.90)),
            "min": int(np.min(lags)),
            "max": int(np.max(lags)),
        },
        "absolute_lag_frames": {
            "mean": float(np.mean(absolute)),
            "p50": float(np.quantile(absolute, 0.50)),
            "p90": float(np.quantile(absolute, 0.90)),
            "max": int(np.max(absolute)),
        },
        "absolute_correlation": {
            "mean": float(np.mean(correlations)),
            "p10": float(np.quantile(correlations, 0.10)),
            "min": float(np.min(correlations)),
        },
        "outside_tolerance": {
            str(tolerance): int(np.sum(absolute > tolerance))
            for tolerance in TOLERANCES
        },
    }


def run(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace alignment audit output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    from remix.license_gate import require_product_weights

    authorization = require_product_weights(
        workspace / "remix/data_sources.json", ("guitar-techs",),
    )
    all_summaries = {}
    all_rows = {}
    for split_index, split in enumerate(SPLITS):
        rows = _load_rows(workspace, split, args.per_profile, 20260951 + split_index)
        grouped = defaultdict(list)
        for row in rows:
            target_x = row.x[row.target_start:]
            target_y = row.y[row.target_start:]
            lag, absolute_correlation = _estimate_residual_lag(target_x, target_y)
            grouped[row.profile_id].append({
                "index": row.index,
                "residual_lag_frames": lag,
                "absolute_correlation": absolute_correlation,
            })
        all_rows[split] = {
            profile_id: values for profile_id, values in sorted(grouped.items())
        }
        all_summaries[split] = {
            profile_id: _summarize(values)
            for profile_id, values in sorted(grouped.items())
        }
        print(json.dumps({"stage": split, "summary": all_summaries[split]}), flush=True)

    manifest = {
        "schema": 1,
        "kind": "black-box-forward-system-identification-real-alignment-audit",
        "source_id": "guitar-techs",
        "profiles": list(PROFILE_IDS),
        "splits": list(SPLITS),
        "fit_examples_per_profile": args.per_profile,
        "calibration_examples_per_profile": args.per_profile,
        "development_examples_per_profile": args.per_profile,
        "residual_search_limit_frames": RESIDUAL_LIMIT,
        "differentiated_before_correlation": True,
        "p3": "excluded by AmpCabPairs and not decoded",
    }
    payload = {
        "schema": 1,
        "status": "diagnostic-real-alignment",
        "manifest": manifest,
        "interpretation": {
            "residual_lag": "local correlation offset after the configured fixed P1/P2 lag",
            "not_a_hardware_delay_claim": True,
            "no_audio_rewritten": True,
            "no_model_selection": True,
        },
        "authorization": authorization,
        "summaries": all_summaries,
        "observations": all_rows,
        "fit": {"git_revision": _git_revision(workspace)},
        "provenance": {
            "source_id": "guitar-techs",
            "license": "CC BY 4.0; attribution is in remix/data_sources.json",
            "p3_audio_opened": False,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "cross-correlation can select a note-edge ambiguity rather than physical delay",
            "only two fixed named Amp+cab+mic profiles",
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
        "algorithm": "differentiated local cross-correlation alignment audit",
        "fit_split": "Guitar-TECHS fit time interval",
        "calibration_split": "Guitar-TECHS calibration time interval",
        "validation_split": "Guitar-TECHS development time interval",
        "locked": False,
        "status": "diagnostic",
        "metrics": str(output / "metrics.json"),
    })
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/clone-real-guitar-techs-alignment-v1"))
    parser.add_argument("--per-profile", type=int, default=24)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
