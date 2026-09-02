#!/usr/bin/env python3
"""Audit product renderer strata and quality-metric coverage in memory only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .foundation_model import Expert, MECHANISMS
from .license_gate import require_product_weights
from .product_data import (
    MAX_DELAY_SECONDS,
    MAX_RELEASE_MS,
    ProductPairs,
    _rir,
    audit as audit_product_data,
    rir_splits,
)
from .restoration_quality import (
    _crest_error,
    _relative_error,
    _transient_envelope,
    measure,
)


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _load_model(run: Path, mechanism: str) -> tuple[Expert, str]:
    checkpoint = run / mechanism / "model.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = payload["architecture"]
    model = Expert(
        mechanism,
        channels=architecture["channels"],
        blocks=architecture["blocks"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def _row(model: Expert, item: dict) -> dict:
    wet = item["wet"].numpy()
    clean = item["clean"].numpy()
    with torch.inference_mode():
        restored = model(item["wet"].unsqueeze(0))[0].numpy()
    clean_rms = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2)))
    wet_rms = float(np.sqrt(np.mean(wet.astype(np.float64) ** 2)))
    clean_attack = _transient_envelope(clean)
    wet_attack = _transient_envelope(wet)
    return {
        "stratum": str(item["controls"].get("kind", item["controls"].get("shape", model.mechanism))),
        "clean_rms": clean_rms,
        "wet_rms": wet_rms,
        "wet_clean_rms_ratio": wet_rms / max(clean_rms, 1.0e-12),
        "wet_clean_mae_ratio": float(np.mean(np.abs(wet - clean)) / max(np.mean(np.abs(clean)), 1.0e-6)),
        "wet_attack_error": _relative_error(wet_attack, clean_attack),
        "wet_crest_error": _crest_error(wet, clean),
        "identity": measure(wet, wet, clean),
        "oracle": measure(wet, clean, clean),
        "candidate": measure(wet, restored, clean),
    }


def _summarize(rows: list[dict]) -> dict:
    candidate_names = (
        "spectral_improvement",
        "high_band_improvement",
        "transient_improvement",
        "dynamics_improvement",
    )
    candidates = [row["candidate"] for row in rows]
    baseline_attack = [row["wet_attack_error"] for row in rows]
    baseline_crest = [row["wet_crest_error"] for row in rows]
    return {
        "examples": len(rows),
        "clean_rms_median": _median([row["clean_rms"] for row in rows]),
        "wet_clean_rms_ratio_median": _median([row["wet_clean_rms_ratio"] for row in rows]),
        "wet_clean_mae_ratio_median": _median([row["wet_clean_mae_ratio"] for row in rows]),
        "minimum_wet_attack_error": float(min(baseline_attack)),
        "minimum_wet_crest_error": float(min(baseline_crest)),
        "denominator_sensitive": bool(min(baseline_attack) < 1.0e-4 or min(baseline_crest) < 1.0e-4),
        "identity_pass_fraction": float(np.mean([row["identity"]["passed"] for row in rows])),
        "oracle_pass_fraction": float(np.mean([row["oracle"]["passed"] for row in rows])),
        "candidate_pass_fraction": float(np.mean([row["candidate"]["passed"] for row in rows])),
        "candidate_median": {
            name: _median([candidate[name] for candidate in candidates]) for name in candidate_names
        },
        "candidate_metric_abs_max": {
            name: float(max(abs(candidate[name]) for candidate in candidates)) for name in candidate_names
        },
        "candidate_metrics_bounded": bool(
            all(abs(candidate[name]) <= 1.0 + 1.0e-9 for candidate in candidates for name in candidate_names)
        ),
    }


def audit(workspace: Path, run: Path, *, samples: int = 24, frames: int = 4096) -> dict:
    workspace = workspace.resolve()
    run = run.resolve()
    product_audit = audit_product_data(workspace)
    reports = {}
    rir_max_frames = max(len(_rir(path)) for paths in rir_splits(workspace).values() for path in paths)
    context_requirements = {
        "nonlinear": 1,
        "dynamics": round(MAX_RELEASE_MS * 48.0),
        "temporal": max(round(MAX_DELAY_SECONDS * 48_000), rir_max_frames),
    }
    context_contract = {}
    for mechanism in MECHANISMS:
        model, checkpoint_sha256 = _load_model(run, mechanism)
        model_context = model.manifest()["input_context_frames"]
        context_contract[mechanism] = {
            "model_input_context_frames": model_context,
            "renderer_required_history_frames": context_requirements[mechanism],
            "passes": model_context >= context_requirements[mechanism],
        }
        dataset = ProductPairs(workspace, mechanism, "development", samples, frames, 20260905)
        grouped = defaultdict(list)
        for index in range(len(dataset)):
            row = _row(model, dataset[index])
            grouped[row["stratum"]].append(row)
        reports[mechanism] = {
            "checkpoint_sha256": checkpoint_sha256,
            "source_ids": sorted(dataset.authorization["sources"]),
            "research_source_ids": [],
            "strata": {name: _summarize(rows) for name, rows in sorted(grouped.items())},
        }
    product_sources = set(product_audit["authorization"]["sources"])
    require_product_weights(workspace / "remix/data_sources.json", sorted(product_sources))
    return {
        "schema": 1,
        "status": "coverage-diagnostic-complete",
        "run": str(run),
        "split": "development",
        "samples_per_mechanism": samples,
        "frames": frames,
        "product_sources_only": True,
        "research_data_isolated": True,
        "physical_audio_devices_used": False,
        "generated_audio_written": False,
        "source_audio_read_only": True,
        "oracle_contract_passes": all(
            stratum["oracle_pass_fraction"] == 1.0
            for report in reports.values()
            for stratum in report["strata"].values()
        ),
        "quality_metric_contract": {
            "improvement_is_bounded": all(
                stratum["candidate_metrics_bounded"]
                for report in reports.values()
                for stratum in report["strata"].values()
            ),
            "denominator_sensitive_strata": [
                f"{mechanism}/{name}"
                for mechanism, report in reports.items()
                for name, stratum in report["strata"].items()
                if stratum["denominator_sensitive"]
            ],
        },
        "history_context_contract": {
            "status": "failed" if not all(item["passes"] for item in context_contract.values()) else "passed",
            "rir_max_frames": rir_max_frames,
            "mechanisms": context_contract,
            "formal_baseline_blocked": not all(item["passes"] for item in context_contract.values()),
        },
        "product_data": product_audit,
        "mechanisms": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--run", type=Path, default=Path("runs/foundation/product1"))
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/product1/coverage-audit.json"))
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--frames", type=int, default=4096)
    args = parser.parse_args()
    report = audit(args.workspace, args.run, samples=args.samples, frames=args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
