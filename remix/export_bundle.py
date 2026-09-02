#!/usr/bin/env python3
"""Validate and package the complete Remixer model bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .delay_model import DelayControlEstimator
from .drive_model import DriveControlEstimator
from .model import PairedEstimator
from .quality import contract_manifest
from .reverb_model import ReverbControlEstimator
from .spec import CONTROL_NAMES, ORDER_PAIRS, SCHEMA_VERSION
from .train import RUN


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=RUN / "remixer-bundle.pt")
    parser.add_argument("--manifest", type=Path, default=RUN / "bundle-manifest.json")
    args = parser.parse_args()

    definitions = {
        "main": ("paired-estimator.pt", PairedEstimator),
        "drive": ("drive-estimator.pt", DriveControlEstimator),
        "delay": ("delay-estimator.pt", DelayControlEstimator),
        "reverb": ("reverb-estimator.pt", ReverbControlEstimator),
    }
    states, sources, counts = {}, {}, {}
    for name, (filename, model_type) in definitions.items():
        path = args.run / filename
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = model_type()
        model.load_state_dict(state)
        model.eval()
        states[name] = state
        sources[name] = {"file": filename, "sha256": sha256(path)}
        counts[name] = sum(parameter.numel() for parameter in model.parameters())

    payload = {
        "bundle_schema": 1,
        "chain_schema": SCHEMA_VERSION,
        "sample_rate": 44_100,
        "analysis_seconds": 5,
        "order_pairs": ORDER_PAIRS,
        "control_names": CONTROL_NAMES,
        "states": states,
        "audio_quality": contract_manifest(),
        "inference_contract": {
            "schema": 1,
            "inputs": ["aligned_clean_audio", "aligned_wet_audio", "active_effect_families"],
            "active_effect_source": "Inspector family detector",
            "alignment": "shared time offset; no automatic shift or time stretch",
            "outputs": [
                "chain_spec_physical_controls",
                "order_decision",
                "reconstruction_error",
                "audio_quality",
            ],
        },
    }
    order_search_metrics = args.run / "order-search-metrics.json"
    order_search = None
    if order_search_metrics.is_file():
        search_report = json.loads(order_search_metrics.read_text())
        order_search = {
            "renderers": search_report["search_renderers"],
            "margin_threshold": search_report["selected_margin_threshold"],
            "score": "gain-aligned normalized waveform MSE",
            "order_equivalence_db": search_report["order_equivalence_db"],
        }
        payload["order_search"] = order_search
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    manifest = {
        "schema": 1,
        "artifact": args.output.name,
        "artifact_sha256": sha256(args.output),
        "sources": sources,
        "parameter_counts": {**counts, "total": sum(counts.values())},
        "input_contract": {
            "paired": ["clean", "wet"],
            "sample_rate": 44_100,
            "seconds": 5,
            "channels": 1,
        },
        "output_contract": {
            "order_pairs": [list(pair) for pair in ORDER_PAIRS],
            "controls": list(CONTROL_NAMES),
        },
        "training_domains": {
            "main_order": ["DAFx25-real", "reference", "alternate"],
            "drive_controls": ["reference", "alternate", "stress"],
            "delay_controls": ["reference", "alternate", "stress", "pedalboard-0.9.24"],
            "reverb_controls": ["reference", "alternate", "stress", "pedalboard-0.9.24"],
        },
        "evaluation_domains": [
            "DAFx25-real",
            "reference",
            "alternate",
            "stress",
            "pedalboard-0.9.24",
        ],
        "excluded_final_renderer": "challenge",
        "audio_quality": contract_manifest(),
        "inference_contract": {
            "schema": 1,
            "reference": "python -m remix.inference",
            "requires_aligned_clean_wet": True,
            "requires_active_effect_families": True,
            "automatic_alignment": False,
        },
    }
    if order_search is not None:
        manifest["order_search"] = {
            **order_search,
            "metrics": {
                "file": order_search_metrics.name,
                "sha256": sha256(order_search_metrics),
            },
        }
    runtime_smoke = args.run / "runtime-smoke.json"
    if runtime_smoke.is_file():
        manifest["runtime_smoke"] = {
            "file": runtime_smoke.name,
            "sha256": sha256(runtime_smoke),
        }
    manifest["forward_renderer"] = {
        "status": "requires aligned real-device control sweeps",
        "capture_validator": "python -m remix.capture_manifest",
        "research": "remix/NEURAL_DSP_RESEARCH.md",
        "synthetic_self_distillation_promotable": False,
    }
    metrics = args.run / "metrics.json"
    if metrics.is_file():
        manifest["canonical_metrics"] = {
            "file": metrics.name,
            "sha256": sha256(metrics),
        }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
