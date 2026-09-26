#!/usr/bin/env python3
"""Fit-only EG-IPT diagnostic for schema-17 checkpoints; never a product gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .eg_ipt_amp_data import EgIptAmpPairs
from .eg_ipt_amp_model import (
    EgIptFixedProfileAmpInverse,
    EgIptFixedProfileAmpTransientInverse,
)
from .quality2 import summarize
from .rusty_amp_stage_model import RustyAmpStagewiseGrayBoxInverse


def _load(path: Path) -> tuple[torch.nn.Module, str]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    architecture = payload.get("architecture", {})
    schema = architecture.get("schema")
    if schema == 17:
        model = RustyAmpStagewiseGrayBoxInverse(
            24, architecture.get("depth", 8), architecture.get("condition_size", 64)
        )
    elif schema == 19:
        model = EgIptFixedProfileAmpInverse(
            architecture.get("depth", 8), architecture.get("condition_size", 64)
        )
    elif schema == 20:
        model = EgIptFixedProfileAmpTransientInverse(
            architecture.get("transient_channels", 16),
            architecture.get("transient_depth", 8),
        )
    else:
        raise ValueError(f"EG-IPT diagnostic requires schema 17, 19 or 20: {path}")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(
    workspace: Path, checkpoints: list[Path], samples: int,
    reservation: Path | None = None,
) -> dict:
    reserved = None
    split = "internal_calibration"
    if reservation is not None:
        manifest = json.loads(reservation.resolve().read_text())
        if manifest.get("status") != "reserved-internal-v2-fit-only-not-product-validation":
            raise PermissionError("EG-IPT reservation is not admitted")
        reserved = frozenset(manifest["relatives"])
        split = manifest.get("source_partition", "reserved")
    dataset = EgIptAmpPairs(
        workspace.resolve(), split, samples, 16_384, 4_096, 20260977,
        include_relatives=reserved,
    )
    results = []
    for checkpoint in checkpoints:
        model, digest = _load(checkpoint.resolve())
        aggregate = ([], [], [])
        pickups = defaultdict(lambda: ([], [], []))
        techniques = defaultdict(lambda: ([], [], []))
        with torch.inference_mode():
            for index in range(len(dataset)):
                row = dataset[index]
                if isinstance(model, RustyAmpStagewiseGrayBoxInverse):
                    restored, _, _ = model(
                        row["wet"].unsqueeze(0),
                        tone_reference=row["tone_reference"].unsqueeze(0),
                    )
                else:
                    restored, _, _ = model(row["wet"].unsqueeze(0))
                start, end = row["crop_start"], row["crop_end"]
                values = (
                    row["wet"][start:end].numpy(),
                    restored[0, start:end].numpy().astype(np.float32),
                    row["clean"][start:end].numpy(),
                )
                for collection in (aggregate, pickups[row["pickup"]], techniques[row["technique"]]):
                    for target, value in zip(collection, values, strict=True):
                        target.append(value)
        results.append({
            "checkpoint": str(checkpoint.resolve()),
            "sha256": digest,
            "aggregate": summarize("amp", *aggregate),
            "pickups": {
                key: summarize("amp", *value) for key, value in sorted(pickups.items())
            },
            "techniques": {
                key: summarize("amp", *value) for key, value in sorted(techniques.items())
            },
        })
    return {
        "schema": 1,
        "status": "fit-only-internal-diagnostic-not-promotable",
        "source": "eg-ipt",
        "samples": samples,
        "split": split,
        "reservation": None if reservation is None else str(reservation.resolve()),
        "product_gate": False,
        "locked_final_audio_opened": False,
        "graph_order_input": False,
        "neighbor_effect_input": False,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=171)
    parser.add_argument("--reservation", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    args = parser.parse_args()
    report = evaluate(args.workspace, args.checkpoints, args.samples, args.reservation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps([
        {
            "checkpoint": row["checkpoint"],
            "accepted": row["aggregate"]["accepted"],
            "pass_fraction": row["aggregate"]["pass_fraction"],
            "metrics": {
                name: value["median_reduction"]
                for name, value in row["aggregate"]["metrics"].items()
            },
        }
        for row in report["results"]
    ], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
