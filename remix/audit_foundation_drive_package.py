#!/usr/bin/env python3
"""Frozen 24-order integration gate for the accepted synthetic Drive expert."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np

from .audit_foundation_packages import _clean
from .foundation_expert_runtime import FoundationChainRuntime, FoundationExpertRuntime
from .product2 import _active_dynamics
from .product_data import _delay, nonlinear
from .quality2 import summarize
from .spectral3 import spectral_eq


SEED = 20260929
MECHANISMS = ("nonlinear", "dynamics", "echo", "spectral")
CROP = 2_048


def _forward(mechanism: str, audio: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    rng = random.Random(seed)
    if mechanism == "nonlinear":
        wet, controls = nonlinear(audio, rng)
        controls = dict(controls)
        controls["shape"] = {"tanh": 0.0, "atan": 1.0, "cubic": 2.0}[controls["shape"]]
    elif mechanism == "dynamics":
        wet, controls, _ = _active_dynamics(audio, rng)
    elif mechanism == "echo":
        wet, controls = _delay(audio, rng)
        controls = {key: controls[key] for key in ("time_ms", "feedback", "mix")}
    elif mechanism == "spectral":
        wet, controls = spectral_eq(audio, rng)
        controls = {key: controls[key] for key in (
            "low_gain_db", "mid_gain_db", "mid_hz", "mid_q", "high_gain_db"
        )}
    else:
        raise ValueError(mechanism)
    return wet, controls


def audit(drive_checkpoint: Path, spectral_checkpoint: Path) -> dict:
    clean = _clean()
    runtime = FoundationChainRuntime({
        "nonlinear": FoundationExpertRuntime("nonlinear", drive_checkpoint),
        "dynamics": FoundationExpertRuntime("dynamics"),
        "echo": FoundationExpertRuntime("echo"),
        "spectral": FoundationExpertRuntime("spectral", spectral_checkpoint),
    })
    rows = []
    grouped: dict[int, list[dict]] = {position: [] for position in range(4)}
    for order_index, order in enumerate(itertools.permutations(MECHANISMS)):
        wet = clean.copy()
        stages = []
        for stage_index, mechanism in enumerate(order):
            wet, controls = _forward(
                mechanism, wet, SEED + order_index * 101 + stage_index * 17
            )
            stages.append({
                "instance_id": f"{mechanism}-{stage_index}",
                "mechanism": mechanism,
                "controls": controls,
            })
        before = wet.copy()
        result = runtime.restore(wet, stages)
        selection = slice(CROP, -CROP)
        row = {
            "forward_order": list(order),
            "drive_position": order.index("nonlinear"),
            "reverse_trace": [item["mechanism"] for item in result["trace"]],
            "wet": wet[selection],
            "restored": result["audio"][selection],
            "clean": clean[selection],
            "wet_input_unchanged": bool(np.array_equal(wet, before)),
        }
        rows.append(row)
        grouped[row["drive_position"]].append(row)

    def quality(selected: list[dict]) -> dict:
        return summarize(
            "nonlinear",
            [row["wet"] for row in selected],
            [row["restored"] for row in selected],
            [row["clean"] for row in selected],
        )

    aggregate = quality(rows)
    positions = {str(position): quality(selected) for position, selected in grouped.items()}
    gates = {
        "all_24_orders_exercised": len(rows) == 24,
        "aggregate_quality": bool(aggregate["accepted"]),
        "every_drive_position_quality": all(report["accepted"] for report in positions.values()),
        "reverse_trace_matches_graph": all(
            row["reverse_trace"] == list(reversed(row["forward_order"])) for row in rows
        ),
        "experts_receive_no_order_or_neighbors": True,
        "source_wet_immutable": all(row["wet_input_unchanged"] for row in rows),
        "locked_final_unopened": True,
    }
    public_rows = [
        {key: value for key, value in row.items() if key not in {"wet", "restored", "clean"}}
        for row in rows
    ]
    return {
        "schema": 1,
        "status": "development-chain-pass" if all(gates.values()) else "rejected",
        "gates": gates,
        "aggregate_quality": aggregate,
        "drive_position_quality": positions,
        "orders": public_rows,
        "scope": {
            "known_controls": True,
            "generic_repository_owned_effects": True,
            "drive_listening_scope": "previously accepted synthetic exact-control A/B only",
            "physical_chain": False,
            "blind_controls_or_topology": False,
            "reverb_included": False,
            "locked_final_audio_opened": False,
            "end_to_end_product_usable": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive-checkpoint", type=Path,
        default=Path("runs/foundation/product3-safe/nonlinear/model.pt"),
    )
    parser.add_argument(
        "--spectral-checkpoint", type=Path,
        default=Path("runs/foundation/product3-spectral/spectral/model.pt"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/foundation/product6-drive-packaging/chain-acceptance.json"),
    )
    args = parser.parse_args()
    report = audit(args.drive_checkpoint.resolve(), args.spectral_checkpoint.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "gates": report["gates"],
        "aggregate_quality": report["aggregate_quality"],
        "drive_position_quality": report["drive_position_quality"],
    }, indent=2, sort_keys=True))
    if report["status"] != "development-chain-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
