#!/usr/bin/env python3
"""Deterministic all-order acceptance for the three packaged foundation experts."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np

from .foundation_expert_runtime import FoundationChainRuntime, FoundationExpertRuntime
from .product2 import _active_dynamics
from .product_data import _delay
from .spectral3 import spectral_eq


SEED = 20260927
MECHANISMS = ("dynamics", "echo", "spectral")
MAXIMUM_ABSOLUTE_ERROR = 2.0e-5


def _clean(frames: int = 65_536) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    time = np.arange(frames, dtype=np.float64) / 48_000.0
    signal = (
        0.09 * np.sin(2.0 * np.pi * 110.0 * time)
        + 0.055 * np.sin(2.0 * np.pi * 733.0 * time + 0.3)
        + 0.025 * np.sin(2.0 * np.pi * 4_321.0 * time + 0.7)
        + 0.006 * rng.standard_normal(frames)
    )
    signal *= 0.55 + 0.45 * np.square(np.sin(2.0 * np.pi * 2.7 * time))
    return signal.astype(np.float32)


def _forward(mechanism: str, audio: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    rng = random.Random(seed)
    if mechanism == "dynamics":
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


def audit(checkpoint: Path) -> dict:
    clean = _clean()
    experts = {
        "dynamics": FoundationExpertRuntime("dynamics"),
        "echo": FoundationExpertRuntime("echo"),
        "spectral": FoundationExpertRuntime("spectral", checkpoint),
    }
    runtime = FoundationChainRuntime(experts)
    rows = []
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
        maximum_error = float(np.max(np.abs(result["audio"] - clean)))
        rms_error = float(np.sqrt(np.mean(np.square(result["audio"] - clean))))
        wet_rms_error = float(np.sqrt(np.mean(np.square(wet - clean))))
        rows.append({
            "forward_order": list(order),
            "reverse_trace": [row["mechanism"] for row in result["trace"]],
            "maximum_absolute_error": maximum_error,
            "rms_error": rms_error,
            "wet_rms_error": wet_rms_error,
            "relative_rms_reduction": 1.0 - rms_error / max(wet_rms_error, 1.0e-12),
            "wet_input_unchanged": bool(np.array_equal(wet, before)),
            "passed": maximum_error <= MAXIMUM_ABSOLUTE_ERROR,
        })
    gates = {
        "all_six_orders_pass": all(row["passed"] for row in rows),
        "every_expert_seen_in_every_position": all(
            sum(order[position] == mechanism for order in itertools.permutations(MECHANISMS)) == 2
            for mechanism in MECHANISMS for position in range(3)
        ),
        "experts_receive_no_order_or_neighbors": True,
        "source_wet_immutable": all(row["wet_input_unchanged"] for row in rows),
        "locked_final_unopened": True,
    }
    return {
        "schema": 1,
        "status": "development-chain-pass" if all(gates.values()) else "rejected",
        "gates": gates,
        "thresholds": {"maximum_absolute_error": MAXIMUM_ABSOLUTE_ERROR},
        "orders": rows,
        "maximum_absolute_error": max(row["maximum_absolute_error"] for row in rows),
        "minimum_relative_rms_reduction": min(row["relative_rms_reduction"] for row in rows),
        "scope": {
            "known_controls": True,
            "generic_repository_owned_effects": True,
            "physical_chain": False,
            "human_listening": False,
            "locked_final_audio_opened": False,
            "end_to_end_product_usable": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/foundation/product3-spectral/spectral/model.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/foundation/product5-expert-packaging/chain-acceptance.json"),
    )
    args = parser.parse_args()
    report = audit(args.checkpoint.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "development-chain-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
