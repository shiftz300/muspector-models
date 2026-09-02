#!/usr/bin/env python3
"""Measure whether swapping an effect pair produces distinguishable audio."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np

from .data import RATE, TOPOLOGIES, _waveform, dry_sources, random_spec
from .render import RENDERERS, render_chain
from .spec import ORDER_PAIRS, ChainSpec, identifiable_order_targets, order_targets


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/corpus/guitar-effects-chains"
OUTPUT = ROOT / "remix/runs/order-identifiability.json"


def swapped(spec: ChainSpec, left: str, right: str) -> ChainSpec:
    effects = list(spec.effects)
    positions = {effect.kind: index for index, effect in enumerate(effects)}
    first, second = positions[left], positions[right]
    effects[first], effects[second] = effects[second], effects[first]
    return ChainSpec(tuple(effects))


def relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(float(np.sqrt(np.mean(left * left))), 1.0e-8)
    return float(np.sqrt(np.mean((left - right) ** 2)) / scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sources", type=int, default=4)
    args = parser.parse_args()

    paths = dry_sources(args.corpus, "valid")[: args.sources]
    values: dict[tuple[str, str, str, bool], list[float]] = defaultdict(list)
    for source_index, path in enumerate(paths):
        dry = _waveform(path)
        dry *= 0.22 / max(float(np.max(np.abs(dry))), 1.0e-5)
        for topology_index, topology in enumerate(TOPOLOGIES):
            if len(topology) < 2:
                continue
            rng = random.Random(20260830 + source_index * 10_007 + topology_index)
            spec = random_spec(rng, topology)
            _, present = order_targets(topology)
            _, identifiable = identifiable_order_targets(topology)
            for renderer in RENDERERS:
                original = render_chain(dry, spec, RATE, renderer)
                for pair_index, (left, right) in enumerate(ORDER_PAIRS):
                    if not present[pair_index]:
                        continue
                    alternative = render_chain(
                        dry, swapped(spec, left, right), RATE, renderer
                    )
                    values[(renderer, left, right, bool(identifiable[pair_index]))].append(
                        relative_difference(original, alternative)
                    )

    groups = []
    for (renderer, left, right, identifiable), distances in sorted(values.items()):
        groups.append(
            {
                "renderer": renderer,
                "pair": [left, right],
                "identifiable": identifiable,
                "samples": len(distances),
                "minimum": min(distances),
                "median": float(np.median(distances)),
                "maximum": max(distances),
            }
        )
    report = {
        "schema": 1,
        "metric": "rms(swapped-original) / rms(original)",
        "threshold": 1.0e-4,
        "sources": len(paths),
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
