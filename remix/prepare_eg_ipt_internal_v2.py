#!/usr/bin/env python3
"""Freeze balanced EG-IPT pairs never sampled by the v30 fixed-profile run."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .eg_ipt_amp_data import EgIptAmpPairs, _partition, discover
from .train_egdb_pg_amp import SEED


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("remix/eg_ipt_internal_v2.json")
    )
    parser.add_argument("--v30-train-samples", type=int, default=3420)
    parser.add_argument("--v30-calibration-samples", type=int, default=342)
    parser.add_argument("--pairs-per-stratum", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace EG-IPT v2 reservation: {output}")
    dataset = EgIptAmpPairs(
        args.workspace.resolve(), "fit", args.v30_train_samples,
        16_384, 4_096, SEED + 61,
    )
    seen = set()
    for index in range(len(dataset)):
        _, rows = dataset.strata[index % len(dataset.strata)]
        rng = random.Random(dataset.seed + index * 104729)
        seen.add(rows[rng.randrange(len(rows))].relative.as_posix())
    calibration = EgIptAmpPairs(
        args.workspace.resolve(), "internal_calibration", args.v30_calibration_samples,
        16_384, 4_096, SEED + 62,
    )
    calibration_seen = set()
    for index in range(len(calibration)):
        _, rows = calibration.strata[index % len(calibration.strata)]
        rng = random.Random(calibration.seed + index * 104729)
        calibration_seen.add(rows[rng.randrange(len(rows))].relative.as_posix())
    available = defaultdict(list)
    for pair in discover(args.workspace.resolve()):
        relative = pair.relative.as_posix()
        if relative not in seen and relative not in calibration_seen:
            available[(pair.pickup, pair.technique)].append(pair)
    selected = []
    for stratum, rows in sorted(available.items()):
        ranked = sorted(
            rows,
            key=lambda pair: hashlib.sha256(
                f"eg-ipt-v2:{pair.relative.as_posix()}".encode()
            ).digest(),
        )
        if len(ranked) < args.pairs_per_stratum:
            raise ValueError(f"not enough unseen EG-IPT pairs for {stratum}")
        selected.extend(ranked[: args.pairs_per_stratum])
    relatives = sorted(pair.relative.as_posix() for pair in selected)
    report = {
        "schema": 1,
        "status": "reserved-internal-v2-fit-only-not-product-validation",
        "source_id": "eg-ipt",
        "selection": "six v30-unseen pairs per pickup-technique stratum by frozen SHA-256 rank",
        "v30_train_seed": SEED + 61,
        "v30_train_samples": args.v30_train_samples,
        "v30_unique_pairs_seen": len(seen),
        "source_partition": "reserved",
        "v30_calibration_samples": args.v30_calibration_samples,
        "v30_calibration_unique_pairs_seen": len(calibration_seen),
        "pairs_per_stratum": args.pairs_per_stratum,
        "strata": len(available),
        "pairs": len(relatives),
        "relatives": relatives,
        "original_partition_counts": {
            name: sum(_partition(pair) == name for pair in selected)
            for name in ("fit", "internal_calibration")
        },
        "relatives_sha256": hashlib.sha256("\n".join(relatives).encode()).hexdigest(),
        "product_gate": False,
        "locked_final_audio_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "v30_unique_pairs_seen", "v30_calibration_unique_pairs_seen",
        "strata", "pairs", "relatives_sha256"
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
