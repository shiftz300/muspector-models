#!/usr/bin/env python3
"""Evaluate the audio-safe Order 2 wrapper on development or sealed sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import SyntheticControlDataset
from .evaluate_order_search import make_domains
from .order import OrderRuntime
from .pedalboard_data import PedalboardControlDataset
from .spec import KINDS, order_targets


MINIMUM = {
    "real": {"exact": 0.72, "pairwise": 0.80},
    "reference": {"exact": 0.70, "pairwise": 0.78},
    "alternate": {"exact": 0.70, "pairwise": 0.78},
    "stress": {"exact": 0.70, "pairwise": 0.78},
    "pedalboard": {"exact": 0.65, "pairwise": 0.75},
}


def guitarjam(root: Path, samples: int) -> dict:
    dry = sorted((root / "guitar_jam/test").glob("*.wav"))
    if len(dry) != 54:
        raise ValueError(f"expected 54 sealed GuitarJam test files, found {len(dry)}")
    domains = {
        name: SyntheticControlDataset(
            dry,
            samples,
            seed=20260921,
            renderers=(name,),
            order_equivalence_db=-30.0,
        )
        for name in ("reference", "alternate", "stress")
    }
    domains["pedalboard"] = PedalboardControlDataset(
        dry,
        samples,
        seed=20260922,
        order_equivalence_db=-30.0,
    )
    return domains


def _prediction(order: tuple[str, ...]) -> np.ndarray:
    return np.asarray(order_targets(order)[0], dtype=bool)


def _metrics(rows: list[dict], key: str) -> dict:
    correct = relations = exact = eligible = 0
    for row in rows:
        mask = np.asarray(row["mask"], dtype=bool)
        if not mask.any():
            continue
        truth = np.asarray(row["truth"], dtype=bool)
        prediction = _prediction(tuple(row[key]))
        matches = prediction == truth
        correct += int((matches & mask).sum())
        relations += int(mask.sum())
        exact += int((matches | ~mask).all())
        eligible += 1
    return {
        "exact": exact / max(eligible, 1),
        "pairwise": correct / max(relations, 1),
        "examples": eligible,
        "relations": relations,
    }


def summarize(rows: list[dict]) -> dict:
    result = {name: _metrics(rows, name) for name in ("original", "proposed", "selected")}
    proposed_delta = np.asarray([row["proposed_error"] - row["original_error"] for row in rows])
    selected_delta = np.asarray([row["selected_error"] - row["original_error"] for row in rows])
    result["audio"] = {
        "proposed_regressions": int((proposed_delta > 0.0).sum()),
        "selected_regressions": int((selected_delta > 0.0).sum()),
        "proposed_max_delta": float(proposed_delta.max(initial=0.0)),
        "selected_max_delta": float(selected_delta.max(initial=0.0)),
        "original_mean": float(np.mean([row["original_error"] for row in rows])),
        "proposed_mean": float(np.mean([row["proposed_error"] for row in rows])),
        "selected_mean": float(np.mean([row["selected_error"] for row in rows])),
    }
    result["guarded"] = sum(row["guarded"] for row in rows)
    result["inputs_unchanged"] = all(row["inputs_unchanged"] for row in rows)
    return result


def accepted(domains: dict[str, dict]) -> tuple[bool, list[str]]:
    failures = []
    for name, metrics in domains.items():
        minimum = MINIMUM[name]
        proposed = metrics["proposed"]
        selected = metrics["selected"]
        original = metrics["original"]
        # The classifier and the audio-safety policy have different jobs.
        # Grade recognition on the unguarded proposal, then require the
        # delivered selection to preserve the baseline and audio quality.
        if proposed["exact"] < minimum["exact"]:
            failures.append(f"{name}.exact")
        if proposed["pairwise"] < minimum["pairwise"]:
            failures.append(f"{name}.pairwise")
        if selected["exact"] < original["exact"] - 0.01:
            failures.append(f"{name}.exact-regression")
        if selected["pairwise"] < original["pairwise"] - 0.015:
            failures.append(f"{name}.pairwise-regression")
        if metrics["audio"]["selected_regressions"]:
            failures.append(f"{name}.audio-regression")
        if not metrics["inputs_unchanged"]:
            failures.append(f"{name}.input-mutation")
    return not failures, failures


def evaluate(source: str, corpus: Path, run: Path, output: Path, samples: int) -> dict:
    if output.exists():
        raise FileExistsError(f"order report already exists: {output}")
    if source == "valid":
        datasets = make_domains(corpus, "valid", samples, -30.0)
    elif source == "calibrate":
        datasets = make_domains(corpus, "calibrate", samples, -30.0)
    elif source == "guitarjam":
        datasets = guitarjam(corpus, samples)
    else:
        raise ValueError(f"unknown source {source}")
    runtime = OrderRuntime(run)
    domains = {}
    for domain, dataset in datasets.items():
        rows = []
        for index in range(len(dataset)):
            item = dataset[index]
            dry = item["dry"].numpy()
            wet = item["wet"].numpy()
            before = hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest()
            topology = tuple(
                KINDS[int(value)] for value in item["topology"].tolist() if int(value) >= 0
            )
            report = runtime.infer(dry, wet, tuple(sorted(topology)))
            safe = report["order2"]
            rows.append(
                {
                    "truth": item["order"].numpy().tolist(),
                    "mask": item["order_mask"].numpy().tolist(),
                    "original": safe.get("original", list(topology)),
                    "proposed": safe.get("proposed", list(topology)),
                    "selected": safe.get("selected", list(topology)),
                    "original_error": safe.get("original_error", 0.0),
                    "proposed_error": safe.get("proposed_error", 0.0),
                    "selected_error": safe.get("selected_error", 0.0),
                    "guarded": bool(safe["guarded"]),
                    "inputs_unchanged": before
                    == hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest(),
                }
            )
            if (index + 1) % 32 == 0 or index + 1 == len(dataset):
                print(json.dumps({"domain": domain, "done": index + 1, "total": len(dataset)}), flush=True)
        domains[domain] = summarize(rows)
    passed, failures = accepted(domains)
    runtime.assert_artifacts_unchanged()
    report = {
        "schema": 1,
        "source": source,
        "samples": samples,
        "gate_semantics": {
            "recognition": "proposed",
            "delivery": "selected",
            "audio_regression_tolerance": 0.0,
        },
        "accepted": passed,
        "status": "accepted-development" if passed else "rejected",
        "minimum": {name: MINIMUM[name] for name in domains},
        "domains": domains,
        "failures": failures,
        "physical_audio_devices_used": False,
        "source_audio_modified": False,
        "automatic_normalization": False,
        "automatic_limiting": False,
        "lossy_reencoding": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("valid", "calibrate", "guitarjam"), required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=160)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    report = evaluate(args.source, args.corpus, args.run, args.output, args.samples)
    print(json.dumps({"accepted": report["accepted"], "failures": report["failures"]}))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
