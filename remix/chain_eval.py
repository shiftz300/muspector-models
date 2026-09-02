#!/usr/bin/env python3
"""Replay the complete abstention-first chain on development data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .chain import ChainRuntime
from .evaluate_order_search import make_domains
from .spec import CONTROL_NAMES, KINDS, order_targets


GATES = {
    "coverage": 0.35,
    "family_exact": 0.98,
    "order_exact": 0.65,
    "order_pairwise": 0.75,
    "knob_mae": 0.13,
    "knob_p95": 0.35,
    "control_mae": 0.28,
    "control_p95": 0.65,
    "audio_regressions": 0,
    "input_mutations": 0,
}


def _digest(dry: np.ndarray, wet: np.ndarray) -> str:
    value = hashlib.sha256()
    value.update(dry.tobytes())
    value.update(wet.tobytes())
    return value.hexdigest()


def _topology(item: dict) -> tuple[str, ...]:
    return tuple(KINDS[int(value)] for value in item["topology"].tolist() if int(value) >= 0)


def summarize(rows: list[dict]) -> dict:
    accepted = [row for row in rows if row["accepted"]]
    ordered = [row for row in accepted if any(row["order_mask"])]
    relation_correct = relation_count = proposed_exact = selected_exact = 0
    audio_regressions = 0
    controls = {name: [] for name in CONTROL_NAMES}
    for row in ordered:
        truth = np.asarray(row["order_truth"], dtype=bool)
        mask = np.asarray(row["order_mask"], dtype=bool)
        proposed = np.asarray(order_targets(tuple(row["proposed"]))[0], dtype=bool)
        selected = np.asarray(order_targets(tuple(row["selected"]))[0], dtype=bool)
        proposed_matches = proposed == truth
        selected_matches = selected == truth
        relation_correct += int((proposed_matches & mask).sum())
        relation_count += int(mask.sum())
        proposed_exact += int((proposed_matches | ~mask).all())
        selected_exact += int((selected_matches | ~mask).all())
        audio_regressions += int(row["selected_error"] > row["original_error"])
    for row in accepted:
        if row["domain"] == "real":
            continue
        for name, error in row["controls"].items():
            controls[name].append(float(error))
    values = [value for per_control in controls.values() for value in per_control]
    return {
        "examples": len(rows),
        "accepted": len(accepted),
        "coverage": len(accepted) / max(len(rows), 1),
        "family_exact": sum(row["family_exact"] for row in accepted) / max(len(accepted), 1),
        "order": {
            "examples": len(ordered),
            "relations": relation_count,
            "proposed_exact": proposed_exact / max(len(ordered), 1),
            "proposed_pairwise": relation_correct / max(relation_count, 1),
            "selected_exact": selected_exact / max(len(ordered), 1),
            "audio_regressions": audio_regressions,
        },
        "knob": {
            "values": len(values),
            "macro_mae": float(np.mean(values)) if values else 0.0,
            "macro_p95": float(np.quantile(values, 0.95)) if values else 0.0,
            "controls": {
                name: {
                    "values": len(errors),
                    "mae": float(np.mean(errors)) if errors else 0.0,
                    "p95": float(np.quantile(errors, 0.95)) if errors else 0.0,
                }
                for name, errors in controls.items()
            },
        },
        "input_mutations": sum(not row["inputs_unchanged"] for row in rows),
    }


def failures(metrics: dict) -> list[str]:
    result = []
    if metrics["coverage"] < GATES["coverage"]: result.append("coverage")
    if metrics["family_exact"] < GATES["family_exact"]: result.append("family-exact")
    if metrics["order"]["proposed_exact"] < GATES["order_exact"]: result.append("order-exact")
    if metrics["order"]["proposed_pairwise"] < GATES["order_pairwise"]: result.append("order-pairwise")
    if metrics["knob"]["macro_mae"] > GATES["knob_mae"]: result.append("knob-mae")
    if metrics["knob"]["macro_p95"] > GATES["knob_p95"]: result.append("knob-p95")
    for name, values in metrics["knob"]["controls"].items():
        if not values["values"]: result.append(f"{name}.missing")
        elif values["mae"] > GATES["control_mae"]: result.append(f"{name}.mae")
        elif values["p95"] > GATES["control_p95"]: result.append(f"{name}.p95")
    if metrics["order"]["audio_regressions"] > GATES["audio_regressions"]: result.append("audio-regression")
    if metrics["input_mutations"] > GATES["input_mutations"]: result.append("input-mutation")
    return result


def evaluate(args) -> dict:
    if args.output.exists():
        raise FileExistsError(f"chain report already exists: {args.output}")
    runtime = ChainRuntime(
        args.family, args.manifest, args.gate, args.order, args.bundle, args.evidence
    )
    rows = []
    for domain, dataset in make_domains(args.corpus, args.source, args.samples, -30.0).items():
        for index in range(len(dataset)):
            item = dataset[index]
            dry, wet = item["dry"].numpy(), item["wet"].numpy()
            before = _digest(dry, wet)
            truth = _topology(item)
            report = runtime.infer(dry, wet)
            accepted = report["decision"] != "abstain"
            order = report["order"] or {}
            safe = order.get("order2", {})
            normalized = np.asarray(order.get("normalized_controls", np.zeros(9)))
            target = item["controls"].numpy()
            control_mask = item["control_mask"].numpy() >= 0.5
            rows.append({
                "domain": domain,
                "index": index,
                "accepted": accepted,
                "family_exact": accepted and set(report["active"]) == set(truth),
                "truth": list(truth),
                "predicted": report["active"],
                "confidence": report["gate"]["confidence"],
                "order_truth": item["order"].numpy().tolist(),
                "order_mask": item["order_mask"].numpy().tolist(),
                "proposed": safe.get("proposed", list(truth)),
                "selected": safe.get("selected", list(truth)),
                "original_error": safe.get("original_error", 0.0),
                "selected_error": safe.get("selected_error", 0.0),
                "controls": {
                    name: float(abs(normalized[position] - target[position]))
                    for position, name in enumerate(CONTROL_NAMES)
                    if accepted and control_mask[position]
                },
                "inputs_unchanged": before == _digest(dry, wet),
            })
            if (index + 1) % 8 == 0 or index + 1 == len(dataset):
                print(json.dumps({"domain": domain, "done": index + 1, "total": len(dataset)}), flush=True)
    runtime.assert_artifacts_unchanged()
    metrics = summarize(rows)
    rejected = failures(metrics)
    report = {
        "schema": 1,
        "status": "accepted-development-replay" if not rejected else "rejected",
        "accepted": not rejected,
        "source": args.source,
        "samples": args.samples,
        "selection_role": "development-replay; not independent sealed evidence",
        "gates": GATES,
        "metrics": metrics,
        "domains": {
            name: summarize([row for row in rows if row["domain"] == name])
            for name in sorted({row["domain"] for row in rows})
        },
        "failures": rejected,
        "rows": rows,
        "artifacts_unchanged": True,
        "source_audio_modified": False,
        "physical_audio_devices_used": False,
        "automatic_normalization": False,
        "automatic_limiting": False,
        "lossy_reencoding": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("calibrate", "valid"), required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = evaluate(args)
    print(json.dumps({"accepted": report["accepted"], "failures": report["failures"], "metrics": report["metrics"]}))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
