#!/usr/bin/env python3
"""One-shot sealed order and knob evaluation on the locked Tele split.

This module reads paired files and in-memory renders only. It never enumerates,
opens, records from, or writes to an audio device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate_order_search import make_domains
from .order_transfer_runtime import TransferRemixerRuntime, file_hash
from .spec import CONTROL_NAMES, KINDS, order_targets


GATES = {
    "order": {
        "real": {"exact": 0.72, "pairwise": 0.80},
        "reference": {"exact": 0.70, "pairwise": 0.78},
        "alternate": {"exact": 0.70, "pairwise": 0.78},
        "stress": {"exact": 0.70, "pairwise": 0.78},
        "pedalboard": {"exact": 0.65, "pairwise": 0.75},
    },
    "knob": {
        "macro_mae": 0.13,
        "macro_p95": 0.35,
        "control_mae": 0.28,
        "control_p95": 0.65,
    },
}


def _digest(*arrays: np.ndarray) -> str:
    value = hashlib.sha256()
    for array in arrays:
        value.update(np.asarray(array).tobytes())
    return value.hexdigest()


def _percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(values, quantile)) if values else 0.0


def summarize(rows: list[dict]) -> dict:
    correct = relations = exact = eligible = 0
    baseline_error = selected_error = 0.0
    controls = {name: [] for name in CONTROL_NAMES}
    unchanged = True
    for row in rows:
        truth = np.asarray(row["truth"], dtype=bool)
        prediction = np.asarray(row["prediction"], dtype=bool)
        mask = np.asarray(row["mask"], dtype=bool)
        if mask.any():
            matches = prediction == truth
            correct += int((matches & mask).sum())
            relations += int(mask.sum())
            exact += int((matches | ~mask).all())
            eligible += 1
            baseline_error += float(row["baseline_error"])
            selected_error += float(row["selected_error"])
        for name, error in row["controls"].items():
            controls[name].append(float(error))
        unchanged = unchanged and bool(row["inputs_unchanged"])
    active = [value for values in controls.values() for value in values]
    return {
        "examples": len(rows),
        "eligible": eligible,
        "relations": relations,
        "exact": exact / max(eligible, 1),
        "pairwise": correct / max(relations, 1),
        "baseline_error": baseline_error / max(eligible, 1),
        "selected_error": selected_error / max(eligible, 1),
        "macro_mae": float(np.mean(active)) if active else 0.0,
        "macro_p95": _percentile(active, 0.95),
        "controls": {
            name: {
                "values": len(values),
                "mae": float(np.mean(values)) if values else 0.0,
                "p95": _percentile(values, 0.95),
            }
            for name, values in controls.items()
        },
        "inputs_unchanged": unchanged,
    }


def accepted(domains: dict[str, dict]) -> tuple[bool, list[str], dict]:
    order_failures = []
    for name, gate in GATES["order"].items():
        metrics = domains[name]
        for metric, minimum in gate.items():
            if metrics[metric] < minimum:
                order_failures.append(f"{name}.{metric}<{minimum}")
        if metrics["selected_error"] > metrics["baseline_error"] + 1.0e-9:
            order_failures.append(f"{name}.audio-regression")
        if not metrics["inputs_unchanged"]:
            order_failures.append(f"{name}.input-mutation")
    synthetic = [domains[name] for name in ("reference", "alternate", "stress", "pedalboard")]
    control_values = {
        control: [
            value
            for domain in synthetic
            for value in domain["controls"][control].get("raw", [])
        ]
        for control in CONTROL_NAMES
    }
    flattened = [value for values in control_values.values() for value in values]
    macro_mae = float(np.mean(flattened)) if flattened else 0.0
    macro_p95 = _percentile(flattened, 0.95)
    knob_failures = []
    if macro_mae > GATES["knob"]["macro_mae"]:
        knob_failures.append("knob.macro_mae")
    if macro_p95 > GATES["knob"]["macro_p95"]:
        knob_failures.append("knob.macro_p95")
    for control, values in control_values.items():
        if not values:
            knob_failures.append(f"{control}.missing")
        elif float(np.mean(values)) > GATES["knob"]["control_mae"]:
            knob_failures.append(f"{control}.mae")
        elif _percentile(values, 0.95) > GATES["knob"]["control_p95"]:
            knob_failures.append(f"{control}.p95")
    failures = order_failures + knob_failures
    capabilities = {
        "order": {"accepted": not order_failures, "failures": order_failures},
        "knob": {"accepted": not knob_failures, "failures": knob_failures},
    }
    return not failures, failures, capabilities


def evaluate(corpus: Path, run: Path, output: Path, samples: int) -> dict:
    if output.exists():
        raise FileExistsError(f"sealed report already exists: {output}")
    tracked = {
        "bundle": Path("remix/runs/order-control-paired-public/remixer-bundle.pt"),
        "heads": run / "order-heads.joblib",
        "card": run / "model-card.json",
        "calibration": run / "blend-calibration.json",
        "development": run / "blend-development.json",
    }
    before = {name: file_hash(path) for name, path in tracked.items()}
    runtime = TransferRemixerRuntime(run)
    results = {}
    for domain, dataset in make_domains(corpus, "test", samples, -30.0).items():
        rows = []
        for index in range(len(dataset)):
            item = dataset[index]
            dry = item["dry"].numpy()
            wet = item["wet"].numpy()
            checksum = _digest(dry, wet)
            topology = tuple(
                KINDS[int(value)] for value in item["topology"].tolist() if int(value) >= 0
            )
            report = runtime.infer(dry, wet, tuple(sorted(topology)))
            order = report.get("order", {})
            selected = tuple(order.get("selected", topology))
            prediction = order_targets(selected)[0]
            ranked = {
                tuple(value["topology"]): float(value["reconstruction_error"])
                for value in order.get("ranked", [])
            }
            original = tuple(report.get("transfer", {}).get("original_selected", selected))
            estimate = np.asarray(report.get("normalized_controls", np.zeros(9)), dtype=np.float32)
            truth = item["controls"].numpy()
            control_mask = item["control_mask"].numpy() >= 0.5
            errors = {
                name: float(abs(estimate[position] - truth[position]))
                for position, name in enumerate(CONTROL_NAMES)
                if control_mask[position]
            }
            rows.append(
                {
                    "truth": item["order"].numpy().tolist(),
                    "prediction": prediction,
                    "mask": item["order_mask"].numpy().tolist(),
                    "baseline_error": ranked.get(original, 0.0),
                    "selected_error": ranked.get(selected, 0.0),
                    "controls": errors,
                    "inputs_unchanged": checksum == _digest(dry, wet),
                }
            )
            if (index + 1) % 32 == 0 or index + 1 == len(dataset):
                print(json.dumps({"domain": domain, "done": index + 1, "total": len(dataset)}), flush=True)
        results[domain] = summarize(rows)
        for name in CONTROL_NAMES:
            results[domain]["controls"][name]["raw"] = [
                float(row["controls"][name]) for row in rows if name in row["controls"]
            ]
    passed, failures, capabilities = accepted(results)
    for domain in results.values():
        for control in domain["controls"].values():
            control.pop("raw", None)
    after = {name: file_hash(path) for name, path in tracked.items()}
    immutable = before == after
    passed = passed and immutable
    if not immutable:
        failures.append("artifact-mutation")
    report = {
        "schema": 1,
        "status": "accepted-sealed-development" if passed else "rejected",
        "accepted": passed,
        "split": "telecaster-locked-test",
        "samples": samples,
        "gates": GATES,
        "domains": results,
        "failures": failures,
        "capabilities": capabilities,
        "artifacts": before,
        "artifacts_unchanged": immutable,
        "source_audio_modified": False,
        "physical_audio_devices_used": False,
        "automatic_normalization": False,
        "automatic_limiting": False,
        "lossy_reencoding": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=160)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = evaluate(args.corpus, args.run, args.output, args.samples)
    print(json.dumps({"accepted": report["accepted"], "failures": report["failures"]}))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
