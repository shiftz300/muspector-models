"""Calibrate and validate abstention for the paired family runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluate_order_search import make_domains
from .family import FAMILIES, FamilyRuntime
from .spec import KINDS


MIN_COVERAGE = 0.35
MIN_ACCEPTED_EXACT = 0.98


def _active(item: dict) -> list[str]:
    return sorted(KINDS[int(value)] for value in item["topology"].tolist() if int(value) >= 0)


def metrics(rows: list[dict], margin: float) -> dict:
    accepted = [row for row in rows if row["margin"] >= margin]
    labels = {}
    for family in FAMILIES:
        tp = sum(family in row["truth"] and family in row["predicted"] for row in rows)
        fp = sum(family not in row["truth"] and family in row["predicted"] for row in rows)
        fn = sum(family in row["truth"] and family not in row["predicted"] for row in rows)
        labels[family] = {
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return {
        "examples": len(rows),
        "raw_exact": sum(row["exact"] for row in rows) / max(len(rows), 1),
        "coverage": len(accepted) / max(len(rows), 1),
        "accepted": len(accepted),
        "accepted_exact": sum(row["exact"] for row in accepted) / max(len(accepted), 1),
        "accepted_errors": sum(not row["exact"] for row in accepted),
        "labels": labels,
    }


def select_margin(rows: list[dict]) -> float:
    errors = [row["margin"] for row in rows if not row["exact"]]
    threshold = 0.0 if not errors else float(np.nextafter(max(errors), np.inf))
    return threshold


def evaluate(
    source: str,
    corpus: Path,
    family_root: Path,
    manifest: Path,
    output: Path,
    samples: int,
    margin: float | None,
) -> dict:
    if output.exists():
        raise FileExistsError(f"family report already exists: {output}")
    runtime = FamilyRuntime(family_root, manifest)
    datasets = make_domains(corpus, source, samples, -30.0)
    rows = []
    for domain, dataset in datasets.items():
        for index in range(len(dataset)):
            item = dataset[index]
            dry, wet = item["dry"].numpy(), item["wet"].numpy()
            digest = hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest()
            result = runtime.infer(dry, wet)
            truth = _active(item)
            predicted = sorted(result["active"])
            rows.append(
                {
                    "domain": domain,
                    "index": index,
                    "input_sha256": digest,
                    "truth": truth,
                    "predicted": predicted,
                    "exact": truth == predicted,
                    "margin": result["minimum_margin"],
                    "scores": result["scores"],
                    "thresholds": result["thresholds"],
                    "source_audio_modified": result["source_audio_modified"],
                }
            )
            if (index + 1) % 16 == 0 or index + 1 == len(dataset):
                print(json.dumps({"domain": domain, "done": index + 1, "total": len(dataset)}), flush=True)
    if margin is None:
        margin = select_margin(rows)
    domains = {
        domain: metrics([row for row in rows if row["domain"] == domain], margin)
        for domain in datasets
    }
    overall = metrics(rows, margin)
    unchanged = all(not row["source_audio_modified"] for row in rows)
    failures = []
    if overall["coverage"] < MIN_COVERAGE:
        failures.append("coverage")
    if overall["accepted_exact"] < MIN_ACCEPTED_EXACT:
        failures.append("accepted-exact")
    if not unchanged:
        failures.append("input-mutation")
    runtime.assert_artifacts_unchanged()
    report = {
        "schema": 1,
        "source": source,
        "samples": samples,
        "margin": margin,
        "minimum": {"coverage": MIN_COVERAGE, "accepted_exact": MIN_ACCEPTED_EXACT},
        "accepted": not failures,
        "failures": failures,
        "overall": overall,
        "domains": domains,
        "rows": rows,
        "family_manifest_sha256": runtime.manifest_sha256,
        "artifacts_unchanged": True,
        "source_audio_modified": not unchanged,
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
    parser.add_argument("--source", choices=("calibrate", "valid"), required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--margin", type=float)
    args = parser.parse_args()
    report = evaluate(
        args.source, args.corpus, args.family, args.manifest, args.output, args.samples, args.margin
    )
    print(json.dumps({"accepted": report["accepted"], "failures": report["failures"], "margin": report["margin"]}))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
