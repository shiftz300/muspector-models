#!/usr/bin/env python3
"""Import, compare, and retain one stable ASRNN effect checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .asrnn_effects import audit_effect, effect_spec
from .evaluate_asrnn_effect import evaluate
from .import_asrnn_effect import import_checkpoint


def evaluate_candidates(
    device: str,
    sources: list[Path],
    corpus: Path,
    output: Path,
) -> dict:
    if output.exists():
        raise ValueError(f"stable effect candidate output already exists: {output}")
    spec = effect_spec(device)
    output.mkdir(parents=True)
    candidates_dir = output / "candidates"
    candidates_dir.mkdir()
    rows = []
    for source in sorted(sources):
        candidate_path = candidates_dir / source.name
        imported = import_checkpoint(source, candidate_path, spec.key)
        metrics = evaluate(candidate_path, corpus, spec.key)
        rows.append(
            {
                "source_name": source.name,
                "source_sha256": imported["source_sha256"],
                "candidate_path": str(candidate_path),
                "candidate_sha256": imported["output_sha256"],
                "candidate_recurrent_infinity_norms": imported[
                    "candidate_recurrent_infinity_norms"
                ],
                "accepted": metrics["accepted"],
                "official_eval": metrics["official_eval"],
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            not row["accepted"],
            row["official_eval"]["mean_per_file_esr"],
            row["official_eval"]["global_esr"],
        ),
    )
    selected_candidate = Path(selected["candidate_path"])
    canonical = output / f"{spec.key}-stable.pt"
    selected_candidate.replace(canonical)
    for path in candidates_dir.glob("*.pt"):
        path.unlink()
    candidates_dir.rmdir()
    selected_metrics = evaluate(canonical, corpus, spec.key)
    report = {
        "schema": 1,
        "phase": "phase-4-multi-device-stable-forward-candidate-selection",
        "device": spec.key,
        "device_name": spec.display_name,
        "control_names": list(spec.control_names),
        "data_audit": audit_effect(corpus, spec.key),
        "candidates": rows,
        "selected_source_name": selected["source_name"],
        "selected_source_sha256": selected["source_sha256"],
        "canonical_checkpoint": str(canonical),
        "canonical_checkpoint_sha256": selected_metrics["checkpoint_sha256"],
        "accepted": selected_metrics["accepted"],
        "status": selected_metrics["status"],
        "official_eval": selected_metrics["official_eval"],
        "quality_policy": selected_metrics["quality_policy"],
        "ui_integration_allowed": False,
    }
    (output / "metrics.json").write_text(
        json.dumps(selected_metrics, indent=2, sort_keys=True) + "\n"
    )
    if not selected_metrics["accepted"]:
        canonical.unlink()
        report["canonical_checkpoint_retained"] = False
    else:
        report["canonical_checkpoint_retained"] = True
    for row in report["candidates"]:
        row["candidate_artifact_retained"] = bool(
            selected_metrics["accepted"]
            and row["source_name"] == selected["source_name"]
        )
    (output / "candidate-comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("dfz", "cs3"), required=True)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_candidates(
        args.device, args.sources, args.corpus.resolve(), args.output
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
