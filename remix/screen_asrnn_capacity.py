#!/usr/bin/env python3
"""Train-side-only capacity screening of independently imported ASRNN weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import torch

from .asrnn_effects import effect_files
from .import_asrnn_effect import import_checkpoint
from .stable_effect import load_stable_effect
from .train_asrnn_phase7 import _evaluate, _partition, _rows


def selection_key(metrics: dict) -> tuple:
    return (
        not metrics["passes_selection_gate"],
        metrics["absolute_peak_error_p95"],
        metrics["worst_attack_absolute_peak_error_p95"],
        metrics["global_esr"],
    )


def screen(args) -> dict:
    if args.output.exists():
        raise ValueError(f"capacity-screen output already exists: {args.output}")
    audit = json.loads(args.group_audit.read_text())
    _, calibration = _partition(effect_files(args.corpus, "cs3", "train"), audit)
    rows = _rows(calibration, "cs3")
    args.output.mkdir(parents=True)
    report = {
        "schema": 1,
        "phase": "phase-8-capacity-screen",
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "group_audit_sha256": hashlib.sha256(args.group_audit.read_bytes()).hexdigest(),
        "calibration_files": len(rows),
        "calibration_caveat": "official pretrained weights may have seen all official train files; this is selection data, not independent final evidence",
        "candidate_development_challenge_evaluated": False,
        "new_locked_final_audio_opened": False,
        "source_audio_modified": False,
        "physical_audio_devices_used": False,
        "screen_complete": False,
        "expected_candidates": 2 * len(args.layers),
        "results": [],
    }
    best_key = None
    with zipfile.ZipFile(args.archive) as archive:
        if archive.testzip() is not None:
            raise ValueError("ASRNN results archive failed CRC validation")
        members = sorted(
            name for name in archive.namelist()
            if "results/long/cs-3/checkpoints/StableLSTM_inf-" in name
            and name.endswith(f"-{args.seed}.pt")
            and any(f"StableLSTM_inf-{layers}-64-" in name for layers in args.layers)
        )
        if len(members) != 2 * len(args.layers):
            raise ValueError("expected GFB and MAE checkpoints for each selected architecture")
        for member in members:
            started = time.monotonic()
            print(json.dumps({"screening": member, "compute_device": "cpu"}), flush=True)
            with tempfile.TemporaryDirectory(prefix="muspector-capacity-") as temporary:
                directory = Path(temporary)
                source = Path(archive.extract(member, directory))
                converted = directory / "converted.pt"
                provenance = import_checkpoint(source, converted, "cs3")
                model, payload = load_stable_effect(converted)
                metrics = _evaluate(model, rows, torch.device("cpu"), args.batch_size)
                row = {
                    "member": member,
                    "source_sha256": provenance["source_sha256"],
                    "checkpoint_sha256": provenance["output_sha256"],
                    "layers": model.layers,
                    "hidden_size": model.hidden_size,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "elapsed_seconds": time.monotonic() - started,
                    "calibration": metrics,
                    "import_runtime": provenance,
                }
                report["results"].append(row)
                key = selection_key(metrics)
                if best_key is None or key < best_key:
                    best_key = key
                    shutil.copyfile(converted, args.output / "selected.pt")
                    report["selected"] = row
                (args.output / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)
    report["screen_complete"] = True
    (args.output / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/downloads/asrnn-results-20406285.zip"))
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/asrnn-physical-effects"))
    parser.add_argument("--group-audit", type=Path, default=Path("remix/runs/asrnn-cs3-phase7-group-audit.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--seed", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    screen(args)


if __name__ == "__main__":
    main()
