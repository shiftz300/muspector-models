#!/usr/bin/env python3
"""One-shot locked-final evaluation for a frozen Capture Pack Drive adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .capture_manifest import sha256
from .capture_pack import audit_capture_manifest
from .drive_adapter import load_drive_adapter
from .forward_chain import DRIVE_CHECKPOINT, _load_drive
from .train import device
from .train_capture_adapter import capture_examples
from .train_drive_adapter import _evaluate, _stream_parity


def evaluate_locked_final(
    manifest: Path,
    checkpoint: Path,
    base_path: Path,
    output: Path,
    *,
    frames: int = 8_192,
    windows_per_record: int = 2,
    batch_size: int = 8,
    open_locked_final: bool = False,
) -> dict:
    """Open locked audio only after an explicit flag, and never overwrite a report."""

    if not open_locked_final:
        raise ValueError("locked-final remains closed without --open-locked-final")
    if output.exists():
        raise ValueError(f"one-shot locked-final report already exists: {output}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
    manifest_hash = sha256(manifest)
    if payload.get("base_checkpoint_sha256") != base_hash:
        raise ValueError("adapter checkpoint was trained against another frozen base")
    if payload.get("capture_manifest_sha256") != manifest_hash:
        raise ValueError("adapter checkpoint was trained against another Capture Pack manifest")
    admission = audit_capture_manifest(manifest, development=False)
    if not admission["passed"]:
        raise ValueError(f"locked-final Capture Pack failed admission: {admission['issues']}")
    if payload.get("capture_dataset_id") != admission["dataset"]["id"]:
        raise ValueError("adapter checkpoint dataset identity differs from Capture Pack")
    base = _load_drive(base_path).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    rows = capture_examples(
        manifest,
        base,
        "locked-final",
        frames=frames,
        windows_per_record=windows_per_record,
        allow_locked_final=True,
    )
    adapter = load_drive_adapter(checkpoint)
    target = device()
    metrics = _evaluate(adapter.to(target), DataLoader(rows, batch_size=batch_size), target)
    runtime = _stream_parity(adapter.cpu().eval())
    gates = {
        "mean_relative_improvement_at_least_20_percent": metrics[
            "mean_relative_improvement"
        ]
        >= 0.20,
        "worst_group_does_not_regress": metrics["worst_context_improvement"] >= 0.0,
        "peak_ratio_p95_within_1_35": metrics["peak_ratio_p95"] <= 1.35,
        "worst_peak_ratio_within_1_75": metrics["worst_peak_ratio"] <= 1.75,
        "streaming_parity": runtime["max_absolute_error"] <= 2.0e-6,
        "silence_bit_exact": runtime["silence_max_absolute_output"] == 0.0,
        "checkpoint_manifest_hash_bound": payload["capture_manifest_sha256"]
        == manifest_hash,
    }
    accepted = all(gates.values())
    report = {
        "schema": 1,
        "status": "accepted-locked-final" if accepted else "rejected-locked-final",
        "accepted": accepted,
        "dataset": admission["dataset"],
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "base_checkpoint_sha256": base_hash,
        "locked_final_examples": len(rows),
        "locked_final_audio_opened": True,
        "model_or_threshold_updates_allowed_after_open": False,
        "metrics": metrics,
        "runtime": runtime,
        "gates": gates,
        "quality_policy": {
            "source_files_read_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=DRIVE_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8_192)
    parser.add_argument("--windows-per-record", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--open-locked-final", action="store_true")
    args = parser.parse_args()
    report = evaluate_locked_final(
        args.manifest,
        args.checkpoint,
        args.base,
        args.output,
        frames=args.frames,
        windows_per_record=args.windows_per_record,
        batch_size=args.batch_size,
        open_locked_final=args.open_locked_final,
    )
    print(json.dumps({"accepted": report["accepted"], "gates": report["gates"]}, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
