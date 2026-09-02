#!/usr/bin/env python3
"""Train the frozen-Drive residual adapter from an offline Capture Pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile
import torch
from torch.utils.data import DataLoader

from .capture_manifest import sha256, validate_manifest
from .capture_pack import audit_capture_manifest
from .drive_adapter import DriveDeviceAdapter
from .forward_chain import DRIVE_CHECKPOINT, _load_drive
from .forward_drive import FORWARD_RATE, forward_loss, normalized_drive_controls
from .spec import Drive
from .train import device
from .train_drive_adapter import _evaluate, _stream_parity


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/drive-capture-adapter"


def _drive(record: dict) -> Drive:
    effects = record.get("chain", {}).get("effects", [])
    values = [effect for effect in effects if effect.get("kind") == "drive"]
    if len(values) != 1 or len(effects) != 1:
        raise ValueError(f"capture {record.get('id')} must contain exactly one Drive")
    fields = dict(values[0])
    fields.pop("kind")
    return Drive(**fields)


def _windows(clean: np.ndarray, wet: np.ndarray, frames: int, count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if frames <= 0 or count <= 0:
        raise ValueError("training window geometry must be positive")
    if clean.shape != wet.shape or clean.ndim != 1:
        raise ValueError("Drive adapter Capture Pack currently requires aligned mono audio")
    if len(clean) < frames:
        padding = frames - len(clean)
        return [
            (
                np.pad(clean, (0, padding)).astype(np.float32, copy=False),
                np.pad(wet, (0, padding)).astype(np.float32, copy=False),
            )
        ]
    maximum = len(clean) - frames
    offsets = np.linspace(0, maximum, num=count, dtype=np.int64)
    return [
        (
            clean[offset : offset + frames].copy(),
            wet[offset : offset + frames].copy(),
        )
        for offset in offsets
    ]


@torch.inference_mode()
def capture_examples(
    manifest: Path,
    base: torch.nn.Module,
    split: str,
    *,
    frames: int,
    windows_per_record: int,
    allow_locked_final: bool = False,
) -> list[dict]:
    """Load only one declared development split and compute frozen-base output."""

    allowed = {"train", "calibrate", "valid"}
    if allow_locked_final:
        allowed.add("locked-final")
    if split not in allowed:
        raise ValueError(f"adapter training cannot open split {split!r}")
    document = json.loads(manifest.read_text())
    rows = []
    for record in document.get("records", []):
        if record.get("split") != split:
            continue
        clean_path = manifest.parent / record["clean"]
        wet_path = manifest.parent / record["wet"]
        clean, clean_rate = soundfile.read(clean_path, dtype="float32", always_2d=True)
        wet, wet_rate = soundfile.read(wet_path, dtype="float32", always_2d=True)
        if clean_rate != FORWARD_RATE or wet_rate != FORWARD_RATE:
            raise ValueError(f"capture {record['id']} is not {FORWARD_RATE} Hz")
        if clean.shape[1] != 1 or wet.shape[1] != 1:
            raise ValueError(f"capture {record['id']} is not mono")
        if not np.isfinite(clean).all() or not np.isfinite(wet).all():
            raise ValueError(f"capture {record['id']} contains non-finite audio")
        effect = _drive(record)
        controls = torch.from_numpy(normalized_drive_controls(effect))
        for clean_window, wet_window in _windows(
            clean[:, 0], wet[:, 0], frames, windows_per_record
        ):
            dry = torch.from_numpy(clean_window).unsqueeze(0)
            generic, _ = base(dry, controls.unsqueeze(0))
            rows.append(
                {
                    "dry": dry.squeeze(0),
                    "base": generic.squeeze(0),
                    "wet": torch.from_numpy(wet_window),
                    "controls": controls,
                    "context": split,
                    "record_id": str(record["id"]),
                }
            )
    if not rows:
        raise ValueError(f"Capture Pack has no {split} examples")
    return rows


def train_capture_adapter(
    manifest: Path,
    base_path: Path,
    output: Path,
    *,
    epochs: int = 8,
    batch_size: int = 8,
    frames: int = 8_192,
    train_windows: int = 2,
    evaluation_windows: int = 1,
    learning_rate: float = 2.0e-4,
) -> dict:
    """Train on train, select on calibrate, and report valid exactly once."""

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if (output / "drive-capture-adapter.pt").exists() or (output / "metrics.json").exists():
        raise ValueError(f"Capture Pack training output already exists: {output}")
    admission = audit_capture_manifest(manifest, development=True)
    if not admission["passed"]:
        raise ValueError(f"Capture Pack failed admission: {admission['issues']}")
    torch.manual_seed(20270830)
    np.random.seed(20270830)
    target = device()
    base = _load_drive(base_path).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    train_rows = capture_examples(
        manifest,
        base,
        "train",
        frames=frames,
        windows_per_record=train_windows,
    )
    calibrate_rows = capture_examples(
        manifest,
        base,
        "calibrate",
        frames=frames,
        windows_per_record=evaluation_windows,
    )
    valid_rows = capture_examples(
        manifest,
        base,
        "valid",
        frames=frames,
        windows_per_record=evaluation_windows,
    )
    train_loader = DataLoader(train_rows, batch_size=batch_size, shuffle=True)
    calibrate_loader = DataLoader(calibrate_rows, batch_size=batch_size)
    valid_loader = DataLoader(valid_rows, batch_size=batch_size)
    adapter = DriveDeviceAdapter().to(target)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=1.0e-6)
    best_score = float("inf")
    best_state = None
    history = []
    for epoch in range(epochs):
        adapter.train()
        losses = []
        for batch in train_loader:
            dry = batch["dry"].to(target)
            generic = batch["base"].to(target)
            wet = batch["wet"].to(target)
            controls = batch["controls"].to(target)
            prediction, _ = adapter(dry, generic, controls)
            loss, _ = forward_loss(prediction, wet)
            peak_ratio = prediction.abs().amax(dim=1) / wet.abs().amax(dim=1).clamp_min(1.0e-6)
            loss = loss + 4.0 * torch.relu(peak_ratio - 1.10).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        calibration = _evaluate(adapter, calibrate_loader, target)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "calibrate_model_esr": calibration["mean_model_esr"],
            "calibrate_improvement": calibration["mean_relative_improvement"],
        }
        history.append(row)
        if calibration["mean_model_esr"] < best_score:
            best_score = calibration["mean_model_esr"]
            best_state = {
                name: value.detach().cpu().clone() for name, value in adapter.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Capture Pack adapter training produced no checkpoint")
    adapter.load_state_dict(best_state)
    validation = _evaluate(adapter, valid_loader, target)
    runtime = _stream_parity(adapter.cpu().eval())
    post_training = validate_manifest(
        manifest,
        excluded_splits=frozenset({"locked-final"}),
    )
    promotable = bool(
        validation["mean_relative_improvement"] >= 0.20
        and validation["worst_context_improvement"] >= 0.0
        and validation["peak_ratio_p95"] <= 1.35
        and validation["worst_peak_ratio"] <= 1.75
        and runtime["max_absolute_error"] <= 2.0e-6
        and runtime["silence_max_absolute_output"] == 0.0
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "drive-capture-adapter.pt"
    base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
    torch.save(
        {
            "schema": 1,
            "sample_rate": FORWARD_RATE,
            "hidden_size": adapter.hidden_size,
            "state_dict": best_state,
            "base_checkpoint_sha256": base_hash,
            "capture_manifest_sha256": sha256(manifest),
            "capture_dataset_id": admission["dataset"]["id"],
        },
        checkpoint,
    )
    report = {
        "schema": 1,
        "status": "development-candidate" if promotable else "development-rejected",
        "promotable_to_locked_final": promotable,
        "scope": (
            "synthetic pipeline proof; no hardware fidelity claim"
            if admission["dataset"]["source_kind"] == "synthetic-smoke"
            else "real Capture Pack development validation; locked-final remains unopened"
        ),
        "dataset": admission["dataset"],
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "base_checkpoint_sha256": base_hash,
        "examples": {
            "train": len(train_rows),
            "calibrate": len(calibrate_rows),
            "valid": len(valid_rows),
        },
        "selection_split": "calibrate",
        "development_validation_split": "valid",
        "locked_final_opened": False,
        "validation": validation,
        "runtime": runtime,
        "history": history,
        "post_training_manifest_validation": post_training,
        "quality_policy": {
            "base_model_frozen": True,
            "source_files_read_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
        },
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=DRIVE_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames", type=int, default=8_192)
    parser.add_argument("--train-windows", type=int, default=2)
    args = parser.parse_args()
    report = train_capture_adapter(
        args.manifest,
        args.base,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        frames=args.frames,
        train_windows=args.train_windows,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "promotable_to_locked_final": report["promotable_to_locked_final"],
                "validation": report["validation"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
