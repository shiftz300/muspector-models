#!/usr/bin/env python3
"""End-to-end offline smoke proof for Capture Pack adapter training."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile
import torch

from .capture_kit import SAMPLE_RATE, drive_capture_plan
from .capture_manifest import sha256
from .drive_adapter import load_drive_adapter
from .evaluate_capture_adapter import evaluate_locked_final
from .forward_chain import DRIVE_CHECKPOINT, _load_drive
from .forward_drive import normalized_drive_controls
from .spec import Drive
from .train_capture_adapter import train_capture_adapter


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "remix/runs/data-readiness-phase1/capture-adapter-pipeline-smoke.json"


def _declared_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@torch.inference_mode()
def _pack(root: Path, *, include_locked_audio: bool = False) -> Path:
    plan = drive_capture_plan(device_id="synthetic-drive", player_id="synthetic-source")
    base = _load_drive(DRIVE_CHECKPOINT).eval()
    frames = 4_096
    time = np.arange(frames, dtype=np.float32) / SAMPLE_RATE
    split_clean: dict[str, tuple[Path, str]] = {}
    program_splits = ["train", "calibrate", "valid"]
    if include_locked_audio:
        program_splits.append("locked-final")
    for split_index, split in enumerate(program_splits):
        phase = 0.31 * split_index
        clean = (
            0.055 * np.sin(2.0 * np.pi * (83.0 + 17.0 * split_index) * time + phase)
            + 0.024 * np.sin(2.0 * np.pi * (211.0 + 31.0 * split_index) * time)
            + 0.012 * np.sin(2.0 * np.pi * (479.0 + 23.0 * split_index) * time)
        ).astype(np.float32)
        path = root / f"{split}-program.wav"
        soundfile.write(path, clean, SAMPLE_RATE, subtype="FLOAT")
        split_clean[split] = (path, sha256(path))

    records = []
    for record in plan["records"]:
        split = record["split"]
        identifier = record["id"]
        effect_fields = dict(record["chain"]["effects"][0])
        effect_fields.pop("kind")
        effect = Drive(**effect_fields)
        if split == "locked-final" and not include_locked_audio:
            clean_hash = _declared_hash("locked-final-program")
            wet_hash = _declared_hash(identifier)
            clean_name = "locked-final-program-unopened.wav"
            wet_name = f"{identifier}-unopened.wav"
        else:
            clean_path, clean_hash = split_clean[split]
            clean, _ = soundfile.read(clean_path, dtype="float32")
            controls = torch.from_numpy(normalized_drive_controls(effect)).unsqueeze(0)
            generic, _ = base(torch.from_numpy(clean).unsqueeze(0), controls)
            gain = 0.35 + 0.65 * float(controls[0, 0])
            tone = 0.25 + 0.75 * float(controls[0, 1])
            residual = np.tanh(clean * (2.0 + 2.5 * gain)) * (0.018 * tone)
            wet = (generic.squeeze(0).numpy() + residual).astype(np.float32)
            wet_name = f"{identifier}-wet.wav"
            wet_path = root / wet_name
            soundfile.write(wet_path, wet, SAMPLE_RATE, subtype="FLOAT")
            wet_hash = sha256(wet_path)
            clean_name = clean_path.name
        records.append(
            {
                "id": identifier,
                "split": split,
                "device_id": plan["device_id"],
                "session_id": record["session_id"],
                "player_id": plan["player_id"],
                "source_program_id": record["source_program_id"],
                "clean": clean_name,
                "wet": wet_name,
                "clean_sha256": clean_hash,
                "wet_sha256": wet_hash,
                "raw_program": clean_name,
                "raw_program_sha256": clean_hash,
                "raw_latency_capture": clean_name,
                "raw_latency_capture_sha256": clean_hash,
                "raw_wet_capture": wet_name,
                "raw_wet_capture_sha256": wet_hash,
                "measured_latency_samples": 0,
                "latency_compensated": True,
                "capture_quality": {
                    "passed": True,
                    "sample_rate": SAMPLE_RATE,
                    "frames": frames,
                    "channels": 1,
                    "measured_latency_samples": 0,
                    "clipped_samples": 0,
                    "dropout_blocks": 0,
                    "automatic_normalization": False,
                },
                "chain": record["chain"],
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "dataset": {
                    "id": "synthetic-capture-pipeline-smoke",
                    "source_kind": "synthetic-smoke",
                    "rights": "synthetic-smoke",
                    "intended_use": "drive-device-adapter-training",
                    "audio_quality_policy": "lossless-immutable-raw-v1",
                },
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return manifest


def smoke_report() -> dict:
    with tempfile.TemporaryDirectory(prefix="muspector-capture-adapter-smoke-") as temporary:
        root = Path(temporary)
        manifest = _pack(root, include_locked_audio=True)
        output = root / "run"
        metrics = train_capture_adapter(
            manifest,
            DRIVE_CHECKPOINT,
            output,
            epochs=2,
            batch_size=16,
            frames=2_048,
            train_windows=2,
            evaluation_windows=1,
            learning_rate=8.0e-4,
        )
        adapter = load_drive_adapter(Path(metrics["checkpoint"]))
        final = evaluate_locked_final(
            manifest,
            Path(metrics["checkpoint"]),
            DRIVE_CHECKPOINT,
            root / "locked-final.json",
            frames=2_048,
            windows_per_record=1,
            batch_size=16,
            open_locked_final=True,
        )
        try:
            evaluate_locked_final(
                manifest,
                Path(metrics["checkpoint"]),
                DRIVE_CHECKPOINT,
                root / "locked-final.json",
                frames=2_048,
                windows_per_record=1,
                batch_size=16,
                open_locked_final=True,
            )
            overwrite_refused = False
        except ValueError:
            overwrite_refused = True
        gates = {
            "full_192_record_contract_admitted": metrics["post_training_manifest_validation"][
                "records"
            ]
            == 192,
            "train_calibrate_valid_loaded": metrics["examples"]
            == {"train": 288, "calibrate": 16, "valid": 16},
            "locked_final_unopened_during_training": not metrics["locked_final_opened"],
            "base_model_frozen": metrics["quality_policy"]["base_model_frozen"],
            "source_files_read_only": metrics["quality_policy"]["source_files_read_only"],
            "checkpoint_reloadable": adapter.hidden_size == 12,
            "streaming_parity": metrics["runtime"]["max_absolute_error"] <= 2.0e-6,
            "silence_bit_exact": metrics["runtime"]["silence_max_absolute_output"] == 0.0,
            "no_physical_audio_devices": not metrics["quality_policy"][
                "physical_audio_devices_used"
            ],
            "one_shot_locked_final_pipeline": final["locked_final_examples"] == 16,
            "one_shot_report_non_overwriting": overwrite_refused,
            "synthetic_locked_final_accepted": final["accepted"],
        }
        report = {
            "schema": 1,
            "pipeline_passed": all(gates.values()),
            "scope": "synthetic pipeline proof only; not a real hardware model",
            "gates": gates,
            "examples": metrics["examples"],
            "selection_split": metrics["selection_split"],
            "development_validation_split": metrics["development_validation_split"],
            "valid_relative_improvement": metrics["validation"][
                "mean_relative_improvement"
            ],
            "valid_peak_ratio_p95": metrics["validation"]["peak_ratio_p95"],
            "synthetic_locked_final_accepted": final["accepted"],
            "synthetic_locked_final_relative_improvement": final["metrics"][
                "mean_relative_improvement"
            ],
            "runtime": metrics["runtime"],
            "temporary_audio_and_checkpoint_removed": True,
            "physical_audio_devices_used": False,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    report = smoke_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pipeline_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
