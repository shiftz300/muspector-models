#!/usr/bin/env python3
"""Deterministic, quality-gated capture preparation for real hardware.

The kit deliberately does not own live audio I/O. It creates a replay program,
measures session latency from a bypass/loopback recording, derives new aligned
Clean/Wet files without touching any raw input, and emits a manifest accepted
by :mod:`remix.capture_manifest`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile
from scipy.signal import chirp, correlate, windows

from .capture_manifest import SCHEMA as MANIFEST_SCHEMA
from .capture_manifest import SPLITS
from .capture_manifest import sha256, validate_manifest
from .quality import checked_audio
from .render import render_chain
from .spec import ChainSpec, Drive


SAMPLE_RATE = 48_000
PLAN_SCHEMA = 1
PROGRAM_SCHEMA = 1
CAPTURE_KIT_VERSION = 1
DEFAULT_PLAN_SEED = 0x4D555350
DEFAULT_SPLIT_COUNTS = {
    "train": 144,
    "calibrate": 16,
    "valid": 16,
    "locked-final": 16,
}
SETTINGS_PER_SESSION = 16
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class CaptureLayout:
    sample_rate: int
    probe_start_frame: int
    probe_frames: int
    payload_start_frame: int
    payload_frames: int
    total_frames: int
    maximum_latency_frames: int


@dataclass(frozen=True)
class QualityLimits:
    source_peak_max_dbfs: float = -6.0
    wet_peak_max_dbfs: float = -3.0
    minimum_source_rms_dbfs: float = -60.0
    minimum_wet_rms_dbfs: float = -60.0
    minimum_capture_snr_db: float = 45.0
    minimum_latency_confidence: float = 0.80
    clip_threshold: float = 0.9999
    dropout_block_frames: int = 480


LIMITS = QualityLimits()


def _checked_identifier(name: str, value: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{name} must use only letters, digits, dot, underscore, or hyphen: {value!r}"
        )
    return value


def _dbfs(value: float) -> float:
    if value <= 0.0:
        # Keep reports strict-JSON compatible while representing digital zero.
        return -400.0
    return 20.0 * math.log10(value)


def _rms(audio: np.ndarray) -> float:
    value = np.asarray(audio, dtype=np.float64)
    if not value.size:
        return 0.0
    return float(np.sqrt(np.mean(value * value)))


def _latin_hypercube(count: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0 or dimensions <= 0:
        raise ValueError("Latin-hypercube dimensions and count must be positive")
    values = np.empty((count, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        strata = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
        values[:, dimension] = strata[rng.permutation(count)]
    return values


def drive_capture_plan(
    *,
    device_id: str,
    player_id: str,
    seed: int = DEFAULT_PLAN_SEED,
    split_counts: dict[str, int] | None = None,
) -> dict:
    """Create session-disjoint, split-local LHS coverage for Drive controls."""

    _checked_identifier("device_id", device_id)
    _checked_identifier("player_id", player_id)
    counts = dict(DEFAULT_SPLIT_COUNTS if split_counts is None else split_counts)
    if set(counts) != set(DEFAULT_SPLIT_COUNTS):
        raise ValueError(f"split_counts must contain {tuple(DEFAULT_SPLIT_COUNTS)}")
    if any(count <= 0 for count in counts.values()):
        raise ValueError("every split must contain at least one setting")

    rng = np.random.default_rng(seed)
    records: list[dict] = []
    sessions: list[dict] = []
    global_index = 0
    for split, count in counts.items():
        normalized = _latin_hypercube(count, 3, rng)
        source_program_id = f"{player_id}-{split}-program"
        session_count = math.ceil(count / SETTINGS_PER_SESSION)
        for session_index in range(session_count):
            session_id = f"{split}-session-{session_index + 1:02d}"
            sessions.append(
                {
                    "id": session_id,
                    "split": split,
                    "drift_anchor": ChainSpec((Drive(15.0, 0.5, -3.0),)).document(),
                }
            )
        for local_index, controls in enumerate(normalized):
            session_index = local_index // SETTINGS_PER_SESSION
            session_id = f"{split}-session-{session_index + 1:02d}"
            gain, tone, level = (float(value) for value in controls)
            effect = Drive(
                gain_db=30.0 * gain,
                tone=tone,
                level_db=-18.0 + 30.0 * level,
            )
            global_index += 1
            records.append(
                {
                    "id": f"{device_id}-drive-{global_index:04d}",
                    "role": "training-pair",
                    "split": split,
                    "session_id": session_id,
                    "source_program_id": source_program_id,
                    "normalized_controls": {
                        "gain": gain,
                        "tone": tone,
                        "level": level,
                    },
                    "chain": ChainSpec((effect,)).document(),
                }
            )

    return {
        "schema": PLAN_SCHEMA,
        "capture_kit_version": CAPTURE_KIT_VERSION,
        "sample_rate": SAMPLE_RATE,
        "seed": seed,
        "device_id": device_id,
        "player_id": player_id,
        "training_records": len(records),
        "split_counts": counts,
        "settings_per_session": SETTINGS_PER_SESSION,
        "source_programs": {
            split: f"{player_id}-{split}-program" for split in counts
        },
        "sessions": sessions,
        "records": records,
        "quality_policy": {
            **asdict(LIMITS),
            "automatic_normalization": False,
            "raw_capture_mutation": "forbidden",
            "aligned_encoding": "FLOAT",
        },
    }


def drive_pilot_session(plan: dict, session_id: str | None = None) -> dict:
    """Create an ordered 16-setting take sheet with before/after drift anchors."""

    if plan.get("schema") != PLAN_SCHEMA or plan.get("sample_rate") != SAMPLE_RATE:
        raise ValueError("unsupported Drive capture plan")
    device_id = _checked_identifier("device_id", str(plan.get("device_id", "")))
    player_id = _checked_identifier("player_id", str(plan.get("player_id", "")))
    sessions = plan.get("sessions", [])
    if not sessions:
        raise ValueError("Drive capture plan has no sessions")
    selected = sessions[0] if session_id is None else next(
        (session for session in sessions if session.get("id") == session_id),
        None,
    )
    if selected is None:
        raise ValueError(f"unknown Drive capture session: {session_id!r}")
    selected_id = _checked_identifier("session_id", str(selected["id"]))
    records = [
        record for record in plan.get("records", []) if record.get("session_id") == selected_id
    ]
    if not records:
        raise ValueError(f"Drive capture session has no settings: {selected_id}")
    if len(records) > SETTINGS_PER_SESSION:
        raise ValueError(f"Drive capture session exceeds {SETTINGS_PER_SESSION} settings")
    anchor_chain = selected.get("drift_anchor")
    if not isinstance(anchor_chain, dict):
        raise ValueError(f"Drive capture session has no drift anchor: {selected_id}")

    def take(identifier: str, role: str, chain: dict, **fields: str) -> dict:
        _checked_identifier("take_id", identifier)
        return {
            "id": identifier,
            "role": role,
            "raw_wet": f"raw/{identifier}.wav",
            "chain": chain,
            **fields,
        }

    prefix = f"{device_id}-{selected_id}"
    takes = [
        take(f"{prefix}-anchor-before", "drift-anchor", anchor_chain, phase="before")
    ]
    takes.extend(
        take(
            str(record["id"]),
            "training-pair",
            record["chain"],
            split=str(record["split"]),
        )
        for record in records
    )
    takes.append(
        take(f"{prefix}-anchor-after", "drift-anchor", anchor_chain, phase="after")
    )
    return {
        "schema": 1,
        "capture_kit_version": CAPTURE_KIT_VERSION,
        "sample_rate": SAMPLE_RATE,
        "device_id": device_id,
        "player_id": player_id,
        "session_id": selected_id,
        "split": selected["split"],
        "source_program_id": records[0]["source_program_id"],
        "loopback_capture": f"raw/{prefix}-loopback.wav",
        "training_takes": len(records),
        "drift_anchor_takes": 2,
        "takes": takes,
        "instructions": [
            "Use only the replay program assigned to this split.",
            "Record the loopback once with the effect bypassed.",
            "Do not change interface input/output gain during the session.",
            "Record takes in order and do not normalize, limit, or trim raw WAV files.",
            "Repeat the center-control drift anchor after the final training take.",
        ],
    }


def plan_coverage(plan: dict) -> dict:
    """Summarize physical coverage and verify one LHS visit per stratum."""

    by_split: dict[str, list[dict]] = {}
    for record in plan["records"]:
        by_split.setdefault(record["split"], []).append(record)
    report = {}
    for split, records in by_split.items():
        count = len(records)
        axes = {}
        for name in ("gain", "tone", "level"):
            values = np.asarray(
                [record["normalized_controls"][name] for record in records],
                dtype=np.float64,
            )
            bins = np.floor(values * count).astype(np.int64)
            complete = np.array_equal(np.sort(bins), np.arange(count))
            axes[name] = {
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "strata_complete": bool(complete),
            }
        report[split] = {"records": count, "axes": axes}
    return report


def synchronization_probe(sample_rate: int = SAMPLE_RATE, frames: int | None = None) -> np.ndarray:
    """Return a low-level, windowed chirp intended for bypass latency measurement."""

    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"capture programs must use {SAMPLE_RATE} Hz")
    count = round(0.20 * sample_rate) if frames is None else int(frames)
    if count < 256:
        raise ValueError("synchronization probe is too short")
    time = np.arange(count, dtype=np.float64) / sample_rate
    signal = chirp(time, f0=180.0, f1=15_000.0, t1=time[-1], method="logarithmic")
    signal *= windows.tukey(count, alpha=0.15)
    signal *= 10.0 ** (-30.0 / 20.0)
    return signal.astype(np.float32)


def build_capture_program(source: np.ndarray, sample_rate: int) -> tuple[np.ndarray, CaptureLayout]:
    """Prepend a sync region and retain enough tail guard for latency alignment."""

    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"source must already be {SAMPLE_RATE} Hz, got {sample_rate}")
    payload = checked_audio(source, name="capture source")
    if payload.shape[0] < sample_rate:
        raise ValueError("capture source must contain at least one second")
    probe = synchronization_probe(sample_rate)
    probe_start = round(0.25 * sample_rate)
    payload_start = probe_start + len(probe) + round(0.25 * sample_rate)
    maximum_latency = round(0.20 * sample_rate)
    total = payload_start + payload.shape[0] + maximum_latency
    channels = 1 if payload.ndim == 1 else payload.shape[1]
    program_2d = np.zeros((total, channels), dtype=np.float32)
    program_2d[probe_start : probe_start + len(probe), :] = probe[:, None]
    program_2d[payload_start : payload_start + payload.shape[0], :] = (
        payload[:, None] if payload.ndim == 1 else payload
    )
    program = program_2d[:, 0] if payload.ndim == 1 else program_2d
    layout = CaptureLayout(
        sample_rate=sample_rate,
        probe_start_frame=probe_start,
        probe_frames=len(probe),
        payload_start_frame=payload_start,
        payload_frames=payload.shape[0],
        total_frames=total,
        maximum_latency_frames=maximum_latency,
    )
    return program, layout


def estimate_latency(
    program: np.ndarray,
    bypass_capture: np.ndarray,
    layout: CaptureLayout,
) -> tuple[int, float]:
    """Measure non-negative I/O latency from a bypass/loopback capture."""

    source = checked_audio(program, name="capture program")
    captured = checked_audio(bypass_capture, name="bypass capture")
    if source.shape[0] != layout.total_frames:
        raise ValueError("capture program does not match its layout")
    if captured.shape[0] < layout.probe_start_frame + layout.probe_frames:
        raise ValueError("bypass capture is shorter than the synchronization region")
    source_mono = source if source.ndim == 1 else source.mean(axis=1)
    captured_mono = captured if captured.ndim == 1 else captured.mean(axis=1)
    reference = source_mono[
        layout.probe_start_frame : layout.probe_start_frame + layout.probe_frames
    ].astype(np.float64)
    search_end = min(
        captured_mono.shape[0],
        layout.probe_start_frame
        + layout.probe_frames
        + layout.maximum_latency_frames,
    )
    search = captured_mono[layout.probe_start_frame:search_end].astype(np.float64)
    if len(search) < len(reference):
        raise ValueError("bypass capture has no complete probe search window")
    reference -= reference.mean()
    search -= search.mean()
    correlations = correlate(search, reference, mode="valid", method="fft")
    latency = int(np.argmax(np.abs(correlations)))
    candidate = search[latency : latency + len(reference)]
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(candidate))
    confidence = abs(float(correlations[latency])) / max(denominator, 1.0e-20)
    if latency > layout.maximum_latency_frames:
        raise ValueError(f"measured latency exceeds capture guard: {latency}")
    return latency, confidence


def align_payload(
    program: np.ndarray,
    wet_capture: np.ndarray,
    layout: CaptureLayout,
    latency_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create aligned copies; raw program and capture arrays remain untouched."""

    source = checked_audio(program, name="capture program")
    captured = checked_audio(wet_capture, name="wet raw capture")
    if not 0 <= latency_frames <= layout.maximum_latency_frames:
        raise ValueError(f"latency is outside the capture guard: {latency_frames}")
    if source.ndim != captured.ndim or (source.ndim == 2 and source.shape[1] != captured.shape[1]):
        raise ValueError(f"program/capture channels differ: {source.shape}/{captured.shape}")
    clean_start = layout.payload_start_frame
    wet_start = clean_start + latency_frames
    wet_end = wet_start + layout.payload_frames
    if wet_end > captured.shape[0]:
        raise ValueError("wet capture is too short to retain the complete aligned payload")
    clean = source[clean_start : clean_start + layout.payload_frames].copy()
    wet = captured[wet_start:wet_end].copy()
    return clean, wet


def _dropout_blocks(clean: np.ndarray, wet: np.ndarray, block_frames: int) -> int:
    clean_2d = clean[:, None] if clean.ndim == 1 else clean
    wet_2d = wet[:, None] if wet.ndim == 1 else wet
    dropouts = 0
    for start in range(0, len(clean_2d), block_frames):
        dry_block = clean_2d[start : start + block_frames]
        wet_block = wet_2d[start : start + block_frames]
        if _dbfs(_rms(dry_block)) > -50.0 and np.count_nonzero(wet_block) == 0:
            dropouts += 1
    return dropouts


def source_quality_report(source: np.ndarray, limits: QualityLimits = LIMITS) -> dict:
    """Preflight replay material before any hardware session is recorded."""

    value = checked_audio(source, name="capture source")
    peak_dbfs = _dbfs(float(np.max(np.abs(value))))
    rms_dbfs = _dbfs(_rms(value))
    failures = []
    if peak_dbfs > limits.source_peak_max_dbfs:
        failures.append("source_headroom")
    if rms_dbfs < limits.minimum_source_rms_dbfs:
        failures.append("source_level")
    return {
        "passed": not failures,
        "failures": failures,
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
        "frames": value.shape[0],
        "channels": 1 if value.ndim == 1 else value.shape[1],
        "automatic_normalization": False,
    }


def capture_quality_report(
    clean: np.ndarray,
    wet: np.ndarray,
    raw_wet: np.ndarray,
    layout: CaptureLayout,
    *,
    latency_frames: int,
    latency_confidence: float,
    limits: QualityLimits = LIMITS,
) -> dict:
    """Return explicit metrics and hard failures; never change gain to pass."""

    clean_value = checked_audio(clean, name="aligned clean")
    wet_value = checked_audio(wet, name="aligned wet")
    raw_value = checked_audio(raw_wet, name="raw wet")
    if clean_value.shape != wet_value.shape:
        raise ValueError(f"aligned pair geometry differs: {clean_value.shape}/{wet_value.shape}")

    clean_peak = float(np.max(np.abs(clean_value)))
    wet_peak = float(np.max(np.abs(wet_value)))
    clean_rms = _rms(clean_value)
    wet_rms = _rms(wet_value)
    noise_end = max(1, layout.probe_start_frame // 2)
    noise_rms = _rms(raw_value[:noise_end])
    snr = _dbfs(wet_rms / max(noise_rms, 1.0e-20))
    clipped = int(np.count_nonzero(np.abs(wet_value) >= limits.clip_threshold))
    dropout_blocks = _dropout_blocks(clean_value, wet_value, limits.dropout_block_frames)
    failures = []
    if _dbfs(clean_peak) > limits.source_peak_max_dbfs:
        failures.append("source_headroom")
    if _dbfs(wet_peak) > limits.wet_peak_max_dbfs:
        failures.append("wet_headroom")
    if _dbfs(clean_rms) < limits.minimum_source_rms_dbfs:
        failures.append("source_level")
    if _dbfs(wet_rms) < limits.minimum_wet_rms_dbfs:
        failures.append("wet_level")
    if snr < limits.minimum_capture_snr_db:
        failures.append("capture_snr")
    if latency_confidence < limits.minimum_latency_confidence:
        failures.append("latency_confidence")
    if clipped:
        failures.append("clipping")
    if dropout_blocks:
        failures.append("dropout")
    return {
        "schema": 1,
        "passed": not failures,
        "failures": failures,
        "sample_rate": layout.sample_rate,
        "frames": layout.payload_frames,
        "channels": 1 if clean_value.ndim == 1 else clean_value.shape[1],
        "measured_latency_samples": latency_frames,
        "latency_confidence": latency_confidence,
        "clean_peak_dbfs": _dbfs(clean_peak),
        "wet_peak_dbfs": _dbfs(wet_peak),
        "clean_rms_dbfs": _dbfs(clean_rms),
        "wet_rms_dbfs": _dbfs(wet_rms),
        "noise_rms_dbfs": _dbfs(noise_rms),
        "capture_snr_db": snr,
        "clipped_samples": clipped,
        "dropout_blocks": dropout_blocks,
        "limits": asdict(limits),
        "automatic_normalization": False,
    }


def drive_anchor_drift_report(
    program: np.ndarray,
    bypass_capture: np.ndarray,
    anchor_before: np.ndarray,
    anchor_after: np.ndarray,
    layout: CaptureLayout,
    *,
    maximum_level_drift_db: float = 0.5,
    maximum_gain_aligned_nrmse: float = 0.03,
) -> dict:
    """Reject a session whose repeated center setting changed materially."""

    latency, confidence = estimate_latency(program, bypass_capture, layout)
    clean, before = align_payload(program, anchor_before, layout, latency)
    _, after = align_payload(program, anchor_after, layout, latency)
    before_quality = capture_quality_report(
        clean,
        before,
        anchor_before,
        layout,
        latency_frames=latency,
        latency_confidence=confidence,
    )
    after_quality = capture_quality_report(
        clean,
        after,
        anchor_after,
        layout,
        latency_frames=latency,
        latency_confidence=confidence,
    )
    before_flat = before.astype(np.float64).reshape(-1)
    after_flat = after.astype(np.float64).reshape(-1)
    level_drift = abs(_dbfs(_rms(after_flat)) - _dbfs(_rms(before_flat)))
    denominator = float(np.dot(after_flat, after_flat))
    gain = float(np.dot(before_flat, after_flat)) / max(denominator, 1.0e-20)
    residual = before_flat - gain * after_flat
    gain_aligned_nrmse = _rms(residual) / max(_rms(before_flat), 1.0e-20)
    failures = []
    if not before_quality["passed"]:
        failures.append("anchor_before_quality")
    if not after_quality["passed"]:
        failures.append("anchor_after_quality")
    if level_drift > maximum_level_drift_db:
        failures.append("anchor_level_drift")
    if gain_aligned_nrmse > maximum_gain_aligned_nrmse:
        failures.append("anchor_response_drift")
    return {
        "schema": 1,
        "passed": not failures,
        "failures": failures,
        "latency_samples": latency,
        "latency_confidence": confidence,
        "level_drift_db": level_drift,
        "gain_aligned_nrmse": gain_aligned_nrmse,
        "maximum_level_drift_db": maximum_level_drift_db,
        "maximum_gain_aligned_nrmse": maximum_gain_aligned_nrmse,
        "anchor_before_quality": before_quality,
        "anchor_after_quality": after_quality,
    }


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = soundfile.read(path, dtype="float32", always_2d=False)
    return checked_audio(audio, name=str(path)), int(sample_rate)


def write_capture_program(source_path: Path, output_path: Path, metadata_path: Path) -> dict:
    if len({source_path.resolve(), output_path.resolve(), metadata_path.resolve()}) != 3:
        raise ValueError("source, replay program, and metadata paths must be distinct")
    source_hash = sha256(source_path)
    source, sample_rate = _read_audio(source_path)
    source_quality = source_quality_report(source)
    if not source_quality["passed"]:
        raise ValueError(f"capture source failed quality gates: {source_quality['failures']}")
    program, layout = build_capture_program(source, sample_rate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(output_path, program, sample_rate, subtype="FLOAT")
    if sha256(source_path) != source_hash:
        raise RuntimeError("source changed while creating the capture program")
    document = {
        "schema": PROGRAM_SCHEMA,
        "capture_kit_version": CAPTURE_KIT_VERSION,
        "source": str(source_path),
        "source_sha256": source_hash,
        "program": str(output_path),
        "program_sha256": sha256(output_path),
        "layout": asdict(layout),
        "source_quality": source_quality,
        "automatic_normalization": False,
    }
    metadata_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as error:
        raise ValueError(f"capture outputs must be contained beside the manifest: {path}") from error


def prepare_capture_record(
    *,
    program_path: Path,
    program_metadata_path: Path,
    latency_capture_path: Path,
    wet_raw_path: Path,
    clean_output_path: Path,
    wet_output_path: Path,
    manifest_path: Path,
    capture_id: str,
    split: str,
    device_id: str,
    session_id: str,
    player_id: str,
    drive: Drive,
    source_program_id: str | None = None,
    rights: str,
    source_kind: str = "hardware-capture",
    dataset_id: str | None = None,
    append: bool = False,
) -> dict:
    """Align one Drive capture, enforce gates, and write an admitted manifest."""

    if split not in SPLITS:
        raise ValueError(f"unsupported capture split: {split!r}")
    for name, value in (
        ("capture_id", capture_id),
        ("device_id", device_id),
        ("session_id", session_id),
        ("player_id", player_id),
    ):
        _checked_identifier(name, value)
    drive.validate()
    source_program_id = source_program_id or f"{player_id}-{split}-program"
    _checked_identifier("source_program_id", source_program_id)
    if rights not in {"owned", "authorized-for-device-adapter-training", "synthetic-smoke"}:
        raise ValueError("capture rights must explicitly authorize adapter training")
    if source_kind not in {"hardware-capture", "synthetic-smoke"}:
        raise ValueError(f"unsupported capture source kind: {source_kind!r}")
    dataset_id = dataset_id or f"{device_id}-capture-pack"
    _checked_identifier("dataset_id", dataset_id)
    raw_paths = (program_path, latency_capture_path, wet_raw_path)
    derived_paths = (clean_output_path, wet_output_path, manifest_path)
    resolved_raw = {path.resolve() for path in (*raw_paths, program_metadata_path)}
    resolved_derived = {path.resolve() for path in derived_paths}
    if len(resolved_derived) != len(derived_paths) or resolved_raw & resolved_derived:
        raise ValueError("aligned outputs and manifest must be distinct from every raw input")
    manifest_root = manifest_path.parent
    for path in (*raw_paths, clean_output_path, wet_output_path):
        _relative(manifest_root, path)
    if clean_output_path.exists() or wet_output_path.exists():
        raise ValueError("aligned output paths must not already exist")
    if manifest_path.exists() and not append:
        raise ValueError("capture manifest already exists; use --append for another record")
    existing_manifest = None
    if append and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text())
        if existing_manifest.get("schema") != MANIFEST_SCHEMA:
            raise ValueError("cannot append to an incompatible capture manifest")
        expected_dataset = {
            "id": dataset_id,
            "source_kind": source_kind,
            "rights": rights,
            "intended_use": "drive-device-adapter-training",
            "audio_quality_policy": "lossless-immutable-raw-v1",
        }
        if existing_manifest.get("dataset") != expected_dataset:
            raise ValueError("cannot append with different Capture Pack provenance")
        if any(record.get("id") == capture_id for record in existing_manifest.get("records", [])):
            raise ValueError(f"capture id already exists in manifest: {capture_id}")
    before_hashes = {path: sha256(path) for path in raw_paths}
    metadata = json.loads(program_metadata_path.read_text())
    if metadata.get("schema") != PROGRAM_SCHEMA:
        raise ValueError("unsupported capture-program metadata")
    if metadata.get("program_sha256") != before_hashes[program_path]:
        raise ValueError("capture program differs from its metadata hash")
    layout = CaptureLayout(**metadata["layout"])
    program, program_rate = _read_audio(program_path)
    latency_capture, latency_rate = _read_audio(latency_capture_path)
    wet_raw, wet_rate = _read_audio(wet_raw_path)
    if {program_rate, latency_rate, wet_rate} != {SAMPLE_RATE}:
        raise ValueError("program, loopback, and wet capture must all use 48 kHz")
    latency, confidence = estimate_latency(program, latency_capture, layout)
    clean, wet = align_payload(program, wet_raw, layout, latency)
    quality = capture_quality_report(
        clean,
        wet,
        wet_raw,
        layout,
        latency_frames=latency,
        latency_confidence=confidence,
    )
    if not quality["passed"]:
        raise ValueError(f"capture failed quality gates: {quality['failures']}")

    pending = manifest_path.with_name(f".{manifest_path.name}.pending")
    try:
        clean_output_path.parent.mkdir(parents=True, exist_ok=True)
        wet_output_path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(clean_output_path, clean, SAMPLE_RATE, subtype="FLOAT")
        soundfile.write(wet_output_path, wet, SAMPLE_RATE, subtype="FLOAT")
        after_hashes = {path: sha256(path) for path in raw_paths}
        if after_hashes != before_hashes:
            raise RuntimeError("a raw capture changed during alignment")
        record = {
            "id": capture_id,
            "split": split,
            "device_id": device_id,
            "session_id": session_id,
            "player_id": player_id,
            "source_program_id": source_program_id,
            "clean": _relative(manifest_root, clean_output_path),
            "wet": _relative(manifest_root, wet_output_path),
            "clean_sha256": sha256(clean_output_path),
            "wet_sha256": sha256(wet_output_path),
            "raw_program": _relative(manifest_root, program_path),
            "raw_program_sha256": before_hashes[program_path],
            "raw_latency_capture": _relative(manifest_root, latency_capture_path),
            "raw_latency_capture_sha256": before_hashes[latency_capture_path],
            "raw_wet_capture": _relative(manifest_root, wet_raw_path),
            "raw_wet_capture_sha256": before_hashes[wet_raw_path],
            "measured_latency_samples": latency,
            "latency_compensated": True,
            "capture_quality": quality,
            "chain": ChainSpec((drive,)).document(),
        }
        manifest = existing_manifest or {
            "schema": MANIFEST_SCHEMA,
            "dataset": {
                "id": dataset_id,
                "source_kind": source_kind,
                "rights": rights,
                "intended_use": "drive-device-adapter-training",
                "audio_quality_policy": "lossless-immutable-raw-v1",
            },
            "records": [],
        }
        manifest["records"].append(record)
        pending.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        try:
            validation = validate_manifest(pending)
            pending.replace(manifest_path)
        finally:
            pending.unlink(missing_ok=True)
    except Exception:
        pending.unlink(missing_ok=True)
        clean_output_path.unlink(missing_ok=True)
        wet_output_path.unlink(missing_ok=True)
        raise
    validation["manifest"] = str(manifest_path)
    return {"manifest": manifest, "validation": validation}


def smoke_report() -> dict:
    """Exercise plan coverage, latency, alignment, and a nonlinear Drive path."""

    plan = drive_capture_plan(device_id="smoke-drive", player_id="synthetic-player")
    coverage = plan_coverage(plan)
    pilot = drive_pilot_session(plan, "train-session-01")
    rng = np.random.default_rng(23)
    time = np.arange(5 * SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    envelope = 0.35 + 0.65 * np.square(np.sin(2.0 * np.pi * 1.7 * time))
    source = envelope * (
        0.11 * np.sin(2.0 * np.pi * 110.0 * time)
        + 0.06 * np.sin(2.0 * np.pi * 220.0 * time)
        + 0.02 * rng.standard_normal(len(time))
    )
    source = source.astype(np.float32)
    program, layout = build_capture_program(source, SAMPLE_RATE)
    latency = 173

    def delayed(value: np.ndarray) -> np.ndarray:
        result = np.zeros_like(value)
        result[latency:] = value[:-latency]
        return result

    loopback = delayed(program * np.float32(0.8))
    measured, confidence = estimate_latency(program, loopback, layout)
    drive = Drive(12.0, 0.62, -12.0)
    effected = render_chain(program, ChainSpec((drive,)), SAMPLE_RATE, "alternate")
    raw_wet = delayed(effected)
    clean, wet = align_payload(program, raw_wet, layout, measured)
    expected = render_chain(clean, ChainSpec((drive,)), SAMPLE_RATE, "alternate")
    quality = capture_quality_report(
        clean,
        wet,
        raw_wet,
        layout,
        latency_frames=measured,
        latency_confidence=confidence,
    )
    clipped = np.clip(raw_wet * 50.0, -1.0, 1.0)
    _, clipped_wet = align_payload(program, clipped, layout, measured)
    clipped_quality = capture_quality_report(
        clean,
        clipped_wet,
        clipped,
        layout,
        latency_frames=measured,
        latency_confidence=confidence,
    )
    anchor_after = raw_wet * np.float32(10.0 ** (0.1 / 20.0))
    stable_drift = drive_anchor_drift_report(
        program,
        loopback,
        raw_wet,
        anchor_after,
        layout,
    )
    changed_effect = render_chain(
        program,
        ChainSpec((Drive(12.0, 0.08, -12.0),)),
        SAMPLE_RATE,
        "alternate",
    )
    changed_drift = drive_anchor_drift_report(
        program,
        loopback,
        raw_wet,
        delayed(changed_effect),
        layout,
    )
    return {
        "schema": 1,
        "sample_rate": SAMPLE_RATE,
        "plan": {
            "training_records": plan["training_records"],
            "sessions": len(plan["sessions"]),
            "coverage": coverage,
            "pilot_takes": len(pilot["takes"]),
        },
        "latency": {
            "injected_samples": latency,
            "measured_samples": measured,
            "confidence": confidence,
        },
        "alignment": {
            "frames": len(clean),
            "max_absolute_error": float(np.max(np.abs(wet - expected))),
        },
        "quality": quality,
        "negative_gate": {
            "clipped_capture_rejected": not clipped_quality["passed"],
            "failures": clipped_quality["failures"],
        },
        "drift_gate": {
            "stable_anchor_passed": stable_drift["passed"],
            "stable_level_drift_db": stable_drift["level_drift_db"],
            "stable_gain_aligned_nrmse": stable_drift["gain_aligned_nrmse"],
            "changed_anchor_rejected": not changed_drift["passed"],
            "changed_failures": changed_drift["failures"],
        },
    }


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="write a Drive capture plan")
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--device-id", required=True)
    plan_parser.add_argument("--player-id", required=True)
    plan_parser.add_argument("--seed", type=int, default=DEFAULT_PLAN_SEED)
    pilot_parser = commands.add_parser("pilot", help="write an ordered 16-setting take sheet")
    pilot_parser.add_argument("--plan", type=Path, required=True)
    pilot_parser.add_argument("--output", type=Path, required=True)
    pilot_parser.add_argument("--session-id")
    program_parser = commands.add_parser("program", help="create a replay program")
    program_parser.add_argument("--source", type=Path, required=True)
    program_parser.add_argument("--output", type=Path, required=True)
    program_parser.add_argument("--metadata", type=Path, required=True)
    align_parser = commands.add_parser("align", help="align and admit one Drive capture")
    align_parser.add_argument("--program", type=Path, required=True)
    align_parser.add_argument("--program-metadata", type=Path, required=True)
    align_parser.add_argument("--latency-capture", type=Path, required=True)
    align_parser.add_argument("--wet-raw", type=Path, required=True)
    align_parser.add_argument("--clean-out", type=Path, required=True)
    align_parser.add_argument("--wet-out", type=Path, required=True)
    align_parser.add_argument("--manifest", type=Path, required=True)
    align_parser.add_argument("--capture-id", required=True)
    align_parser.add_argument("--split", required=True)
    align_parser.add_argument("--device-id", required=True)
    align_parser.add_argument("--session-id", required=True)
    align_parser.add_argument("--player-id", required=True)
    align_parser.add_argument(
        "--source-program-id",
        help="split-local replay-program identity; defaults to <player>-<split>-program",
    )
    align_parser.add_argument("--dataset-id")
    align_parser.add_argument(
        "--rights",
        required=True,
        choices=("owned", "authorized-for-device-adapter-training"),
    )
    align_parser.add_argument("--gain-db", type=float, required=True)
    align_parser.add_argument("--tone", type=float, required=True)
    align_parser.add_argument("--level-db", type=float, required=True)
    align_parser.add_argument("--append", action="store_true")
    drift_parser = commands.add_parser("drift", help="audit repeated session anchors")
    drift_parser.add_argument("--program", type=Path, required=True)
    drift_parser.add_argument("--program-metadata", type=Path, required=True)
    drift_parser.add_argument("--latency-capture", type=Path, required=True)
    drift_parser.add_argument("--anchor-before", type=Path, required=True)
    drift_parser.add_argument("--anchor-after", type=Path, required=True)
    drift_parser.add_argument("--output", type=Path)
    smoke_parser = commands.add_parser("smoke", help="run an in-memory capture-path audit")
    smoke_parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.command == "plan":
        document = drive_capture_plan(
            device_id=args.device_id,
            player_id=args.player_id,
            seed=args.seed,
        )
        document["coverage"] = plan_coverage(document)
        _write_json(args.output, document)
        print(json.dumps({"output": str(args.output), "records": len(document["records"])}, indent=2))
    elif args.command == "pilot":
        document = drive_pilot_session(json.loads(args.plan.read_text()), args.session_id)
        _write_json(args.output, document)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "session_id": document["session_id"],
                    "takes": len(document["takes"]),
                },
                indent=2,
            )
        )
    elif args.command == "program":
        print(json.dumps(write_capture_program(args.source, args.output, args.metadata), indent=2))
    elif args.command == "align":
        result = prepare_capture_record(
            program_path=args.program,
            program_metadata_path=args.program_metadata,
            latency_capture_path=args.latency_capture,
            wet_raw_path=args.wet_raw,
            clean_output_path=args.clean_out,
            wet_output_path=args.wet_out,
            manifest_path=args.manifest,
            capture_id=args.capture_id,
            split=args.split,
            device_id=args.device_id,
            session_id=args.session_id,
            player_id=args.player_id,
            source_program_id=args.source_program_id,
            dataset_id=args.dataset_id,
            rights=args.rights,
            drive=Drive(args.gain_db, args.tone, args.level_db),
            append=args.append,
        )
        print(json.dumps(result["validation"], indent=2, sort_keys=True))
    elif args.command == "drift":
        metadata = json.loads(args.program_metadata.read_text())
        layout = CaptureLayout(**metadata["layout"])
        program, program_rate = _read_audio(args.program)
        loopback, loopback_rate = _read_audio(args.latency_capture)
        anchor_before, before_rate = _read_audio(args.anchor_before)
        anchor_after, after_rate = _read_audio(args.anchor_after)
        if {program_rate, loopback_rate, before_rate, after_rate} != {SAMPLE_RATE}:
            raise ValueError("all drift-audit files must use 48 kHz")
        document = drive_anchor_drift_report(
            program,
            loopback,
            anchor_before,
            anchor_after,
            layout,
        )
        if args.output:
            _write_json(args.output, document)
        print(json.dumps(document, indent=2, sort_keys=True))
        if not document["passed"]:
            raise SystemExit(2)
    else:
        document = smoke_report()
        if args.output:
            _write_json(args.output, document)
        print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
