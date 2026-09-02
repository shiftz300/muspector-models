#!/usr/bin/env python3
"""Offline data-contract and readiness audits for real-device Capture Packs.

This module only reads manifests and ordinary files. It does not enumerate,
open, configure, or record from an audio device.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .capture_kit import DEFAULT_SPLIT_COUNTS, SAMPLE_RATE, plan_coverage
from .capture_manifest import validate_manifest


REGISTRY_SCHEMA = 1
PLAN_MINIMUMS = dict(DEFAULT_SPLIT_COUNTS)
DEVELOPMENT_SPLITS = frozenset({"train", "calibrate", "valid"})
LOCKED_SPLIT = "locked-final"
CAPTURE_RIGHTS = {"owned", "authorized-for-device-adapter-training", "synthetic-smoke"}


def _issue(code: str, severity: str, message: str, **evidence: object) -> dict:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "evidence": evidence,
    }


def validate_source_registry(path: Path, *, workspace: Path | None = None) -> dict:
    """Validate source rights and report local availability without reading audio."""

    document = json.loads(path.read_text())
    if document.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(f"unsupported source registry schema: {document.get('schema')}")
    sources = document.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source registry must contain sources")
    root = workspace or path.parent.parent
    identifiers: set[str] = set()
    issues: list[dict] = []
    local = 0
    admitted = Counter()
    entries = []
    for source in sources:
        identifier = str(source.get("id", "")).strip()
        if not identifier or identifier in identifiers:
            raise ValueError(f"source id is empty or duplicated: {identifier!r}")
        identifiers.add(identifier)
        for field in (
            "title",
            "license",
            "rights_status",
            "allowed_uses",
            "pairing",
            "control_labels",
            "audio_quality_caveats",
            "admission",
        ):
            if field not in source:
                raise ValueError(f"source {identifier} is missing {field}")
        allowed = source["allowed_uses"]
        if not isinstance(allowed, list) or not allowed:
            raise ValueError(f"source {identifier} has no allowed uses")
        rights = str(source["rights_status"])
        rights_verified = (
            rights == "verified"
            or rights.startswith("verified-")
            or rights == "project-owned"
        )
        if not rights_verified and any(
            use.startswith("train") or use == "redistribute-derived-model" for use in allowed
        ):
            raise ValueError(
                f"source {identifier} grants training/redistribution with unverified rights"
            )
        paths = [root / value for value in source.get("local_paths", [])]
        present = [str(path.relative_to(root)) for path in paths if path.exists()]
        local += bool(present)
        admitted[str(source["admission"])] += 1
        if not rights_verified:
            issues.append(
                _issue(
                    "rights-unresolved",
                    "blocker" if source["admission"] == "metadata-only" else "warning",
                    f"{identifier} is not eligible for unrestricted model training",
                    rights_status=rights,
                    allowed_uses=allowed,
                )
            )
        entries.append(
            {
                "id": identifier,
                "admission": source["admission"],
                "rights_status": rights,
                "local_paths_present": present,
                "local": bool(present),
            }
        )
    return {
        "schema": REGISTRY_SCHEMA,
        "registry": str(path),
        "sources": len(sources),
        "sources_local": local,
        "admission_counts": dict(sorted(admitted.items())),
        "entries": entries,
        "issues": issues,
        "audio_files_opened": 0,
        "audio_devices_accessed": 0,
        "passed": True,
    }


def audit_drive_plan(plan: dict) -> dict:
    """Audit split, session, replay-program, and control-space coverage."""

    issues: list[dict] = []
    if plan.get("schema") != 1 or plan.get("sample_rate") != SAMPLE_RATE:
        raise ValueError("unsupported Drive capture plan")
    records = plan.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Drive capture plan has no records")
    split_counts = Counter(str(record.get("split")) for record in records)
    sessions: dict[tuple[str, str], set[str]] = defaultdict(set)
    programs: dict[str, set[str]] = defaultdict(set)
    for record in records:
        split = str(record.get("split"))
        session = str(record.get("session_id", ""))
        program = str(record.get("source_program_id", ""))
        sessions[(str(plan.get("device_id", "")), session)].add(split)
        programs[program].add(split)
    session_leaks = {
        f"{device}/{session}": sorted(splits)
        for (device, session), splits in sessions.items()
        if not session or len(splits) != 1
    }
    program_leaks = {
        program: sorted(splits)
        for program, splits in programs.items()
        if not program or len(splits) != 1
    }
    if session_leaks:
        issues.append(
            _issue(
                "session-split-leakage",
                "blocker",
                "a device session crosses data splits",
                sessions=session_leaks,
            )
        )
    if program_leaks:
        issues.append(
            _issue(
                "replay-program-split-leakage",
                "blocker",
                "a replay program crosses data splits",
                programs=program_leaks,
            )
        )
    coverage = plan_coverage(plan)
    for split, minimum in PLAN_MINIMUMS.items():
        actual = split_counts.get(split, 0)
        if actual < minimum:
            issues.append(
                _issue(
                    "insufficient-split-volume",
                    "blocker",
                    f"{split} has {actual} settings; requires at least {minimum}",
                    split=split,
                    actual=actual,
                    required=minimum,
                )
            )
        for axis, values in coverage.get(split, {}).get("axes", {}).items():
            if not values.get("strata_complete"):
                issues.append(
                    _issue(
                        "incomplete-control-strata",
                        "blocker",
                        f"{split}/{axis} does not visit every Latin-hypercube stratum",
                        split=split,
                        axis=axis,
                    )
                )
    return {
        "schema": 1,
        "grain": "one device/session/setting/replay-program aligned Clean/Wet pair",
        "records": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "sessions": len(sessions),
        "source_programs": len(programs),
        "session_disjoint": not session_leaks,
        "replay_program_disjoint": not program_leaks,
        "coverage": coverage,
        "issues": issues,
        "passed": not any(issue["severity"] == "blocker" for issue in issues),
        "audio_files_opened": 0,
        "audio_devices_accessed": 0,
    }


def _normalized_drive_controls(records: Iterable[dict]) -> dict[str, list[float]]:
    controls = {"gain": [], "tone": [], "level": []}
    for record in records:
        effects = record.get("chain", {}).get("effects", [])
        drives = [effect for effect in effects if effect.get("kind") == "drive"]
        if len(drives) != 1:
            continue
        drive = drives[0]
        controls["gain"].append(float(drive["gain_db"]) / 30.0)
        controls["tone"].append(float(drive["tone"]))
        controls["level"].append((float(drive["level_db"]) + 18.0) / 30.0)
    return controls


def audit_capture_manifest(path: Path, *, development: bool = True) -> dict:
    """Audit a real pack; development mode never opens locked-final audio."""

    excluded = frozenset({LOCKED_SPLIT}) if development else frozenset()
    document = json.loads(path.read_text())
    dataset = document.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Capture Pack is missing top-level dataset provenance")
    if dataset.get("source_kind") not in {"hardware-capture", "synthetic-smoke"}:
        raise ValueError(f"unsupported Capture Pack source_kind: {dataset.get('source_kind')!r}")
    if dataset.get("rights") not in CAPTURE_RIGHTS:
        raise ValueError("Capture Pack has no usable rights attestation")
    if dataset.get("intended_use") != "drive-device-adapter-training":
        raise ValueError("Capture Pack is not admitted for Drive adapter training")
    if not str(dataset.get("id", "")).strip():
        raise ValueError("Capture Pack dataset id is empty")
    validation = validate_manifest(path, excluded_splits=excluded)
    development_records = [
        record for record in document["records"] if record.get("split") in DEVELOPMENT_SPLITS
    ]
    issues: list[dict] = []
    counts = Counter(record["split"] for record in document["records"])
    for split, minimum in PLAN_MINIMUMS.items():
        if counts.get(split, 0) < minimum:
            issues.append(
                _issue(
                    "insufficient-split-volume",
                    "blocker",
                    f"{split} has {counts.get(split, 0)} records; requires at least {minimum}",
                    split=split,
                    actual=counts.get(split, 0),
                    required=minimum,
                )
            )
    coverage = {}
    for split in sorted(DEVELOPMENT_SPLITS):
        split_records = [record for record in development_records if record["split"] == split]
        axes = _normalized_drive_controls(split_records)
        coverage[split] = {}
        for axis, values in axes.items():
            array = np.asarray(values, dtype=np.float64)
            occupied = len(set(np.minimum((array * 6).astype(int), 5))) if len(array) else 0
            coverage[split][axis] = {
                "minimum": float(array.min()) if len(array) else None,
                "maximum": float(array.max()) if len(array) else None,
                "occupied_six_bins": occupied,
            }
            if occupied < 6:
                issues.append(
                    _issue(
                        "incomplete-control-coverage",
                        "blocker",
                        f"{split}/{axis} occupies {occupied}/6 control bins",
                        split=split,
                        axis=axis,
                        occupied_bins=occupied,
                    )
                )
    return {
        "schema": 1,
        "mode": "development" if development else "one-shot-final",
        "dataset": dataset,
        "validation": validation,
        "coverage": coverage,
        "issues": issues,
        "locked_audio_opened": not development,
        "audio_devices_accessed": 0,
        "passed": not any(issue["severity"] == "blocker" for issue in issues),
    }
