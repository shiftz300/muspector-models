"""Fail-closed manifest reader for the physical multi-effect presence veto."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .blind2 import LABELS


FORBIDDEN_KEY_PARTS = ("order", "position", "sequence", "slot")
MIN_CLEAN_EXAMPLES = 73
MIN_GROUPS = 12
MIN_HARDWARE_GROUPS = 2
MIN_POSITIVES_PER_FAMILY = 25
MIN_POSITIVES_PER_HARDWARE_FAMILY = 5


@dataclass(frozen=True)
class PhysicalChainPresence:
    path: Path
    group: str
    hardware_group: str
    capture_variant_id: str
    target: np.ndarray


def _reject_order_metadata(value: object, location: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in FORBIDDEN_KEY_PARTS):
                raise ValueError(f"order metadata is forbidden at {location}.{key}")
            _reject_order_metadata(nested, f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_order_metadata(nested, f"{location}[{index}]")


def _required_text(row: dict, key: str, location: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be non-empty text")
    return value.strip()


def discover(manifest_path: Path) -> tuple[dict, list[PhysicalChainPresence]]:
    """Read a veto-only manifest without exposing pedal order to inference."""

    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise ValueError("physical veto manifest must use schema 1")
    _reject_order_metadata(manifest)
    if manifest.get("scope") != "physical-multi-effect-veto":
        raise ValueError("manifest scope must be physical-multi-effect-veto")
    if manifest.get("physical_hardware") is not True:
        raise ValueError("physical_hardware must be true")
    if manifest.get("product_evaluation_authorized") is not True:
        raise ValueError("product_evaluation_authorized must be true")
    _required_text(manifest, "source_id", "manifest")
    _required_text(manifest, "rights_basis", "manifest")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest.rows must be a non-empty list")

    root = manifest_path.parent
    discovered: list[PhysicalChainPresence] = []
    seen_paths: set[Path] = set()
    for index, raw in enumerate(rows):
        location = f"manifest.rows[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{location} must be an object")
        relative = Path(_required_text(raw, "path", location))
        if relative.is_absolute():
            raise ValueError(f"{location}.path must be relative to the manifest")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{location}.path escapes the manifest directory") from error
        if path.suffix.lower() != ".wav" or not path.is_file():
            raise ValueError(f"{location}.path must name an existing WAV file")
        if path in seen_paths:
            raise ValueError(f"duplicate audio path: {relative}")
        seen_paths.add(path)

        families = raw.get("families")
        if not isinstance(families, list) or any(not isinstance(item, str) for item in families):
            raise ValueError(f"{location}.families must be a list of family names")
        if len(families) != len(set(families)):
            raise ValueError(f"{location}.families contains duplicates")
        unknown = set(families) - set(LABELS)
        if unknown:
            raise ValueError(f"{location}.families has unknown labels: {sorted(unknown)}")
        if len(families) == 1:
            raise ValueError(f"{location} is single-effect; this veto requires physical chains")
        target = np.asarray([float(label in set(families)) for label in LABELS], dtype=np.float32)
        discovered.append(
            PhysicalChainPresence(
                path=path,
                group=_required_text(raw, "group", location),
                hardware_group=_required_text(raw, "hardware_group", location),
                capture_variant_id=_required_text(raw, "capture_variant_id", location),
                target=target,
            )
        )
    return manifest, discovered


def inventory(rows: list[PhysicalChainPresence]) -> dict:
    expected = np.stack([row.target for row in rows]).astype(bool)
    clean = ~expected.any(axis=1)
    hardware_groups = sorted({row.hardware_group for row in rows if row.target.any()})
    family_positives = {
        label: int(expected[:, index].sum()) for index, label in enumerate(LABELS)
    }
    per_hardware_family_positives = {
        hardware: {
            label: int(sum(row.hardware_group == hardware and bool(row.target[index]) for row in rows))
            for index, label in enumerate(LABELS)
        }
        for hardware in hardware_groups
    }
    coverage_gates = {
        "clean_examples": int(clean.sum()) >= MIN_CLEAN_EXAMPLES,
        "dry_performance_groups": len({row.group for row in rows}) >= MIN_GROUPS,
        "hardware_groups": len(hardware_groups) >= MIN_HARDWARE_GROUPS,
        "each_family_positives": all(
            count >= MIN_POSITIVES_PER_FAMILY for count in family_positives.values()
        ),
        "each_hardware_family_positives": bool(per_hardware_family_positives)
        and all(
            count >= MIN_POSITIVES_PER_HARDWARE_FAMILY
            for values in per_hardware_family_positives.values()
            for count in values.values()
        ),
    }
    return {
        "files": len(rows),
        "clean_examples": int(clean.sum()),
        "effect_examples": int((~clean).sum()),
        "dry_performance_groups": len({row.group for row in rows}),
        "hardware_groups": hardware_groups,
        "family_positives": family_positives,
        "per_hardware_family_positives": per_hardware_family_positives,
        "minimums": {
            "clean_examples": MIN_CLEAN_EXAMPLES,
            "dry_performance_groups": MIN_GROUPS,
            "hardware_groups": MIN_HARDWARE_GROUPS,
            "family_positives": MIN_POSITIVES_PER_FAMILY,
            "hardware_family_positives": MIN_POSITIVES_PER_HARDWARE_FAMILY,
        },
        "coverage_gates": coverage_gates,
        "coverage_passed": all(coverage_gates.values()),
    }
