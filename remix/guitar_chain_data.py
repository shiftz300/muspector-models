"""Presence-only view of the CC BY 4.0 random-position guitar-chain corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .blind2 import LABELS


SOURCE_ID = "dafx25-guitar-effects-chains"
DIRECTORIES = {
    "clean": "0__unprocessed_samples",
    "random_position": "3__Dataset3_vary_params_and_position",
}
SPLITS = ("fit", "calibration", "development", "locked-final")


@dataclass(frozen=True)
class ChainPresence:
    path: Path
    group: str
    guitar: str
    split: str
    target: np.ndarray
    presence_bits: str


def split_for_group(group: str) -> str:
    """Keep every derivative of one clean performance in exactly one split."""

    guitar = group.split("_", 1)[0]
    if guitar in {"les", "prs"}:
        return "fit"
    if guitar == "tele":
        return "locked-final"
    if guitar == "strat":
        match = re.search(r"(\d+)$", group)
        if match is None:
            raise ValueError(f"Strat performance has no take number: {group}")
        return "calibration" if int(match.group(1)) <= 12 else "development"
    raise ValueError(f"unknown guitar-chain instrument: {group}")


def target_from_bits(bits: str) -> np.ndarray:
    """Map overdrive/chorus/tremolo/delay/reverb presence to product families."""

    if len(bits) != 5 or set(bits) - {"0", "1"}:
        raise ValueError(f"invalid effect-presence bits: {bits!r}")
    overdrive, chorus, tremolo, delay, reverb = (value == "1" for value in bits)
    values = {
        "nonlinear": overdrive,
        "echo": delay,
        "ambience": reverb,
        "unknown": chorus or tremolo,
    }
    return np.asarray([float(values[label]) for label in LABELS], dtype=np.float32)


def parse_effected(path: Path) -> ChainPresence:
    try:
        group, bits = path.stem.rsplit("__", 1)
    except ValueError as error:
        raise ValueError(f"effected filename has no presence suffix: {path.name}") from error
    return ChainPresence(
        path.resolve(),
        group,
        group.split("_", 1)[0],
        split_for_group(group),
        target_from_bits(bits),
        bits,
    )


def _resolve_root(root: Path) -> Path:
    root = root.resolve()
    nested = root / "DATASET_guitar_effects" / "DATASET_guitar_effects"
    if not (root / DIRECTORIES["clean"]).is_dir() and nested.is_dir():
        return nested
    return root


def discover(root: Path, split: str) -> list[ChainPresence]:
    if split not in SPLITS:
        raise ValueError(f"invalid split: {split}")
    root = _resolve_root(root)
    clean_root = root / DIRECTORIES["clean"]
    effected_root = root / DIRECTORIES["random_position"]
    if not clean_root.is_dir() or not effected_root.is_dir():
        raise ValueError(f"incomplete guitar-chain corpus: {root}")

    rows = [
        parse_effected(path)
        for path in sorted(effected_root.rglob("*.wav"))
        if split_for_group(path.stem.rsplit("__", 1)[0]) == split
    ]
    for path in sorted(clean_root.rglob("*.wav")):
        group = path.stem
        if split_for_group(group) != split:
            continue
        rows.append(
            ChainPresence(
                path.resolve(),
                group,
                group.split("_", 1)[0],
                split,
                np.zeros(len(LABELS), dtype=np.float32),
                "00000",
            )
        )
    return sorted(rows, key=lambda row: (row.group, row.presence_bits, row.path.name))


def inventory(root: Path, *, include_locked_final: bool = False) -> dict:
    resolved = _resolve_root(root)
    visible = SPLITS if include_locked_final else SPLITS[:-1]
    rows = {split: discover(root, split) for split in visible}
    splits = {
        split: {
            "files": len(values),
            "groups": len({row.group for row in values}),
            "guitars": sorted({row.guitar for row in values}),
        }
        for split, values in rows.items()
    }
    if not include_locked_final:
        splits["locked-final"] = {"sealed": True}
    return {
        "source_id": SOURCE_ID,
        "presence_labels_only": True,
        "order_labels_consumed": False,
        "author_order_metadata_exists": (
            resolved / DIRECTORIES["random_position"] / "file_effects_order_vary_params.json"
        ).is_file(),
        "splits": splits,
    }
