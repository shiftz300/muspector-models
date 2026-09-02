"""Audited real-order and synthetic-control paired datasets."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .order_identifiability import perceptual_order_targets
from .quality import analysis_pair
from .render import RENDERERS, Renderer, render_chain
from .spec import (
    ChainSpec,
    Delay,
    Drive,
    KINDS,
    Reverb,
    control_targets,
    identifiable_order_targets,
    order_targets,
)


RATE = 44_100
SECONDS = 5
SAMPLES = RATE * SECONDS
TOPOLOGIES = tuple(
    topology
    for size in range(len(KINDS) + 1)
    for topology in __import__("itertools").permutations(KINDS, size)
)
UPSTREAM_NAMES = {"overdrive": "drive", "delay": "delay", "reverb": "reverb"}
KIND_INDEX = {kind: index for index, kind in enumerate(KINDS)}


def topology_tensor(order: Sequence[str]) -> torch.Tensor:
    values = [KIND_INDEX[kind] for kind in order]
    values.extend([-1] * (len(KINDS) - len(values)))
    return torch.tensor(values, dtype=torch.int64)


@dataclass(frozen=True)
class OrderRecord:
    wet: Path
    dry: Path
    group: str
    split: str
    order: tuple[str, ...]


def _performance(filename: str) -> str:
    return filename.split("__", 1)[0]


def _split(filename: str) -> str:
    performance = _performance(filename)
    guitar = performance.split("_", 1)[0]
    if guitar in {"les", "prs"}:
        return "train"
    if guitar == "tele":
        return "test"
    if guitar == "strat":
        # Raw dry sources retain their .wav suffix while Dataset 3 effect
        # filenames are stripped at ``__``. Parse the stem in both cases so
        # Strat 13-25 cannot leak into the synthetic validation source pool.
        match = re.search(r"(\d+)$", Path(performance).stem)
        take = int(match.group(1)) if match else 0
        return "valid" if take <= 12 else "calibrate"
    raise ValueError(f"unknown guitar in {filename}")


def _guitar_directory(filename: str) -> str:
    guitar = filename.split("_", 1)[0]
    return "str" if guitar == "strat" else guitar


def discover_order_records(root: Path) -> list[OrderRecord]:
    """Read author-provided Dataset 3 order labels without inferring filenames."""

    base = root / "DATASET_guitar_effects" / "DATASET_guitar_effects"
    variant = base / "3__Dataset3_vary_params_and_position"
    metadata = variant / "file_effects_order_vary_params.json"
    payload = json.loads(metadata.read_text())
    if len(payload["file_name"]) != len(payload["effects_order"]):
        raise ValueError("DAFx order metadata columns have different lengths")
    records = []
    for filename, upstream_order in zip(payload["file_name"], payload["effects_order"]):
        order = tuple(UPSTREAM_NAMES[name] for name in upstream_order if name in UPSTREAM_NAMES)
        performance = _performance(filename)
        wet = variant / _guitar_directory(filename) / filename
        dry = base / "0__unprocessed_samples" / f"{performance}.wav"
        if not wet.is_file() or not dry.is_file():
            raise FileNotFoundError(f"incomplete DAFx pair: {dry} / {wet}")
        records.append(OrderRecord(wet, dry, performance, _split(filename), order))
    if len(records) != 12_800:
        raise ValueError(f"expected 12800 DAFx order records, found {len(records)}")
    return records


def order_manifest(root: Path) -> dict:
    records = discover_order_records(root)
    counts = Counter(record.split for record in records)
    topologies = Counter("->".join(record.order) or "clean" for record in records)
    metadata = (
        root
        / "DATASET_guitar_effects"
        / "DATASET_guitar_effects"
        / "3__Dataset3_vary_params_and_position"
        / "file_effects_order_vary_params.json"
    )
    return {
        "schema": 1,
        "source": "https://doi.org/10.5281/zenodo.7871720",
        "license": "CC-BY-4.0",
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "policy": {
            "target_effects": list(KINDS),
            "ignored_hard_negatives": ["chorus", "tremolo"],
            "split": "Les Paul and PRS train; Strat 1-12 valid and 13-25 calibrate; Telecaster locked test",
            "parameters": "varied upstream but exact values are not published; never use as knob labels",
        },
        "counts": {
            "records": len(records),
            "splits": dict(sorted(counts.items())),
            "topologies": dict(sorted(topologies.items())),
        },
    }


def _waveform(path: Path, offset: int = 0) -> np.ndarray:
    value, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    value = value.mean(axis=1)
    if sample_rate != RATE:
        common = math.gcd(sample_rate, RATE)
        value = resample_poly(value, RATE // common, sample_rate // common).astype(np.float32)
    if len(value) < SAMPLES:
        value = np.pad(value, (0, SAMPLES - len(value)))
    maximum = max(0, len(value) - SAMPLES)
    start = min(max(offset, 0), maximum)
    return np.asarray(value[start : start + SAMPLES], dtype=np.float32)


def _pair(record: OrderRecord) -> tuple[np.ndarray, np.ndarray]:
    dry = _waveform(record.dry)
    wet = _waveform(record.wet)
    return analysis_pair(dry, wet)


class DAFxOrderDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        self.records = [record for record in discover_order_records(root) if record.split == split]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        dry, wet = _pair(record)
        order, order_mask = order_targets(record.order)
        return {
            "dry": torch.from_numpy(dry.copy()),
            "wet": torch.from_numpy(wet.copy()),
            "order": torch.tensor(order, dtype=torch.float32),
            "order_mask": torch.tensor(order_mask, dtype=torch.float32),
            "order_present": torch.tensor(order_mask, dtype=torch.float32),
            "topology": topology_tensor(record.order),
            "renderer": torch.tensor(-1, dtype=torch.int64),
            "reconstruction_mask": torch.tensor(0.0, dtype=torch.float32),
            "controls": torch.zeros(9, dtype=torch.float32),
            "control_mask": torch.zeros(9, dtype=torch.float32),
            "control_train_mask": torch.zeros(9, dtype=torch.float32),
        }


def random_spec(rng: random.Random, topology: Sequence[str]) -> ChainSpec:
    effects = []
    for kind in topology:
        if kind == "drive":
            effects.append(Drive(rng.uniform(0.0, 30.0), rng.random(), rng.uniform(-12.0, 6.0)))
        elif kind == "delay":
            effects.append(
                Delay(
                    40.0 * (1_000.0 / 40.0) ** rng.random(),
                    rng.uniform(0.0, 0.85),
                    rng.uniform(0.05, 0.7),
                )
            )
        elif kind == "reverb":
            effects.append(
                Reverb(
                    0.2 * (8.0 / 0.2) ** rng.random(),
                    rng.random(),
                    rng.uniform(0.05, 0.7),
                )
            )
        else:
            raise ValueError(f"unknown topology effect: {kind}")
    return ChainSpec(tuple(effects))


class SyntheticControlDataset(Dataset):
    """Generate aligned pairs while retaining every sampled knob value."""

    def __init__(
        self,
        dry_paths: Sequence[Path],
        samples: int,
        seed: int = 20260830,
        renderers: Sequence[Renderer] = RENDERERS,
        order_equivalence_db: float | None = None,
    ) -> None:
        if not dry_paths:
            raise ValueError("at least one dry source is required")
        self.dry_paths = tuple(dry_paths)
        self.samples = samples
        self.seed = seed
        if not renderers:
            raise ValueError("at least one renderer is required")
        self.renderers = tuple(renderers)
        self.order_equivalence_db = order_equivalence_db

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = random.Random(self.seed + index * 104_729)
        path = self.dry_paths[index % len(self.dry_paths)]
        info = soundfile.info(path)
        frames = round(info.frames * RATE / info.samplerate)
        offset = rng.randrange(max(1, frames - SAMPLES + 1))
        dry = _waveform(path, offset)
        peak = max(float(np.max(np.abs(dry))), 1.0e-5)
        dry = dry * (0.22 / peak)
        topology = TOPOLOGIES[index % len(TOPOLOGIES)]
        spec = random_spec(rng, topology)
        renderer_index = (index // len(TOPOLOGIES)) % len(self.renderers)
        renderer = self.renderers[renderer_index]
        wet = render_chain(dry, spec, RATE, renderer)
        order, order_present = order_targets(topology)
        if self.order_equivalence_db is None:
            _, order_mask = identifiable_order_targets(topology)
        else:
            _, order_mask = perceptual_order_targets(
                dry,
                wet,
                spec,
                lambda value, candidate: render_chain(value, candidate, RATE, renderer),
                self.order_equivalence_db,
            )
        controls, control_mask = control_targets(spec)
        control_train_mask = list(control_mask)
        control_train_mask[3:6] = [0.0] * 3
        return {
            "dry": torch.from_numpy(dry.copy()),
            "wet": torch.from_numpy(wet.copy()),
            "order": torch.tensor(order, dtype=torch.float32),
            "order_mask": torch.tensor(order_mask, dtype=torch.float32),
            "order_present": torch.tensor(order_present, dtype=torch.float32),
            "topology": topology_tensor(topology),
            "renderer": torch.tensor(renderer_index, dtype=torch.int64),
            # The current differentiable surrogate has verified parity only
            # for Drive/Delay. Reverb stays under direct control supervision.
            "reconstruction_mask": torch.tensor(
                float(bool(topology) and "reverb" not in topology), dtype=torch.float32
            ),
            "controls": torch.tensor(controls, dtype=torch.float32),
            "control_mask": torch.tensor(control_mask, dtype=torch.float32),
            "control_train_mask": torch.tensor(control_train_mask, dtype=torch.float32),
        }


def dry_sources(root: Path, split: str) -> list[Path]:
    base = root / "DATASET_guitar_effects" / "DATASET_guitar_effects" / "0__unprocessed_samples"
    result = []
    for path in sorted(base.glob("*.wav")):
        if _split(path.name) == split:
            result.append(path)
    return result
