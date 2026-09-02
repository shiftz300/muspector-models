"""Deterministic evaluation-only pairs rendered by Spotify Pedalboard."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile
import torch
from torch.utils.data import Dataset

from .data import RATE, SAMPLES, TOPOLOGIES, _waveform, topology_tensor
from .pedalboard_renderer import (
    MAX_REVERB_DECAY_SECONDS,
    MIN_REVERB_DECAY_SECONDS,
    render_pedalboard_chain,
)
from .order_identifiability import perceptual_order_targets
from .spec import (
    ChainSpec,
    Delay,
    Drive,
    Reverb,
    control_targets,
    identifiable_order_targets,
    order_targets,
)


def random_pedalboard_spec(rng: random.Random, topology: Sequence[str]) -> ChainSpec:
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
                    MIN_REVERB_DECAY_SECONDS
                    * (MAX_REVERB_DECAY_SECONDS / MIN_REVERB_DECAY_SECONDS)
                    ** rng.random(),
                    rng.random(),
                    rng.uniform(0.05, 0.7),
                )
            )
        else:
            raise ValueError(f"unknown topology effect: {kind}")
    return ChainSpec(tuple(effects))


class PedalboardControlDataset(Dataset):
    """Generate an external DSP holdout without entering any training path."""

    def __init__(
        self,
        dry_paths: Sequence[Path],
        samples: int,
        seed: int = 20260904,
        order_equivalence_db: float | None = None,
    ) -> None:
        if not dry_paths:
            raise ValueError("at least one dry source is required")
        self.dry_paths = tuple(dry_paths)
        self.samples = samples
        self.seed = seed
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
        spec = random_pedalboard_spec(rng, topology)
        wet = render_pedalboard_chain(dry, spec, RATE)
        order, order_present = order_targets(topology)
        if self.order_equivalence_db is None:
            _, order_mask = identifiable_order_targets(topology)
        else:
            _, order_mask = perceptual_order_targets(
                dry,
                wet,
                spec,
                lambda value, candidate: render_pedalboard_chain(value, candidate, RATE),
                self.order_equivalence_db,
            )
        controls, control_mask = control_targets(spec)
        return {
            "dry": torch.from_numpy(dry.copy()),
            "wet": torch.from_numpy(wet.copy()),
            "order": torch.tensor(order, dtype=torch.float32),
            "order_mask": torch.tensor(order_mask, dtype=torch.float32),
            "order_present": torch.tensor(order_present, dtype=torch.float32),
            "topology": topology_tensor(topology),
            "renderer": torch.tensor(-2, dtype=torch.int64),
            "reconstruction_mask": torch.tensor(0.0, dtype=torch.float32),
            "controls": torch.tensor(controls, dtype=torch.float32),
            "control_mask": torch.tensor(control_mask, dtype=torch.float32),
            "control_train_mask": torch.zeros(9, dtype=torch.float32),
        }
