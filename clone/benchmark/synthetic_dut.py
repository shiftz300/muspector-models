"""Repository-owned hidden DUTs for the first Structural-OOD benchmark.

The estimators never receive the topology name.  This module is the benchmark
oracle only: it creates paired input/output observations and keeps the hidden
topology in evaluation metadata outside the estimator API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


RATE = 48_000
FRAMES = 4_096
TOPOLOGIES = ("A", "B", "C", "D", "E")
TRAIN_SOURCES = tuple(range(8))
OOD_SOURCES = tuple(range(8, 12))
TRAIN_LEVELS = (0.08, 0.16, 0.28)
OOD_LEVELS = (0.38, 0.52)
TRAIN_CONTROLS = (0.20, 0.50, 0.80)
OOD_CONTROLS = (0.08, 0.92)
HISTORY_PREFIX = 1_024


@dataclass(frozen=True)
class Observation:
    """One black-box observation plus metadata held outside model fitting."""

    x: np.ndarray
    y: np.ndarray
    control: float
    topology: str
    source_id: int
    level: float
    split: str

    def public(self) -> dict[str, np.ndarray | float]:
        """Return only information allowed to a black-box estimator."""

        return {"x": self.x, "y": self.y, "control": self.control}


@dataclass(frozen=True)
class HistoryPair:
    """Same payload after two different histories, held out from fitting."""

    x_quiet: np.ndarray
    x_excited: np.ndarray
    y_quiet: np.ndarray
    y_excited: np.ndarray
    control: float
    topology: str
    payload_start: int


def make_history_pairs(topologies: Iterable[str] = TOPOLOGIES) -> list[HistoryPair]:
    rows: list[HistoryPair] = []
    for topology in topologies:
        for source_id in (12, 13):
            payload = source_signal(source_id) * 0.22
            quiet_prefix = np.zeros(HISTORY_PREFIX, dtype=np.float32)
            excited_prefix = np.zeros(HISTORY_PREFIX, dtype=np.float32)
            excited_prefix[::8] = 0.75
            quiet = np.concatenate((quiet_prefix, payload)).astype(np.float32)
            excited = np.concatenate((excited_prefix, payload)).astype(np.float32)
            control = 0.8
            rows.append(HistoryPair(
                quiet, excited,
                render_dut(topology, quiet, control),
                render_dut(topology, excited, control),
                control, topology, HISTORY_PREFIX,
            ))
    return rows


def make_history_training_observations(
    topologies: Iterable[str] = TOPOLOGIES,
) -> list[Observation]:
    """Create fit-only prefix variations for estimators with explicit state."""

    rows: list[Observation] = []
    for topology in topologies:
        for source_id in TRAIN_SOURCES[:4]:
            payload = source_signal(source_id) * 0.20
            for variant in range(3):
                prefix = np.zeros(HISTORY_PREFIX, dtype=np.float32)
                if variant == 1:
                    prefix[::8] = 0.70
                elif variant == 2:
                    prefix[:] = 0.18
                    prefix[::16] = 0.78
                value = np.concatenate((prefix, payload)).astype(np.float32)
                control = TRAIN_CONTROLS[variant]
                rows.append(Observation(
                    value, render_dut(topology, value, control), control,
                    topology, source_id, 0.20, "fit_history",
                ))
    return rows


def _one_pole_lowpass(value: np.ndarray, cutoff: float) -> np.ndarray:
    alpha = float(np.exp(-2.0 * np.pi * cutoff / RATE))
    result = np.empty_like(value, dtype=np.float64)
    state = 0.0
    for index, sample in enumerate(value):
        state = (1.0 - alpha) * float(sample) + alpha * state
        result[index] = state
    return result


def _lowpass(value: np.ndarray, cutoff: float) -> np.ndarray:
    return _one_pole_lowpass(value, cutoff)


def _highpass(value: np.ndarray, cutoff: float) -> np.ndarray:
    return value - _lowpass(value, cutoff)


def _asymmetric_clip(value: np.ndarray, drive: float) -> np.ndarray:
    positive = 0.44 + 0.14 / max(drive, 0.1)
    negative = 0.31 + 0.10 / max(drive, 0.1)
    positive_part = np.tanh(np.maximum(value, 0.0) * drive / positive) * positive
    negative_part = np.tanh(np.minimum(value, 0.0) * drive / negative) * negative
    return positive_part + negative_part


def _dynamic_bias_clip(value: np.ndarray, drive: float) -> np.ndarray:
    attack = float(np.exp(-1.0 / (0.0015 * RATE)))
    release = float(np.exp(-1.0 / (0.075 * RATE)))
    envelope = 0.0
    bias = 0.0
    result = np.empty_like(value, dtype=np.float64)
    for index, sample in enumerate(value):
        magnitude = abs(float(sample))
        coefficient = attack if magnitude > envelope else release
        envelope = coefficient * envelope + (1.0 - coefficient) * magnitude
        bias = 0.998 * bias + 0.002 * np.tanh(3.5 * envelope) * 0.16
        result[index] = np.tanh((float(sample) - bias) * drive) + 0.35 * bias
    return result


def render_dut(topology: str, value: np.ndarray, control: float) -> np.ndarray:
    """Render one hidden DUT without exposing its implementation to estimators."""

    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown hidden topology: {topology}")
    if not 0.0 <= control <= 1.0:
        raise ValueError(f"control outside [0,1]: {control}")
    drive = 0.9 + 3.0 * float(control)
    if topology == "A":
        first = _lowpass(value, 1_850.0)
        nonlinear = np.tanh(first * drive)
        result = _lowpass(nonlinear, 6_400.0)
    elif topology == "B":
        first = _highpass(value, 310.0)
        nonlinear = _asymmetric_clip(first, drive)
        result = _lowpass(nonlinear, 4_900.0)
    elif topology == "C":
        branch = _lowpass(value, 1_250.0)
        nonlinear = np.tanh(branch * drive)
        result = 0.42 * value + 0.78 * nonlinear
    elif topology == "D":
        first = _lowpass(value, 1_600.0)
        nonlinear = _dynamic_bias_clip(first, drive)
        result = _lowpass(nonlinear, 5_700.0)
    else:
        first = _asymmetric_clip(value, drive)
        middle = _highpass(first, 190.0)
        result = _lowpass(_asymmetric_clip(middle, 0.75 * drive), 3_700.0)
    return np.asarray((0.86 * result), dtype=np.float32)


def source_signal(source_id: int, frames: int = FRAMES) -> np.ndarray:
    """Create a deterministic, non-overlapping musical-like excitation."""

    rng = np.random.default_rng(0x534F5552 + source_id * 104_729)
    time = np.arange(frames, dtype=np.float64) / RATE
    base = 82.0 + 23.0 * (source_id % 7)
    frequencies = (base, base * 1.497, base * 2.031, base * 3.017)
    phases = rng.uniform(0.0, 2.0 * np.pi, len(frequencies))
    amplitudes = np.asarray((0.42, 0.25, 0.17, 0.10), dtype=np.float64)
    signal = sum(
        amplitude * np.sin(2.0 * np.pi * frequency * time + phase)
        for amplitude, frequency, phase in zip(amplitudes, frequencies, phases)
    )
    # Slow note-like envelope plus deterministic broadband detail prevents the
    # benchmark from being only a stationary sine-wave fit.
    envelope = 0.62 + 0.28 * np.sin(2.0 * np.pi * (0.7 + 0.03 * source_id) * time + phases[0])
    detail = rng.standard_normal(frames).astype(np.float64)
    detail = np.convolve(detail, np.ones(9) / 9.0, mode="same")
    signal = (signal * envelope) + 0.045 * detail
    peak = max(float(np.max(np.abs(signal))), 1.0e-6)
    return np.asarray(signal / peak, dtype=np.float32)


def _split_grid(split: str) -> tuple[Iterable[int], Iterable[float], Iterable[float]]:
    if split == "fit":
        return TRAIN_SOURCES, TRAIN_LEVELS, TRAIN_CONTROLS
    if split == "source_ood":
        return OOD_SOURCES, TRAIN_LEVELS, TRAIN_CONTROLS
    if split == "level_ood":
        return TRAIN_SOURCES, OOD_LEVELS, TRAIN_CONTROLS
    if split == "control_ood":
        return TRAIN_SOURCES, TRAIN_LEVELS, OOD_CONTROLS
    if split == "topology_ood":
        return OOD_SOURCES, TRAIN_LEVELS, TRAIN_CONTROLS
    raise ValueError(f"unknown benchmark split: {split}")


def make_observations(split: str, topologies: Iterable[str] = TOPOLOGIES) -> list[Observation]:
    sources, levels, controls = _split_grid(split)
    rows: list[Observation] = []
    for topology in topologies:
        for source_id in sources:
            clean = source_signal(source_id)
            for level in levels:
                x = np.asarray(clean * level, dtype=np.float32)
                for control in controls:
                    y = render_dut(topology, x, float(control))
                    rows.append(Observation(
                        x=x.copy(), y=y, control=float(control), topology=topology,
                        source_id=source_id, level=float(level), split=split,
                    ))
    return rows


def write_manifest(path: Path) -> dict:
    manifest = {
        "schema": 1,
        "kind": "black-box-forward-system-identification",
        "sample_rate": RATE,
        "frames": FRAMES,
        "topologies": list(TOPOLOGIES),
        "splits": ["fit", "source_ood", "level_ood", "control_ood", "topology_ood"],
        "fit_sources": list(TRAIN_SOURCES),
        "source_ood_sources": list(OOD_SOURCES),
        "fit_levels": list(TRAIN_LEVELS),
        "level_ood_levels": list(OOD_LEVELS),
        "fit_controls": list(TRAIN_CONTROLS),
        "control_ood_controls": list(OOD_CONTROLS),
        "topology_ood_contract": "fit only on DUT A; evaluate on hidden DUT D and E",
        "estimator_input": ["x", "y for fit only", "control"],
        "topology_hidden_from_estimators": True,
        "history_test": "same 4096-frame payload after quiet versus strong 1024-frame prefix",
        "dynamic_fit_augmentation": "state-bank estimators receive three fit-only prefix conditions per source",
        "external_audio": False,
        "license": "repository-owned synthetic benchmark; no external audio",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
