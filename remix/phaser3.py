"""Safe-subdomain Stone-style Phaser inverse with Wet-only phase confidence."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly, stft

from .chorus3 import ChorusPairsV3
from .license_gate import require_product_weights


RATE = 48_000
SEARCH_RATE = 12_000
PHASE_CANDIDATES = 64
CONTROL_WIDTH = 5
RATE_MIN, RATE_MAX = 0.7, 2.4
MIX_MIN, MIX_MAX = 0.32, 0.50
DEFAULT_MARGIN_THRESHOLD = 0.04
UPSTREAM_URL = "https://github.com/jpcima/stone-phaser"
UPSTREAM_COMMIT = "f057ea5c1e368d6cd937f4e359a9514ac72fe902"
UPSTREAM_LICENSE = "CC0-1.0-or-BSL-1.0"


def _analog_triangle(position: np.ndarray) -> np.ndarray:
    fraction = np.remainder(position, 1.0)
    roundness = 0.975
    folded = np.where(fraction < 0.5, fraction, 1.0 - fraction)
    return 1.0 - np.sin(2.0 * folded * np.arcsin(roundness)) / roundness


def _coefficient_grid(
    rate_hz: float, phases: np.ndarray, color: bool, sample_rate: int, frames: int
) -> np.ndarray:
    time = np.arange(frames, dtype=np.float64) / sample_rate
    position = rate_hz * time[None, :] + phases[:, None] / (2.0 * np.pi)
    shape = _analog_triangle(position)
    low, high = ((80.0, 2200.0) if color else (300.0, 6000.0))
    midi_low = 69.0 + 12.0 * math.log2(low / 440.0)
    midi_high = 69.0 + 12.0 * math.log2(high / 440.0)
    frequency = 440.0 * np.power(
        2.0, (midi_low + shape * (midi_high - midi_low) - 69.0) / 12.0
    )
    return np.clip(-1.0 + 2.0 * np.pi * frequency / sample_rate, -0.9995, -0.05)


def _mix_coefficients(mix: float) -> tuple[float, float]:
    return math.cos(mix * math.pi / 2.0), math.sin(mix * math.pi / 2.0)


def render_phaser(clean: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict, np.ndarray]:
    values = {
        "family": "phaser",
        "rate_hz": math.exp(rng.uniform(math.log(RATE_MIN), math.log(RATE_MAX))),
        "mix": rng.uniform(MIX_MIN, MIX_MAX),
        "color": bool(rng.randrange(2)),
        "feedback": 0.0,
        "hidden_phase_radians": rng.uniform(0.0, 2.0 * math.pi),
        "renderer": "muspector-stone-style-four-stage-allpass-v1",
    }
    clean = np.asarray(clean, dtype=np.float64)
    d, w = _mix_coefficients(values["mix"])
    pole = math.exp(-2.0 * math.pi * 33.0 / RATE)
    feedforward = 0.5 * (1.0 + pole)
    coefficients = _coefficient_grid(
        values["rate_hz"], np.array([values["hidden_phase_radians"]]),
        values["color"], RATE, len(clean)
    )[0]
    input_previous = highpass_previous = 0.0
    stage_input = np.zeros(4, dtype=np.float64)
    stage_output = np.zeros(4, dtype=np.float64)
    wet = np.empty(len(clean), dtype=np.float32)
    for index, sample in enumerate(clean):
        highpass = feedforward * (sample - input_previous) + pole * highpass_previous
        input_previous, highpass_previous = sample, highpass
        coefficient = float(coefficients[index])
        value = highpass
        for stage in range(4):
            output = coefficient * value + stage_input[stage] - coefficient * stage_output[stage]
            stage_input[stage], stage_output[stage] = value, output
            value = output
        wet[index] = d * sample + w * value
    time = np.arange(len(clean), dtype=np.float64) / RATE
    lfo = _analog_triangle(
        values["rate_hz"] * time + values["hidden_phase_radians"] / (2.0 * np.pi)
    )
    return wet, values, lfo.astype(np.float32)


def normalized_controls(values: dict) -> np.ndarray:
    result = np.zeros(CONTROL_WIDTH, dtype=np.float32)
    result[0] = math.log(values["rate_hz"] / RATE_MIN) / math.log(RATE_MAX / RATE_MIN)
    result[1] = (values["mix"] - MIX_MIN) / (MIX_MAX - MIX_MIN)
    result[2] = float(values["color"])
    return np.clip(result, 0.0, 1.0)


def _invert_candidates(
    wet: np.ndarray, values: dict, phases: np.ndarray, sample_rate: int
) -> np.ndarray:
    wet = np.asarray(wet, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64)
    d, w = _mix_coefficients(float(values["mix"]))
    pole = math.exp(-2.0 * math.pi * 33.0 / sample_rate)
    feedforward = 0.5 * (1.0 + pole)
    coefficients = _coefficient_grid(
        float(values["rate_hz"]), phases, bool(values["color"]), sample_rate, len(wet)
    )
    count = len(phases)
    input_previous = np.zeros(count, dtype=np.float64)
    highpass_previous = np.zeros(count, dtype=np.float64)
    stage_input = np.zeros((4, count), dtype=np.float64)
    stage_output = np.zeros((4, count), dtype=np.float64)
    restored = np.empty((count, len(wet)), dtype=np.float32)
    for index, observed in enumerate(wet):
        coefficient = coefficients[:, index]
        highpass_constant = -feedforward * input_previous + pole * highpass_previous
        direct = np.full(count, feedforward, dtype=np.float64)
        constant = highpass_constant
        for stage in range(4):
            constant = coefficient * constant + stage_input[stage] - coefficient * stage_output[stage]
            direct *= coefficient
        clean = (observed - w * constant) / (d + w * direct)
        highpass = feedforward * clean + highpass_constant
        value = highpass
        for stage in range(4):
            output = coefficient * value + stage_input[stage] - coefficient * stage_output[stage]
            stage_input[stage], stage_output[stage] = value, output
            value = output
        input_previous, highpass_previous = clean, highpass
        restored[:, index] = clean
    return restored


def invert_known_phase(wet: np.ndarray, values: dict, phase: float) -> np.ndarray:
    wet = np.asarray(wet, dtype=np.float64)
    d, w = _mix_coefficients(float(values["mix"]))
    pole = math.exp(-2.0 * math.pi * 33.0 / RATE)
    feedforward = 0.5 * (1.0 + pole)
    coefficients = _coefficient_grid(
        float(values["rate_hz"]), np.array([phase]), bool(values["color"]), RATE, len(wet)
    )[0]
    input_previous = highpass_previous = 0.0
    stage_input = np.zeros(4, dtype=np.float64)
    stage_output = np.zeros(4, dtype=np.float64)
    restored = np.empty(len(wet), dtype=np.float32)
    for index, observed in enumerate(wet):
        coefficient = float(coefficients[index])
        highpass_constant = -feedforward * input_previous + pole * highpass_previous
        direct = feedforward
        constant = highpass_constant
        for stage in range(4):
            constant = coefficient * constant + stage_input[stage] - coefficient * stage_output[stage]
            direct *= coefficient
        clean = (observed - w * constant) / (d + w * direct)
        highpass = feedforward * clean + highpass_constant
        value = highpass
        for stage in range(4):
            output = coefficient * value + stage_input[stage] - coefficient * stage_output[stage]
            stage_input[stage], stage_output[stage] = value, output
            value = output
        input_previous, highpass_previous = clean, highpass
        restored[index] = clean
    return restored


def _periodic_scores(
    candidates: np.ndarray, rate_hz: float, sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    _, times, spectrum = stft(
        candidates, fs=sample_rate, window="hann", nperseg=1024, noverlap=768,
        nfft=1024, boundary=None, padded=False, axis=1
    )
    log_magnitude = np.log(np.abs(spectrum[:, 6:343, :]) + 1.0e-6)
    log_magnitude -= log_magnitude.mean(axis=1, keepdims=True)
    normalized_time = np.linspace(-1.0, 1.0, log_magnitude.shape[-1])
    design = np.stack((np.ones_like(normalized_time), normalized_time, normalized_time**2), axis=1)
    basis, _ = np.linalg.qr(design)
    log_magnitude -= np.einsum(
        "cft,tk,sk->cfs", log_magnitude, basis, basis, optimize=True
    )
    energy = np.sum(log_magnitude * log_magnitude, axis=(1, 2)) + 1.0e-12
    frequency_score = np.zeros(log_magnitude.shape[:2], dtype=np.float64)
    for harmonic in (1.0, 2.0, 3.0, 4.0):
        angle = 2.0 * np.pi * harmonic * rate_hz * times
        cosine, sine = np.cos(angle), np.sin(angle)
        projected_cosine = np.einsum("cft,t->cf", log_magnitude, cosine, optimize=True)
        projected_sine = np.einsum("cft,t->cf", log_magnitude, sine, optimize=True)
        frequency_score += projected_cosine**2 + projected_sine**2
    score = np.sum(frequency_score, axis=1) / energy
    boundaries = (0, 50, 120, 220, frequency_score.shape[1])
    band_scores = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        band_energy = np.sum(log_magnitude[:, left:right] ** 2, axis=(1, 2)) + 1.0e-12
        band_scores.append(np.sum(frequency_score[:, left:right], axis=1) / band_energy)
    return score, np.stack(band_scores, axis=1)


def estimate_phase_diagnostics(wet: np.ndarray, values: dict) -> dict:
    reduced = resample_poly(np.asarray(wet, dtype=np.float32), SEARCH_RATE, RATE).astype(np.float32)
    phases = np.arange(PHASE_CANDIDATES, dtype=np.float64) * (2.0 * np.pi / PHASE_CANDIDATES)
    candidates = _invert_candidates(reduced, values, phases, SEARCH_RATE)
    scores, band_scores = _periodic_scores(candidates, float(values["rate_hz"]), SEARCH_RATE)
    selected_index = int(np.argmin(scores))
    selected = float(phases[selected_index])
    distance = np.abs((phases - selected + np.pi) % (2.0 * np.pi) - np.pi)
    distant = scores[distance >= 0.35]
    margin = float((np.min(distant) - scores[selected_index]) / max(scores[selected_index], 1.0e-12))
    band_phases = phases[np.argmin(band_scores, axis=0)]
    band_distances = np.abs((band_phases - selected + np.pi) % (2.0 * np.pi) - np.pi)
    width = 2.0 * np.pi / PHASE_CANDIDATES
    refined_phases = np.remainder(selected + np.linspace(-width, width, 17), 2.0 * np.pi)
    refined = _invert_candidates(reduced, values, refined_phases, SEARCH_RATE)
    refined_scores, _ = _periodic_scores(refined, float(values["rate_hz"]), SEARCH_RATE)
    refined_index = int(np.argmin(refined_scores))
    return {
        "phase": float(refined_phases[refined_index]),
        "score": float(refined_scores[refined_index]),
        "margin": margin,
        "band_phase_distances": [float(value) for value in band_distances],
        "band_consensus_median_radians": float(np.median(band_distances)),
        "band_consensus_p90_radians": float(np.quantile(band_distances, 0.90)),
    }


def estimate_phase(wet: np.ndarray, values: dict) -> tuple[float, float, float]:
    diagnostics = estimate_phase_diagnostics(wet, values)
    return diagnostics["phase"], diagnostics["score"], diagnostics["margin"]


def restore_phaser(
    wet: np.ndarray, values: dict, margin_threshold: float = DEFAULT_MARGIN_THRESHOLD
) -> tuple[np.ndarray, bool, float, float, np.ndarray]:
    phase, score, margin = estimate_phase(wet, values)
    if margin < margin_threshold:
        return np.asarray(wet, dtype=np.float32).copy(), False, phase, margin, np.zeros(len(wet), dtype=np.float32)
    restored = invert_known_phase(wet, values, phase)
    time = np.arange(len(wet), dtype=np.float64) / RATE
    lfo = _analog_triangle(float(values["rate_hz"]) * time + phase / (2.0 * np.pi))
    return restored, True, phase, margin, lfo.astype(np.float32)


class StonePhaserPairsV3(torch.utils.data.Dataset):
    def __init__(self, workspace: Path, split: str, samples: int, seed: int) -> None:
        self.base = ChorusPairsV3(workspace, split, samples, 96_000, seed)
        self.workspace, self.split, self.samples, self.seed = workspace.resolve(), split, samples, seed
        source_ids = sorted(set(self.base.authorization["sources"]) | {"stone-phaser-cc0"})
        self.authorization = require_product_weights(
            self.workspace / "remix/data_sources.json", source_ids
        )

    def __len__(self) -> int:
        return self.samples

    def realized_source_counts(self) -> dict:
        return self.base.realized_source_counts()

    def __getitem__(self, index: int) -> dict:
        source = self.base[index]
        start = int(source["target_start"])
        clean = source["clean"].numpy()[start : start + 96_000]
        wet, values, lfo = render_phaser(clean, random.Random(self.seed + index * 104729))
        return {
            "wet": torch.from_numpy(wet),
            "clean": torch.from_numpy(clean.copy()),
            "controls": torch.from_numpy(normalized_controls(values)),
            "lfo": torch.from_numpy(lfo),
            "control_values": values,
            "target_start": 0,
            "target_end": len(clean),
            "source_id": source["source_id"],
            "group": source["group"],
        }


def manifest(margin_threshold: float) -> dict:
    return {
        "schema": 1,
        "architecture": "wet-analysis-by-synthesis-phase-search-plus-four-stage-allpass-inverse",
        "mechanism": "modulation",
        "family": "phaser",
        "parameters": 0,
        "controls": ["rate_hz", "mix", "color", "feedback_zero_safe_subdomain"],
        "rate_domain_hz": [RATE_MIN, RATE_MAX],
        "mix_domain": [MIX_MIN, MIX_MAX],
        "feedback_domain": [0.0, 0.0],
        "phase_candidates": PHASE_CANDIDATES,
        "confidence_margin_threshold": margin_threshold,
        "ambiguous_input_behavior": "abstain-and-pass-through",
        "hidden_forward_state": "lfo_start_phase",
        "hidden_forward_state_is_inference_input": False,
        "graph_order_input": False,
        "neighbouring_effect_input": False,
        "clean_or_oracle_input": False,
        "physical_device_claim": False,
        "bounded_context_frames": 96_000,
        "causal_inverse_after_noncausal_bounded_phase_estimation": True,
        "renderer_provenance": {
            "upstream_url": UPSTREAM_URL,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_license": UPSTREAM_LICENSE,
            "scope": "CC0 Stone-style topology without feedback; generic mechanism only",
        },
    }
