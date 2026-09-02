"""Product-safe long-tail ambience pairs with split-disjoint measured RIRs."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import fftconvolve

from .foundation_data import RATE
from .license_gate import require_product_weights
from .product2 import _condition_clean
from .product_data import Clean, RESEARCH_SOURCE_IDS, _read, _rir, discover_clean, rir_splits


HISTORY_FRAMES = round(2.25 * RATE)
TARGET_FRAMES_MINIMUM = 16_384
CONTROL_WIDTH = 3


class AmbienceAbstention(RuntimeError):
    """The current single-effect ambience profile has no stable inverse."""


def blind_late_suppression(
    wet: torch.Tensor,
    *,
    n_fft: int = 1024,
    hop: int = 256,
    delay: int = 3,
    taps: int = 6,
    strength: float = 0.25,
) -> torch.Tensor:
    """Single-channel delayed linear prediction used only inside ambience."""
    if wet.ndim != 2 or wet.shape[1] < n_fft * 2 or not torch.isfinite(wet).all():
        raise ValueError("late suppression expects finite [batch,time] ambience audio")
    if not 0.0 <= strength <= 1.0 or delay < 1 or taps < 2:
        raise ValueError("invalid late suppression geometry")
    window = torch.hann_window(n_fft, device=wet.device, dtype=wet.dtype)
    rows = []
    for audio in wet:
        spectrum = torch.stft(
            audio[None], n_fft, hop, window=window, center=True,
            pad_mode="constant", return_complex=True,
        )[0]
        start = delay + taps - 1
        predictors = torch.stack(
            [spectrum[:, start - delay - index : spectrum.shape[1] - delay - index] for index in range(taps)],
            dim=1,
        )
        target = spectrum[:, start:]
        covariance = predictors @ predictors.conj().transpose(1, 2)
        diagonal = covariance.diagonal(dim1=1, dim2=2).real.mean(1).clamp_min(1.0e-8)
        identity = torch.eye(taps, device=wet.device, dtype=covariance.dtype)[None]
        coefficients = torch.linalg.solve(
            covariance + identity * diagonal[:, None, None] * 1.0e-3,
            predictors @ target.conj().unsqueeze(-1),
        )
        predicted = (coefficients.conj().transpose(1, 2) @ predictors).squeeze(1)
        corrected = spectrum.clone()
        corrected[:, start:] = target - strength * predicted
        rows.append(torch.istft(
            corrected[None], n_fft, hop, window=window, center=True, length=audio.shape[0]
        )[0])
    return torch.stack(rows)


def restore_known_ambience_profile(
    wet: np.ndarray,
    impulse: np.ndarray,
    control_values: dict,
    *,
    maximum_inverse_coefficient: float = 10.0,
) -> tuple[np.ndarray, dict]:
    """Exact causal power-series inverse with a hard noise-amplification veto."""
    wet = np.asarray(wet, dtype=np.float64)
    impulse = np.asarray(impulse, dtype=np.float64)
    if wet.ndim != 1 or impulse.ndim != 1 or not len(wet) or not len(impulse):
        raise ValueError("known ambience inverse expects nonempty mono arrays")
    if not np.isfinite(wet).all() or not np.isfinite(impulse).all():
        raise ValueError("known ambience inverse expects finite audio")
    mix = float(control_values["mix"])
    room_gain_db = float(control_values["room_gain_db"])
    if not 0.18 <= mix <= 0.62 or not -12.0 <= room_gain_db <= -4.0:
        raise ValueError("known ambience controls are outside the admitted domain")
    transfer = impulse * (mix * 10.0 ** (room_gain_db / 20.0))
    transfer = transfer.copy()
    transfer[0] += 1.0 - mix
    if abs(transfer[0]) < 1.0e-6:
        raise AmbienceAbstention("ambience direct path is not invertible")
    inverse = np.asarray((1.0 / transfer[0],), dtype=np.float64)
    while len(inverse) < len(wet):
        length = min(len(wet), len(inverse) * 2)
        closure = fftconvolve(transfer[:length], inverse, mode="full")[:length]
        correction = -closure
        correction[0] += 2.0
        inverse = fftconvolve(inverse, correction, mode="full")[:length]
        peak = float(np.max(np.abs(inverse)))
        if not math.isfinite(peak) or peak > maximum_inverse_coefficient:
            raise AmbienceAbstention(
                f"ambience inverse noise gain {peak:.6g} exceeds {maximum_inverse_coefficient:.6g}"
            )
    restored = fftconvolve(wet, inverse, mode="full")[: len(wet)]
    if not np.isfinite(restored).all() or float(np.max(np.abs(restored))) > 1.05:
        raise AmbienceAbstention("ambience inverse output is unstable or clips")
    return restored.astype(np.float32), {
        "implementation": "known-profile-causal-power-series-inverse",
        "maximum_inverse_coefficient": float(np.max(np.abs(inverse))),
        "profile_required": True,
        "chain_order_input": False,
        "neighbor_effect_input": False,
    }


def _rir_decay_seconds(impulse: np.ndarray, fraction: float = 0.999) -> float:
    energy = np.cumsum(np.square(np.asarray(impulse, dtype=np.float64)))
    if not len(energy) or energy[-1] <= 0.0:
        raise ValueError("ambience2 RIR has no energy")
    index = int(np.searchsorted(energy, fraction * energy[-1]))
    return (index + 1) / RATE


def _render(clean: np.ndarray, impulse: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    mix = rng.uniform(0.18, 0.62)
    room_gain_db = rng.uniform(-12.0, -4.0)
    room_gain = 10.0 ** (room_gain_db / 20.0)
    diffuse = fftconvolve(
        np.asarray(clean, dtype=np.float64),
        np.asarray(impulse, dtype=np.float64) * room_gain,
        mode="full",
    )[: len(clean)]
    wet = (1.0 - mix) * clean + mix * diffuse
    controls = {
        "mix": mix,
        "room_gain_db": room_gain_db,
        "decay_p999_seconds": _rir_decay_seconds(impulse),
    }
    return wet.astype(np.float32), controls


def _controls(values: dict) -> np.ndarray:
    result = np.asarray(
        (
            (values["mix"] - 0.18) / (0.62 - 0.18),
            (values["room_gain_db"] + 12.0) / 8.0,
            (values["decay_p999_seconds"] - 1.25) / (2.30 - 1.25),
        ),
        dtype=np.float32,
    )
    if not np.isfinite(result).all() or np.any(result < -1.0e-6) or np.any(result > 1.0 + 1.0e-6):
        raise ValueError(f"ambience2 controls are outside the admitted domain: {result}")
    return np.clip(result, 0.0, 1.0)


class AmbiencePairsV2(torch.utils.data.Dataset):
    """A 2.25-second real prehistory precedes every supervised dereverb tail."""

    def __init__(
        self,
        workspace: Path,
        split: str,
        samples: int,
        target_frames: int,
        seed: int,
        include_late_base: bool = True,
    ) -> None:
        if split not in {"fit", "calibration", "development", "locked-final"}:
            raise ValueError(f"unsupported ambience2 split: {split}")
        if samples < 1 or target_frames < TARGET_FRAMES_MINIMUM:
            raise ValueError("ambience2 request is too small")
        self.workspace = workspace.resolve()
        self.split = split
        self.samples = samples
        self.target_frames = target_frames
        self.history_frames = HISTORY_FRAMES
        self.total_frames = HISTORY_FRAMES + target_frames
        self.seed = seed
        self.include_late_base = include_late_base
        buckets: dict[str, list[Clean]] = defaultdict(list)
        for item in discover_clean(self.workspace):
            if item.split == split:
                buckets[item.source_id].append(item)
        if not buckets:
            raise ValueError(f"no ambience2 clean programs for {split}")
        self.buckets = {name: tuple(rows) for name, rows in sorted(buckets.items())}
        self.source_ids = tuple(self.buckets)
        self.rirs = tuple(rir_splits(self.workspace)[split])
        if not self.rirs:
            raise ValueError(f"no ambience2 RIRs for {split}")
        realized = set(self.realized_source_counts()) | {"aachen-chapel-rir", "muspector-dsp"}
        if realized & RESEARCH_SOURCE_IDS:
            raise PermissionError(f"research source entered ambience2: {realized & RESEARCH_SOURCE_IDS}")
        self.authorization = require_product_weights(
            self.workspace / "remix/data_sources.json", sorted(realized)
        )
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return self.samples

    def _selection(self, index: int) -> Clean:
        source = self.source_ids[index % len(self.source_ids)]
        rows = self.buckets[source]
        cycle = index // len(self.source_ids)
        return rows[(cycle * 104729 + self.seed) % len(rows)]

    def realized_source_counts(self) -> dict[str, int]:
        return dict(Counter(self._selection(index).source_id for index in range(self.samples)))

    def realized_rir_counts(self) -> dict[str, int]:
        return dict(Counter(self.rirs[index % len(self.rirs)].name for index in range(self.samples)))

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.samples:
            raise IndexError(index)
        if index in self._cache:
            return self._cache[index]
        rng = random.Random(self.seed + index * 104729)
        selected = self._selection(index)
        clean = _read(selected, self.total_frames, rng.getrandbits(63))
        clean, input_gain_db = _condition_clean(clean, rng)
        rir_path = self.rirs[index % len(self.rirs)]
        impulse = _rir(rir_path)
        wet, control_values = _render(clean, impulse, rng)
        target_wet = wet[self.history_frames :]
        target_clean = clean[self.history_frames :]
        if wet.shape != clean.shape or not np.isfinite(wet).all():
            raise ValueError("ambience2 renderer changed geometry or emitted non-finite audio")
        distance = math.sqrt(float(np.mean(np.square(target_wet - target_clean))) + 1.0e-12)
        clean_rms = math.sqrt(float(np.mean(np.square(target_clean))) + 1.0e-12)
        if distance < 0.03 * clean_rms:
            raise ValueError("ambience2 pair has no meaningful supervised reverberation")
        result = {
            "wet": torch.from_numpy(wet.copy()),
            "clean": torch.from_numpy(clean.copy()),
            "controls": torch.from_numpy(_controls(control_values)),
            "control_values": control_values,
            "target_start": self.history_frames,
            "source_id": selected.source_id,
            "group": selected.group,
            "rir": rir_path.name,
            "input_gain_db": input_gain_db,
        }
        if self.include_late_base:
            with torch.inference_mode():
                result["late_base"] = blind_late_suppression(result["wet"].unsqueeze(0))[0]
        self._cache[index] = result
        return result
