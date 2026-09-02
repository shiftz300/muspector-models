"""Profile-conditioned ambience inversion with bounded fallback and abstention."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.fft import irfft, next_fast_len, rfft

from .ambience2 import (
    AmbienceAbstention,
    AmbiencePairsV2,
    restore_known_ambience_profile,
)
from .product_data import _rir, rir_splits
from .reverb_quality import envelope


MAXIMUM_REGULARIZED_GAIN = 6.0


def _transfer(impulse: np.ndarray, controls: dict) -> np.ndarray:
    mix = float(controls["mix"])
    room_gain_db = float(controls["room_gain_db"])
    if not 0.18 <= mix <= 0.62 or not -12.0 <= room_gain_db <= -4.0:
        raise ValueError("ambience3 controls are outside the admitted domain")
    transfer = np.asarray(impulse, dtype=np.float64) * (
        mix * 10.0 ** (room_gain_db / 20.0)
    )
    transfer = transfer.copy()
    transfer[0] += 1.0 - mix
    return transfer


def regularized_profile_inverse(
    wet: np.ndarray,
    impulse: np.ndarray,
    controls: dict,
    *,
    maximum_gain: float = MAXIMUM_REGULARIZED_GAIN,
) -> tuple[np.ndarray, dict]:
    """Noncausal bounded inverse for profiles rejected by the exact causal path."""
    wet = np.asarray(wet, dtype=np.float64)
    impulse = np.asarray(impulse, dtype=np.float64)
    if wet.ndim != 1 or impulse.ndim != 1 or not len(wet) or not len(impulse):
        raise ValueError("ambience3 regularized inverse expects nonempty mono arrays")
    if not np.isfinite(wet).all() or not np.isfinite(impulse).all() or maximum_gain <= 1.0:
        raise ValueError("ambience3 regularized inverse received invalid input")
    transfer = _transfer(impulse, controls)
    fft_size = next_fast_len(len(wet) + len(transfer) - 1)
    response = rfft(transfer, fft_size)
    magnitude = np.abs(response)
    phase_inverse = np.conj(response) / np.maximum(magnitude, 1.0e-12)
    inverse = phase_inverse / np.maximum(magnitude, 1.0 / maximum_gain)
    restored = irfft(rfft(wet, fft_size) * inverse, fft_size)[: len(wet)]
    if not np.isfinite(restored).all() or float(np.max(np.abs(restored))) > 1.05:
        raise AmbienceAbstention("ambience3 bounded profile inverse is non-finite or clips")
    return restored.astype(np.float32), {
        "implementation": "bounded-frequency-profile-inverse",
        "maximum_frequency_gain": float(np.max(np.abs(inverse))),
        "maximum_gain_contract": maximum_gain,
        "profile_required": True,
        "causal": False,
        "graph_order_input": False,
        "neighbor_effect_input": False,
    }


def profile_inverse_base(
    wet: np.ndarray,
    impulse: np.ndarray,
    controls: dict,
) -> tuple[np.ndarray, dict]:
    """Keep the exact stable path; use bounded inversion only after abstention."""
    try:
        restored, report = restore_known_ambience_profile(wet, impulse, controls)
        return restored, {**report, "mode": "exact", "fallback": False}
    except AmbienceAbstention as error:
        restored, report = regularized_profile_inverse(wet, impulse, controls)
        return restored, {
            **report,
            "mode": "regularized-fallback",
            "fallback": True,
            "exact_abstention": str(error),
        }


def wet_candidate_tail_ratio(wet: np.ndarray, candidate: np.ndarray, target_start: int) -> float:
    """Observable post-activity safety score requiring no hidden Clean target."""
    wet_env = envelope(np.asarray(wet, dtype=np.float32)[target_start:])
    candidate_env = envelope(np.asarray(candidate, dtype=np.float32)[target_start:])
    length = min(len(wet_env), len(candidate_env))
    wet_env, candidate_env = wet_env[:length], candidate_env[:length]
    active = wet_env > np.quantile(wet_env, 0.75)
    recent = np.convolve(active.astype(np.float64), np.ones(40), mode="full")[:length] > 0
    quiet = wet_env <= np.quantile(wet_env, 0.35)
    mask = quiet & recent
    if int(mask.sum()) < 4:
        return 0.0
    wet_level = float(np.mean(wet_env[mask]))
    candidate_level = float(np.mean(candidate_env[mask]))
    return candidate_level / max(wet_level, 1.0e-8)


class AmbiencePairsV3(torch.utils.data.Dataset):
    """Ambience2 pairs enriched with a rights-safe exact/regularized profile base."""

    def __init__(self, workspace: Path, split: str, samples: int, target_frames: int, seed: int) -> None:
        self.base = AmbiencePairsV2(
            workspace,
            split,
            samples,
            target_frames,
            seed,
            include_late_base=False,
        )
        self.workspace = self.base.workspace
        self.split = split
        self.samples = samples
        self.target_frames = target_frames
        self.history_frames = self.base.history_frames
        self.total_frames = self.base.total_frames
        self.authorization = self.base.authorization
        self.rir_paths = {path.name: path for path in rir_splits(self.workspace)[split]}
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.base)

    def realized_source_counts(self) -> dict[str, int]:
        return self.base.realized_source_counts()

    def realized_rir_counts(self) -> dict[str, int]:
        return self.base.realized_rir_counts()

    def __getitem__(self, index: int) -> dict:
        if index in self._cache:
            return self._cache[index]
        row = dict(self.base[index])
        impulse = _rir(self.rir_paths[row["rir"]])
        restored, profile = profile_inverse_base(
            row["wet"].numpy(),
            impulse,
            row["control_values"],
        )
        row["profile_base"] = torch.from_numpy(restored.copy())
        row["profile_fallback"] = bool(profile["fallback"])
        row["profile_report"] = profile
        self._cache[index] = row
        return row
