"""Capture-fitted hybrid forward renderer for long-memory Reverb."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.signal import fftconvolve

from .forward_drive import FORWARD_RATE
from .pedalboard_renderer import MIN_REVERB_DECAY_SECONDS, render_pedalboard_chain
from .quality import validate_render
from .render import render_chain
from .spec import ChainSpec, Reverb


ReverbForwardDomain = Literal[
    "reference", "alternate", "stress", "challenge", "final_challenge", "pedalboard"
]
PROFILE_SECONDS = 8
PROFILE_DECAYS = np.geomspace(0.2, 8.0, 9).astype(np.float64)
PROFILE_DAMPING = np.linspace(0.0, 1.0, 5, dtype=np.float64)
PROFILE_RANK = 8


def _render_domain(
    audio: np.ndarray,
    effect: Reverb,
    sample_rate: int,
    domain: ReverbForwardDomain,
) -> np.ndarray:
    spec = ChainSpec((effect,))
    if domain == "pedalboard":
        return render_pedalboard_chain(audio, spec, sample_rate)
    return render_chain(audio, spec, sample_rate, domain)


def _wet_impulse(
    domain: ReverbForwardDomain,
    decay_s: float,
    damping: float,
    sample_rate: int,
    frames: int,
) -> np.ndarray:
    dry = np.zeros(frames, dtype=np.float32)
    dry[0] = 1.0
    mix = 0.7
    rendered = _render_domain(dry, Reverb(decay_s, damping, mix), sample_rate, domain)
    return np.asarray((rendered - dry * (1.0 - mix)) / mix, dtype=np.float32)


@dataclass(frozen=True)
class ReverbDeviceProfile:
    sample_rate: int
    mean_ir: np.ndarray
    basis_ir: np.ndarray
    decay_grid: np.ndarray
    damping_grid: np.ndarray
    coefficient_grid: np.ndarray
    source_domain: str
    calibration_hash: str

    def validate(self) -> None:
        frames = self.sample_rate * PROFILE_SECONDS
        if self.mean_ir.shape != (frames,) or self.basis_ir.ndim != 2:
            raise ValueError("Reverb profile impulse geometry is invalid")
        expected = (len(self.decay_grid), len(self.damping_grid), len(self.basis_ir))
        if self.basis_ir.shape[1] != frames or self.coefficient_grid.shape != expected:
            raise ValueError("Reverb profile coefficient geometry is invalid")
        for value in (
            self.mean_ir,
            self.basis_ir,
            self.decay_grid,
            self.damping_grid,
            self.coefficient_grid,
        ):
            if not np.isfinite(value).all():
                raise ValueError("Reverb profile contains non-finite values")

    def impulse(self, decay_s: float, damping: float, frames: int | None = None) -> np.ndarray:
        Reverb(decay_s, damping, 0.0).validate()
        self.validate()
        length = len(self.mean_ir) if frames is None else min(frames, len(self.mean_ir))
        log_grid = np.log(self.decay_grid)
        coefficients = []
        for component in range(len(self.basis_ir)):
            at_damping = [
                float(
                    np.interp(
                        math.log(decay_s),
                        log_grid,
                        self.coefficient_grid[:, damping_index, component],
                    )
                )
                for damping_index in range(len(self.damping_grid))
            ]
            coefficients.append(float(np.interp(damping, self.damping_grid, at_damping)))
        value = self.mean_ir[:length].astype(np.float64) + np.asarray(
            coefficients, dtype=np.float64
        ) @ self.basis_ir[:, :length].astype(np.float64)
        return value.astype(np.float32)

    def render(self, dry: np.ndarray, effect: Reverb) -> np.ndarray:
        effect.validate()
        source = np.asarray(dry, dtype=np.float32)
        if source.ndim != 1 or not np.isfinite(source).all():
            raise ValueError("Reverb forward input must be finite mono audio")
        if effect.mix == 0.0:
            return source.copy()
        impulse = self.impulse(effect.decay_s, effect.damping)
        wet = fftconvolve(source, impulse, mode="full")[: len(source)].astype(np.float32)
        result = np.asarray(source * (1.0 - effect.mix) + wet * effect.mix, dtype=np.float32)
        validate_render(source, result)
        return result


def fit_reverb_profile(
    domain: ReverbForwardDomain,
    sample_rate: int = FORWARD_RATE,
) -> ReverbDeviceProfile:
    """Fit a device fingerprint from an in-memory impulse calibration grid."""

    frames = sample_rate * PROFILE_SECONDS
    minimum = MIN_REVERB_DECAY_SECONDS if domain == "pedalboard" else 0.2
    decay_grid = PROFILE_DECAYS[PROFILE_DECAYS >= minimum].copy()
    if decay_grid[0] > minimum + 1.0e-9:
        decay_grid = np.insert(decay_grid, 0, minimum)
    captures = []
    digest = hashlib.sha256()
    for decay_s in decay_grid:
        for damping in PROFILE_DAMPING:
            impulse = _wet_impulse(domain, float(decay_s), damping, sample_rate, frames)
            digest.update(impulse.tobytes())
            captures.append(impulse)
    matrix = np.stack(captures).astype(np.float32)
    mean_ir = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = matrix - mean_ir
    gram = centered @ centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram.astype(np.float64))
    order = np.argsort(eigenvalues)[::-1]
    keep = order[: min(PROFILE_RANK, len(matrix) - 1)]
    scales = np.sqrt(np.maximum(eigenvalues[keep], 1.0e-20))
    basis = ((eigenvectors[:, keep].T @ centered) / scales[:, None]).astype(np.float32)
    coefficients = (centered @ basis.T).reshape(
        len(decay_grid), len(PROFILE_DAMPING), len(basis)
    )
    profile = ReverbDeviceProfile(
        sample_rate,
        mean_ir,
        basis,
        decay_grid,
        PROFILE_DAMPING.copy(),
        coefficients.astype(np.float32),
        domain,
        digest.hexdigest(),
    )
    profile.validate()
    return profile


def save_reverb_profile(profile: ReverbDeviceProfile, path: Path) -> str:
    profile.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray(1, dtype=np.int64),
        sample_rate=np.asarray(profile.sample_rate, dtype=np.int64),
        mean_ir=profile.mean_ir,
        basis_ir=profile.basis_ir,
        decay_grid=profile.decay_grid,
        damping_grid=profile.damping_grid,
        coefficient_grid=profile.coefficient_grid,
        source_domain=np.asarray(profile.source_domain),
        calibration_hash=np.asarray(profile.calibration_hash),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reverb_profile(path: Path) -> ReverbDeviceProfile:
    with np.load(path, allow_pickle=False) as payload:
        if int(payload["schema"]) != 1:
            raise ValueError("unsupported Reverb profile schema")
        profile = ReverbDeviceProfile(
            int(payload["sample_rate"]),
            payload["mean_ir"].astype(np.float32),
            payload["basis_ir"].astype(np.float32),
            payload["decay_grid"].astype(np.float64),
            payload["damping_grid"].astype(np.float64),
            payload["coefficient_grid"].astype(np.float32),
            str(payload["source_domain"]),
            str(payload["calibration_hash"]),
        )
    profile.validate()
    return profile


@dataclass(frozen=True)
class ReverbStreamState:
    dry_history: np.ndarray
    effect: Reverb


def stream_reverb_block(
    profile: ReverbDeviceProfile,
    dry: np.ndarray,
    effect: Reverb,
    state: ReverbStreamState | None = None,
) -> tuple[np.ndarray, ReverbStreamState]:
    """Reference streaming convolution; native runtime should partition the FIR."""

    source = np.asarray(dry, dtype=np.float32)
    if source.ndim != 1 or not np.isfinite(source).all():
        raise ValueError("streaming Reverb expects finite mono audio")
    effect.validate()
    if state is None:
        history = np.empty(0, dtype=np.float32)
    else:
        if state.effect != effect:
            raise ValueError("Reverb controls cannot change while reusing stream state")
        history = state.dry_history
    if effect.mix == 0.0:
        rendered = source.copy()
    else:
        impulse = profile.impulse(effect.decay_s, effect.damping)
        context = np.concatenate((history, source))
        wet = fftconvolve(context, impulse, mode="full")
        start = len(history)
        rendered = np.asarray(
            source * (1.0 - effect.mix) + wet[start : start + len(source)] * effect.mix,
            dtype=np.float32,
        )
    keep = min(len(profile.mean_ir) - 1, len(history) + len(source))
    next_history = np.concatenate((history, source))[-keep:].copy()
    return rendered, ReverbStreamState(next_history, effect)


def profile_manifest(profile: ReverbDeviceProfile, path: Path) -> dict:
    return {
        "schema": 1,
        "model_family": "reverb-forward-capture-profile",
        "profile": str(path),
        "profile_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_rate": profile.sample_rate,
        "tail_seconds": PROFILE_SECONDS,
        "source_domain": profile.source_domain,
        "calibration_hash": profile.calibration_hash,
        "learned_device_state": ["mean_impulse", "eight_impulse_bases", "control_grid"],
        "physical_controls": ["decay_s", "damping", "mix"],
        "mix_is_exact": True,
        "runtime_automatic_normalization": False,
        "physical_audio_devices_used": False,
    }


def write_manifest(profile: ReverbDeviceProfile, path: Path, output: Path) -> None:
    output.write_text(json.dumps(profile_manifest(profile, path), indent=2, sort_keys=True) + "\n")
