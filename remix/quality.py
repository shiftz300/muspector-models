"""Audio-quality invariants shared by data, evaluation, and reference DSP.

Analysis may create a mono, 44.1 kHz working copy. The source file is always
read-only, and render/export paths must preserve frames, channels, sample rate,
and floating-point headroom without hidden normalization or limiting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AudioQualityContract:
    schema: int = 1
    source_audio_mutation: str = "forbidden"
    analysis_copy_may_downmix_or_resample: bool = True
    render_preserves_frames: bool = True
    render_preserves_channels: bool = True
    render_preserves_sample_rate: bool = True
    processing_sample_format: str = "float32"
    automatic_normalization: bool = False
    automatic_limiting: bool = False
    automatic_dither: bool = False
    lossy_reencoding: bool = False
    bypass_max_absolute_error: float = 0.0


CONTRACT = AudioQualityContract()


def contract_manifest() -> dict:
    manifest = asdict(CONTRACT)
    manifest["release_gates"] = {
        "non_finite_samples": 0,
        "render_geometry_changes": 0,
        "bypass_max_absolute_error": 0.0,
        "source_file_writes": 0,
        "unintended_lossy_encodes": 0,
        "report_reconstruction_error": True,
    }
    return manifest


def checked_sample_rate(sample_rate: int) -> int:
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError(f"sample rate must be a positive integer, got {sample_rate!r}")
    return int(sample_rate)


def checked_audio(audio: np.ndarray, *, name: str = "audio") -> np.ndarray:
    """Return a float32 working copy, rejecting silent corruption."""

    value = np.asarray(audio, dtype=np.float32)
    if value.ndim not in (1, 2) or value.size == 0:
        raise ValueError(f"{name} must be non-empty mono or stereo audio, got {value.shape}")
    if value.ndim == 2 and min(value.shape) > 2:
        raise ValueError(f"{name} has an unsupported channel layout: {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinite samples")
    return value.copy()


def validate_render(source: np.ndarray, rendered: np.ndarray) -> None:
    """Enforce the no-resample/no-downmix/no-silent-repair render boundary."""

    before = checked_audio(source, name="source")
    after = checked_audio(rendered, name="rendered")
    if after.shape != before.shape:
        raise ValueError(f"renderer changed audio geometry: {before.shape} -> {after.shape}")


def analysis_pair(dry: np.ndarray, wet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Jointly scale an analysis copy without changing dry/wet gain relation."""

    dry_copy = checked_audio(dry, name="dry analysis copy")
    wet_copy = checked_audio(wet, name="wet analysis copy")
    if dry_copy.shape != wet_copy.shape:
        raise ValueError(f"paired audio geometry differs: {dry_copy.shape}/{wet_copy.shape}")
    scale = max(
        float(np.max(np.abs(dry_copy))),
        float(np.max(np.abs(wet_copy))),
        1.0e-5,
    )
    return dry_copy / scale, wet_copy / scale
