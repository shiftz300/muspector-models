"""Independent Spotify Pedalboard domain for out-of-distribution audits.

This module is intentionally outside ``render.py``: Pedalboard is an optional
research dependency, never part of the application runtime or canonical
synthetic training contract.
"""

from __future__ import annotations

from importlib.metadata import version

import numpy as np
from pedalboard import Delay as PedalboardDelay
from pedalboard import Distortion, Gain, LowpassFilter, Pedalboard
from pedalboard import Reverb as PedalboardReverb

from .quality import checked_audio, checked_sample_rate, validate_render
from .spec import ChainSpec, Delay, Drive, Reverb


# Measured with a 12-second impulse at 44.1 kHz, using a Schroeder energy
# decay curve and a -5 to -30 dB linear fit extrapolated to T60. Damping has
# less than 4% effect on these values; the target damping remains independently
# randomized and labeled. Values below 0.50 seconds are not representable by
# this implementation and therefore are not sampled in this audit domain.
REVERB_ROOM_SIZE = np.linspace(0.0, 1.0, 11, dtype=np.float64)
REVERB_T60_SECONDS = np.asarray(
    (0.518, 0.584, 0.664, 0.765, 0.893, 1.071, 1.325, 1.727, 2.416, 3.882, 9.484),
    dtype=np.float64,
)
MIN_REVERB_DECAY_SECONDS = 0.52
MAX_REVERB_DECAY_SECONDS = 8.0


def reverb_room_size(decay_seconds: float) -> float:
    """Invert the measured Pedalboard room-size/T60 curve."""

    if not MIN_REVERB_DECAY_SECONDS <= decay_seconds <= MAX_REVERB_DECAY_SECONDS:
        raise ValueError(
            "Pedalboard validation decay must be within "
            f"[{MIN_REVERB_DECAY_SECONDS}, {MAX_REVERB_DECAY_SECONDS}], "
            f"got {decay_seconds}"
        )
    return float(np.interp(decay_seconds, REVERB_T60_SECONDS, REVERB_ROOM_SIZE))


def render_pedalboard_chain(
    audio: np.ndarray, spec: ChainSpec, sample_rate: int
) -> np.ndarray:
    """Render a semantic chain with Pedalboard's JUCE-backed built-ins."""

    spec.validate()
    checked_sample_rate(sample_rate)
    source = checked_audio(audio)
    frame_major = source.ndim == 2 and source.shape[1] <= 2
    value = source.T if frame_major else source

    plugins = []
    for effect in spec.effects:
        if isinstance(effect, Drive):
            # Pedalboard Distortion exposes drive only. The two explicit studio
            # processors preserve Muspector's Tone and Level control contract.
            cutoff = 900.0 * (12_000.0 / 900.0) ** effect.tone
            plugins.extend(
                (
                    Distortion(drive_db=effect.gain_db),
                    LowpassFilter(cutoff_frequency_hz=cutoff),
                    Gain(gain_db=effect.level_db),
                )
            )
        elif isinstance(effect, Delay):
            plugins.append(
                PedalboardDelay(
                    delay_seconds=effect.time_ms / 1_000.0,
                    feedback=effect.feedback,
                    mix=effect.mix,
                )
            )
        elif isinstance(effect, Reverb):
            plugins.append(
                PedalboardReverb(
                    room_size=reverb_room_size(effect.decay_s),
                    damping=effect.damping,
                    wet_level=effect.mix,
                    dry_level=1.0 - effect.mix,
                    width=1.0,
                    freeze_mode=0.0,
                )
            )
        else:  # pragma: no cover - guarded by ChainSpec validation.
            raise TypeError(f"unsupported effect: {effect!r}")

    rendered = Pedalboard(plugins)(value, sample_rate, reset=True)
    rendered = np.asarray(rendered, dtype=np.float32)
    if rendered.shape != value.shape:
        raise ValueError(f"Pedalboard changed audio shape: {value.shape} -> {rendered.shape}")
    result = rendered.T if frame_major else rendered
    validate_render(source, result)
    return result


def pedalboard_manifest() -> dict:
    return {
        "schema": 1,
        "evaluation_only": True,
        "library": "spotify/pedalboard",
        "version": version("pedalboard"),
        "license": "GPL-3.0",
        "source": "https://github.com/spotify/pedalboard",
        "documentation": "https://spotify.github.io/pedalboard/reference/pedalboard.html",
        "citation": "https://doi.org/10.5281/zenodo.7817838",
        "processors": {
            "drive": ["Distortion", "LowpassFilter", "Gain"],
            "delay": ["Delay"],
            "reverb": ["Reverb"],
        },
        "reverb_calibration": {
            "method": "12 s impulse, Schroeder EDC, -5 to -30 dB T30 fit extrapolated to T60",
            "sample_rate": 44_100,
            "damping": 0.5,
            "room_size": REVERB_ROOM_SIZE.tolist(),
            "measured_t60_seconds": REVERB_T60_SECONDS.tolist(),
            "sampled_decay_seconds": [
                MIN_REVERB_DECAY_SECONDS,
                MAX_REVERB_DECAY_SECONDS,
            ],
            "excluded_contract_range_seconds": [0.2, MIN_REVERB_DECAY_SECONDS],
        },
    }
