"""Product-safe on-the-fly data for Wet-only multi-label presence training."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import soundfile
import torch
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

from .blind2 import LABELS, RATE, WINDOW
from .guitar_chain_data import discover as discover_chain_presence
from .guitar_chain_data import inventory as chain_presence_inventory
from .license_gate import require_product_uses
from .product_data import RATE as RENDER_RATE
from .product_data import Clean, PRODUCT_CLEAN_SOURCE_IDS, _read, _rir, discover_clean, rir_splits


RENDER_FRAMES = RENDER_RATE * 5
REAL_EFFECTS = {
    "nonlinear": ("BluesDriver", "RAT", "TubeScreamer"),
    "echo": ("Digital-Delay", "Sweep-Echo", "TapeEcho"),
    "ambience": ("Hall-Reverb", "Plate-Reverb", "Spring-Reverb"),
    "unknown": ("Chorus", "Flanger", "Phaser"),
}
SOURCE_CYCLE_WEIGHT = {
    # These are isolated-note domains from one player/instrument, so they may
    # broaden timbre and technique without dominating real phrase sources.
    "eg-ipt": 2,
    # These are genuine phrase performances, but all 28 days repeat one fixed
    # routine per performer.  Keep them below a fully diverse source cycle.
    "longitudinal-guitar-string-ageing": 3,
    "freepats-electric-guitar-direct": 1,
    "karoryfer-emilyguitar": 1,
    # The fixed amplifier line output is not raw DI.  It is admitted only as
    # positive-only family-presence augmentation, never as Clean supervision.
    "multimodal-electric-guitar-data": 1,
}
DEFAULT_SOURCE_CYCLE_WEIGHT = 4
MULTIMODAL_SOURCE_ID = "multimodal-electric-guitar-data"
MULTIMODAL_EXCLUDED = {
    "guit_raw_dir/p13/free_70/free_70.wav",
    "guit_raw_dir/p13/bending_soft_70/bending_soft_70.wav",
    "guit_raw_dir/p24/free_encore_120/free_encore_120.wav",
    "guit_raw_dir/p24/free_70/free_70.wav",
    "guit_raw_dir/p26/sustained_soft_70/sustained_soft_70.wav",
}


def _multimodal_fit_sources(workspace: Path) -> list[Clean]:
    """Discover fit-only amplifier baselines with known labels masked out."""
    root = workspace / "data/corpus/multimodal-electric-guitar"
    result = []
    for path in sorted(root.rglob("*.wav")):
        relative = path.relative_to(root).as_posix()
        if relative in MULTIMODAL_EXCLUDED:
            continue
        parts = Path(relative).parts
        if len(parts) < 4 or parts[0] != "guit_raw_dir" or not parts[1].startswith("p"):
            raise ValueError(f"unexpected Multimodal Electric Guitar path: {relative}")
        player = parts[1]
        result.append(
            Clean(
                path.resolve(),
                f"multimodal-electric-guitar-data:{player}",
                "fit",
                MULTIMODAL_SOURCE_ID,
            )
        )
    return result


def _resample(value: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == RATE:
        result = value
    else:
        divisor = math.gcd(source_rate, RATE)
        result = resample_poly(value, RATE // divisor, source_rate // divisor)
    if len(result) < WINDOW:
        result = np.pad(result, (0, WINDOW - len(result)))
    return np.asarray(result[:WINDOW], dtype=np.float32)


NONLINEAR_IMPLEMENTATIONS = (
    "tanh",
    "atan",
    "hard-clip",
    "asymmetric-tanh",
    "square-root",
    "gate-tanh",
)


def _nonlinear(
    value: np.ndarray,
    rng: random.Random,
    details: dict[str, str | float | int] | None = None,
) -> np.ndarray:
    drive = rng.uniform(1.4, 10.0)
    bias = rng.uniform(-0.15, 0.15)
    shaped = value.astype(np.float64) * drive + bias
    implementation = rng.randrange(len(NONLINEAR_IMPLEMENTATIONS))
    if implementation == 0:
        shaped = np.tanh(shaped) - math.tanh(bias)
    elif implementation == 1:
        shaped = (2.0 / math.pi) * np.arctan(shaped * 1.8)
        shaped -= (2.0 / math.pi) * math.atan(bias * 1.8)
    elif implementation == 2:
        shaped = np.clip(shaped, -1.0, 1.0)
    elif implementation == 3:
        shaped = np.where(shaped >= 0.0, np.tanh(shaped), 0.58 * np.tanh(1.8 * shaped))
    elif implementation == 4:
        shaped = np.sign(shaped) * np.sqrt(np.minimum(np.abs(shaped), 1.0))
    else:
        gate = rng.uniform(0.015, 0.08)
        shaped = np.where(np.abs(shaped) < gate, 0.0, np.tanh(shaped * 2.4))
    cutoff = rng.uniform(1_600.0, 15_000.0)
    shaped = sosfilt(butter(2, cutoff, btype="lowpass", fs=RENDER_RATE, output="sos"), shaped)
    output_gain = rng.uniform(0.35, 0.9)
    result = np.asarray(shaped * output_gain, dtype=np.float32)
    if details is not None:
        dry = np.asarray(value, dtype=np.float64)
        wet = np.asarray(result, dtype=np.float64)
        scale = float(np.dot(dry, wet) / (np.dot(dry, dry) + 1.0e-12))
        residual = wet - scale * dry
        residual_ratio = math.sqrt(float(np.mean(residual**2)) + 1.0e-12) / (
            math.sqrt(float(np.mean((scale * dry) ** 2)) + 1.0e-12)
        )
        details.update({
            "implementation": NONLINEAR_IMPLEMENTATIONS[implementation],
            "implementation_index": implementation,
            "drive": drive,
            "cutoff_hz": cutoff,
            "output_gain": output_gain,
            "scale_invariant_residual_db": 20.0 * math.log10(max(residual_ratio, 1.0e-12)),
        })
    return result


def _echo(value: np.ndarray, rng: random.Random) -> np.ndarray:
    implementation = rng.randrange(4)
    if implementation == 0:
        delay = rng.uniform(0.16, 0.65)
    elif implementation == 1:
        delay = rng.uniform(0.035, 0.14)
    elif implementation == 2:
        delay = rng.uniform(0.25, 0.55)
    else:
        delay = rng.uniform(0.08, 0.42)
    frames = max(1, round(delay * RENDER_RATE))
    feedback = rng.uniform(0.08, 0.72)
    mix = rng.uniform(0.16, 0.68)
    echo = np.zeros_like(value, dtype=np.float64)
    gain = 1.0
    for offset in range(frames, len(value), frames):
        echo[offset:] += gain * value[:-offset]
        gain *= feedback
        if gain < 1.0e-3:
            break
    if implementation in (2, 3):
        cutoff = rng.uniform(1_800.0, 6_500.0)
        echo = sosfilt(butter(1, cutoff, btype="lowpass", fs=RENDER_RATE, output="sos"), echo)
    if implementation == 3:
        timeline = np.arange(len(echo), dtype=np.float64) / RENDER_RATE
        echo *= 1.0 + 0.04 * np.sin(2.0 * math.pi * rng.uniform(0.2, 1.2) * timeline)
    return np.asarray((1.0 - mix) * value + mix * echo, dtype=np.float32)


def _synthetic_rir(rng: random.Random) -> np.ndarray:
    seconds = rng.uniform(0.28, 2.5)
    frames = max(64, round(seconds * RENDER_RATE))
    time = np.arange(frames, dtype=np.float64) / RENDER_RATE
    decay = np.exp(-time * rng.uniform(2.0, 10.0))
    noise = np.random.default_rng(rng.getrandbits(63)).normal(size=frames)
    impulse = noise * decay
    impulse[0] += rng.uniform(1.0, 2.5)
    for _ in range(rng.randrange(2, 9)):
        index = rng.randrange(1, min(frames, round(0.16 * RENDER_RATE)))
        impulse[index] += rng.uniform(-0.8, 0.8) * math.exp(-index / RENDER_RATE * 5.0)
    impulse /= math.sqrt(float(np.sum(impulse**2)) + 1.0e-12)
    return impulse.astype(np.float32)


def _ambience(value: np.ndarray, rng: random.Random, measured: list[Path]) -> np.ndarray:
    impulse = _rir(rng.choice(measured)) if measured and rng.random() < 0.6 else _synthetic_rir(rng)
    diffuse = fftconvolve(value, impulse, mode="full")[: len(value)].astype(np.float64)
    source_rms = math.sqrt(float(np.mean(value.astype(np.float64) ** 2)) + 1.0e-12)
    diffuse_rms = math.sqrt(float(np.mean(diffuse**2)) + 1.0e-12)
    diffuse *= source_rms / diffuse_rms
    mix = rng.uniform(0.16, 0.72)
    return np.asarray((1.0 - mix) * value + mix * diffuse, dtype=np.float32)


def _unknown(value: np.ndarray, rng: random.Random) -> np.ndarray:
    implementation = rng.randrange(4)
    time = np.arange(len(value), dtype=np.float64) / RENDER_RATE
    if implementation == 0:  # tremolo
        depth = rng.uniform(0.25, 0.9)
        mod = 1.0 - depth + depth * (0.5 + 0.5 * np.sin(2.0 * math.pi * rng.uniform(1.0, 11.0) * time))
        return np.asarray(value * mod, dtype=np.float32)
    rate = rng.uniform(0.08, 4.5)
    if implementation in (1, 2):  # chorus/flanger
        base_ms = rng.uniform(9.0, 28.0) if implementation == 1 else rng.uniform(0.5, 6.0)
        depth_ms = rng.uniform(1.0, 8.0) if implementation == 1 else rng.uniform(0.2, 3.0)
        delay = (base_ms + depth_ms * np.sin(2.0 * math.pi * rate * time)) * RENDER_RATE / 1000.0
        source = np.arange(len(value), dtype=np.float64) - delay
        shifted = np.interp(source, np.arange(len(value)), value, left=0.0, right=0.0)
        mix = rng.uniform(0.25, 0.65)
        return np.asarray((1.0 - mix) * value + mix * shifted, dtype=np.float32)
    # ring modulation / rotary-like amplitude and polarity movement
    carrier = np.sin(2.0 * math.pi * rng.uniform(18.0, 85.0) * time)
    mix = rng.uniform(0.18, 0.52)
    return np.asarray((1.0 - mix) * value + mix * value * carrier, dtype=np.float32)


def _nuisance(value: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply label-preserving linear recording variation only.

    Clipping cannot be treated as a nuisance here: it is itself a nonlinear
    effect and would create false-negative nonlinear labels.
    """

    result = np.asarray(value, dtype=np.float64) * 10.0 ** (rng.uniform(-10.0, 6.0) / 20.0)
    if rng.random() < 0.7:
        low = rng.uniform(35.0, 180.0)
        high = rng.uniform(5_000.0, 18_000.0)
        result = sosfilt(butter(2, (low, high), btype="bandpass", fs=RENDER_RATE, output="sos"), result)
    if rng.random() < 0.5:
        noise = np.random.default_rng(rng.getrandbits(63)).normal(size=len(result))
        signal_rms = math.sqrt(float(np.mean(result**2)) + 1.0e-12)
        result += noise * signal_rms * 10.0 ** (-rng.uniform(28.0, 60.0) / 20.0)
    return np.asarray(result, dtype=np.float32)


def _read_real(path: Path) -> np.ndarray:
    value, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return _resample(value.mean(axis=1), rate)


def _real_inventory(workspace: Path, split: str) -> dict[str, list[Path]]:
    clean_split = {item.path.stem: item.split for item in discover_clean(workspace) if item.source_id == "egfxset"}
    result = {label: [] for label in LABELS}
    root = workspace / "data/corpus/egfxset"
    for label, directories in REAL_EFFECTS.items():
        for directory in directories:
            for path in sorted((root / directory).rglob("*.wav")):
                if clean_split.get(path.stem) == split:
                    result[label].append(path.resolve())
    return result


class BlindPresenceData(torch.utils.data.Dataset):
    """Random-order chains; output contains presence only, never order."""

    def __init__(
        self,
        workspace: Path,
        split: str,
        samples: int,
        seed: int,
        *,
        include_nonlinear_audit_pair: bool = False,
    ) -> None:
        if split not in {"fit", "calibration", "development", "locked-final"}:
            raise ValueError(f"invalid split: {split}")
        if samples < 1:
            raise ValueError("samples must be positive")
        self.workspace = workspace.resolve()
        self.split = split
        self.samples = samples
        self.seed = seed
        self.epoch = 0
        self.include_nonlinear_audit_pair = include_nonlinear_audit_pair
        self.clean = [item for item in discover_clean(self.workspace) if item.split == split]
        if split == "fit":
            self.clean.extend(_multimodal_fit_sources(self.workspace))
        self.clean_by_source = {
            source: [item for item in self.clean if item.source_id == source]
            for source in sorted({item.source_id for item in self.clean})
        }
        self.source_schedule = tuple(
            source
            for source in sorted(self.clean_by_source)
            for _ in range(SOURCE_CYCLE_WEIGHT.get(source, DEFAULT_SOURCE_CYCLE_WEIGHT))
        )
        self.rirs = rir_splits(self.workspace)[split]
        self.real = _real_inventory(self.workspace, split)
        self.random_position_chains = discover_chain_presence(
            self.workspace / "data/corpus/guitar-effects-chains", split
        )
        if not self.clean or not all(self.real.values()) or not self.random_position_chains:
            raise ValueError(f"incomplete Blind product inventory for {split}")
        requirements = {
            source: "product-clean-source" for source in PRODUCT_CLEAN_SOURCE_IDS
        }
        requirements["egfxset"] = ("product-clean-source", "train-family")
        requirements["dafx25-guitar-effects-chains"] = (
            "product-clean-source", "train-family", "train-order"
        )
        requirements[MULTIMODAL_SOURCE_ID] = "train-family-positive-only"
        requirements["muspector-dsp"] = ("product-pair-generation", "train-restoration")
        requirements["aachen-chapel-rir"] = "train-reverb"
        self.authorization = require_product_uses(
            self.workspace / "remix/data_sources.json", requirements
        )

    def __len__(self) -> int:
        return self.samples

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        rng = random.Random(self.seed + self.epoch * 15_485_863 + index * 104_729)
        # Real normalized EGFx Wet is admitted for family recognition only and
        # serves as a single-effect hardware anchor, never an inverse target.
        if index % 4 == 0:
            label = LABELS[(index // 4) % len(LABELS)]
            path = self.real[label][
                (index // (4 * len(LABELS)) + self.epoch * 1_009) % len(self.real[label])
            ]
            audio = _read_real(path)
            target = np.asarray([float(name == label) for name in LABELS], dtype=np.float32)
            target_mask = np.ones(len(LABELS), dtype=np.float32)
            return {
                "audio": torch.from_numpy(audio.copy()),
                "target": torch.from_numpy(target),
                "target_mask": torch.from_numpy(target_mask),
                "source": "egfxset-real",
            }

        # Author-published Dataset 3 changes effect positions and parameters.
        # This package consumes presence bits only; the separate order package
        # owns the realized-order JSON and no order enters this target.
        if index % 8 == 1:
            selected = self.random_position_chains[
                (index // 8 + self.epoch * 1_009) % len(self.random_position_chains)
            ]
            return {
                "audio": torch.from_numpy(_read_real(selected.path).copy()),
                "target": torch.from_numpy(selected.target.copy()),
                "target_mask": torch.ones(len(LABELS), dtype=torch.float32),
                "source": "dafx-random-position-presence",
            }

        synthetic_rank = index - (index + 3) // 4 - (index + 6) // 8
        source_name = self.source_schedule[
            (synthetic_rank + self.epoch) % len(self.source_schedule)
        ]
        source_rows = self.clean_by_source[source_name]
        selected = source_rows[
            (synthetic_rank // len(self.source_schedule) + self.epoch * 1_009) % len(source_rows)
        ]
        audio = _read(selected, RENDER_FRAMES, rng.getrandbits(63))
        audio = _nuisance(audio, rng)
        positive_only = selected.source_id == MULTIMODAL_SOURCE_ID
        draw = rng.random()
        if positive_only:
            # The undocumented amplifier state makes absent labels unknown.
            # Apply overlapping chains and supervise only effects we add.
            count = 2 if draw < 0.55 else 3 if draw < 0.90 else 4
        else:
            count = 0 if draw < 0.30 else 1 if draw < 0.55 else 2 if draw < 0.85 else 3 if draw < 0.97 else 4
        active = rng.sample(list(LABELS), count)
        rng.shuffle(active)
        nonlinear_details: dict[str, str | float | int] = {}
        nonlinear_predecessor: np.ndarray | None = None
        nonlinear_wet: np.ndarray | None = None
        for label in active:
            if label == "nonlinear":
                if getattr(self, "include_nonlinear_audit_pair", False):
                    nonlinear_predecessor = audio.copy()
                audio = _nonlinear(audio, rng, nonlinear_details)
                if getattr(self, "include_nonlinear_audit_pair", False):
                    nonlinear_wet = audio.copy()
            elif label == "echo":
                audio = _echo(audio, rng)
            elif label == "ambience":
                audio = _ambience(audio, rng, self.rirs)
            else:
                audio = _unknown(audio, rng)
        audio = _nuisance(audio, rng)
        audio = _resample(audio, RENDER_RATE)
        if not np.isfinite(audio).all() or audio.shape != (WINDOW,):
            raise ValueError("Blind renderer emitted invalid audio")
        target = np.asarray([float(label in active) for label in LABELS], dtype=np.float32)
        target_mask = np.asarray(
            [float(label in active) if positive_only else 1.0 for label in LABELS],
            dtype=np.float32,
        )
        if nonlinear_details:
            drive = float(nonlinear_details["drive"])
            drive_bucket = "low" if drive < 2.5 else "medium" if drive < 5.0 else "high"
            residual_db = float(nonlinear_details["scale_invariant_residual_db"])
            residual_bucket = (
                "weak" if residual_db < -12.0 else "medium" if residual_db < -6.0 else "strong"
            )
            diagnostic_source = (
                "synthetic-product-render:nonlinear:"
                f"{nonlinear_details['implementation']}:{drive_bucket}:{residual_bucket}:"
                f"count-{count}:{selected.source_id}"
            )
        else:
            diagnostic_source = (
                f"synthetic-product-render:no-nonlinear:count-{count}:{selected.source_id}"
            )
        result = {
            "audio": torch.from_numpy(audio.copy()),
            "target": torch.from_numpy(target),
            "target_mask": torch.from_numpy(target_mask),
            "source": diagnostic_source,
        }
        if getattr(self, "include_nonlinear_audit_pair", False) and nonlinear_predecessor is not None:
            assert nonlinear_wet is not None
            result["nonlinear_predecessor"] = torch.from_numpy(
                nonlinear_predecessor.astype(np.float32, copy=True)
            )
            result["nonlinear_wet"] = torch.from_numpy(
                nonlinear_wet.astype(np.float32, copy=True)
            )
            result["nonlinear_pair_sample_rate"] = RENDER_RATE
        return result


def inventory(workspace: Path) -> dict:
    workspace = workspace.resolve()
    clean = discover_clean(workspace)
    positive_only = _multimodal_fit_sources(workspace)
    return {
        "clean_programs": len(clean),
        "fit_clean_programs": sum(item.split == "fit" for item in clean),
        "positive_only_amplifier_baselines": len(positive_only),
        "positive_only_player_groups": len({item.group for item in positive_only}),
        "random_position_chain_presence": chain_presence_inventory(
            workspace / "data/corpus/guitar-effects-chains"
        ),
        "real_hardware_examples": {
            split: {label: len(paths) for label, paths in _real_inventory(workspace, split).items()}
            for split in ("fit", "calibration", "development", "locked-final")
        },
        "generated_wet_written": False,
        "order_labels_emitted": False,
        "controls_emitted": False,
    }
