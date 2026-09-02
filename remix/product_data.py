"""Product-weight-safe, in-memory Wet/Predecessor pair generation.

Only explicitly authorized clean subsets and repository-owned DSP enter this
module.  Research-only physical datasets are structurally absent, so they
cannot accidentally contribute gradients or model-selection thresholds.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile
import torch
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

from .data import dry_sources
from .foundation_data import RATE
from .license_gate import load_registry, require_product_weights


SPLITS = ("fit", "calibration", "development", "locked-final")
MAX_DELAY_SECONDS = 0.65
MAX_RELEASE_MS = 450.0
RESEARCH_SOURCE_IDS = {
    "asrnn-physical-effects",
    "apple-au-local",
    "spotify-pedalboard-renderer",
    "tonetwist-local-collection",
    "remfx-local",
    "pod-set",
}
PRODUCT_CLEAN_SOURCE_IDS = (
    "guitarjam",
    "guitar-techs",
    "eg-ipt",
    "longitudinal-guitar-string-ageing",
    "freepats-electric-guitar-direct",
    "karoryfer-emilyguitar",
    "dafx25-guitar-effects-chains",
    "egfxset",
)


@dataclass(frozen=True)
class Clean:
    path: Path
    group: str
    split: str
    source_id: str


def _hash(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)


def discover_clean(workspace: Path) -> list[Clean]:
    corpus = workspace / "data/corpus"
    result = []

    # GuitarJam is one player/session and therefore fit-only.  It expands
    # performance content but never pretends to be a held-out domain.
    for path in sorted((corpus / "guitarjam").rglob("*.wav")):
        result.append(Clean(path.resolve(), f"guitarjam:{path.stem}", "fit", "guitarjam"))

    # Guitar-TECHS contributes only electric-guitar direct-input tracks.  Keep
    # whole players out of calibration/development: P1/P2 are fit and the
    # limited P3 musical excerpts remain sealed as locked-final.
    techs_root = corpus / "guitar-techs"
    for path in sorted(techs_root.rglob("*.wav")):
        if "__MACOSX" in path.parts or "directinput" not in path.parts:
            continue
        player = next(
            (part.split("_", 1)[0] for part in path.parts if part.startswith(("P1_", "P2_", "P3_"))),
            None,
        )
        if player not in {"P1", "P2", "P3"}:
            raise ValueError(f"unrecognized Guitar-TECHS player: {path}")
        split = "fit" if player in {"P1", "P2"} else "locked-final"
        group = f"guitar-techs:{player}:{path.parent.parent.parent.name}:{path.stem}"
        result.append(Clean(path.resolve(), group, split, "guitar-techs"))

    # EG-IPT is one player, one guitar and one recording session.  Admit only
    # the raw DI branch and keep the complete domain fit-only; splitting by
    # pickup or note would leak the same performance/session across gates.
    eg_ipt_root = corpus / "eg-ipt"
    for path in sorted(eg_ipt_root.rglob("*.wav")):
        if "DI" not in path.parts:
            continue
        relative = path.relative_to(eg_ipt_root)
        result.append(Clean(path.resolve(), f"eg-ipt:{relative}", "fit", "eg-ipt"))

    # The longitudinal string-ageing corpus contains two performers repeating
    # the same routine over 28 days.  It is verified raw DI, but the repeated
    # material must not create a deceptively easy held-out split.  Keep both
    # complete performer domains fit-only and use one group per performer so a
    # future split change cannot scatter that performer across gates.
    longitudinal_root = (
        corpus / "longitudinal-guitar-string-ageing/LongiGuitarDataset"
    )
    for performer in ("guitarist_1", "guitarist_2"):
        for path in sorted((longitudinal_root / performer).glob("day_*/*.wav")):
            result.append(
                Clean(
                    path.resolve(),
                    f"longitudinal-guitar-string-ageing:{performer}",
                    "fit",
                    "longitudinal-guitar-string-ageing",
                )
            )

    # FreePats is a CC0 sampled instrument, not a corpus of independent
    # performances.  It expands pitch/velocity coverage but remains fit-only
    # and is down-weighted by BlindPresenceData.
    freepats_root = corpus / "freepats-electric-guitar-direct"
    for path in sorted(freepats_root.rglob("*.flac")):
        relative = path.relative_to(freepats_root)
        result.append(
            Clean(
                path.resolve(),
                f"freepats-electric-guitar-direct:{relative}",
                "fit",
                "freepats-electric-guitar-direct",
            )
        )

    # Emilyguitar is also a CC0 sampled instrument.  Only sustained note
    # attacks are useful Clean programs; release and handling noises remain
    # outside training even though their license is compatible.
    emily_root = corpus / "karoryfer-emilyguitar"
    for path in sorted(emily_root.rglob("notes/*.wav")):
        relative = path.relative_to(emily_root)
        result.append(
            Clean(
                path.resolve(),
                f"karoryfer-emilyguitar:{relative}",
                "fit",
                "karoryfer-emilyguitar",
            )
        )

    dafx_map = {
        "train": "fit",
        "valid": "calibration",
        "calibrate": "development",
        "test": "locked-final",
    }
    dafx_root = corpus / "guitar-effects-chains"
    for upstream, split in dafx_map.items():
        for path in dry_sources(dafx_root, upstream):
            result.append(Clean(path.resolve(), f"dafx:{path.stem}", split, "dafx25-guitar-effects-chains"))

    # Keep the same note/performance in one split across all pickup positions.
    egfx_root = corpus / "egfxset/Clean"
    for path in sorted(egfx_root.rglob("*.wav")):
        group = f"egfx:{path.stem}"
        bucket = _hash(group) % 10
        split = "fit" if bucket < 7 else "calibration" if bucket == 7 else "development" if bucket == 8 else "locked-final"
        result.append(Clean(path.resolve(), group, split, "egfxset"))

    owners = {}
    for clean in result:
        if not clean.path.is_file():
            raise FileNotFoundError(clean.path)
        previous = owners.setdefault(clean.group, clean.split)
        if previous != clean.split:
            raise ValueError(f"clean group crosses product splits: {clean.group}: {previous}/{clean.split}")
        if clean.source_id in RESEARCH_SOURCE_IDS:
            raise PermissionError(f"research source entered product clean inventory: {clean.source_id}")
    if not result:
        raise ValueError("no product-eligible clean programs")
    return result


def _read(clean: Clean, frames: int, seed: int) -> np.ndarray:
    info = soundfile.info(clean.path)
    target_frames = round(info.frames * RATE / info.samplerate)
    rng = random.Random(seed)
    start_target = rng.randrange(max(1, target_frames - frames + 1))
    start_source = round(start_target * info.samplerate / RATE)
    requested = math.ceil(frames * info.samplerate / RATE) + 16
    with soundfile.SoundFile(clean.path) as stream:
        stream.seek(min(start_source, len(stream)))
        value = stream.read(requested, dtype="float32", always_2d=True).mean(axis=1)
    if info.samplerate != RATE:
        divisor = math.gcd(info.samplerate, RATE)
        value = resample_poly(value, RATE // divisor, info.samplerate // divisor).astype(np.float32)
    if len(value) < frames:
        value = np.pad(value, (0, frames - len(value)))
    value = np.asarray(value[:frames], dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite clean program: {clean.path}")
    return value


def nonlinear(clean: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    drive = rng.uniform(1.8, 8.0)
    bias = rng.uniform(-0.12, 0.12)
    shape = rng.randrange(3)
    oversampled = resample_poly(clean, 2, 1).astype(np.float64)
    value = oversampled * drive + bias
    if shape == 0:
        value = np.tanh(value) - math.tanh(bias)
        name = "tanh"
    elif shape == 1:
        value = (2.0 / math.pi) * np.arctan(value * 1.8) - (2.0 / math.pi) * math.atan(bias * 1.8)
        name = "atan"
    else:
        value = np.clip(value, -1.5, 1.5)
        value = value - value**3 / 6.75
        name = "cubic"
    value = resample_poly(value, 1, 2)[: len(clean)]
    cutoff = rng.uniform(1_800.0, 11_000.0)
    value = sosfilt(butter(2, cutoff, btype="lowpass", fs=RATE, output="sos"), value)
    level = rng.uniform(0.45, 0.85)
    return np.asarray(value * level, dtype=np.float32), {
        "shape": name,
        "drive": drive,
        "bias": bias,
        "cutoff_hz": cutoff,
        "level": level,
    }


def dynamics(clean: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    threshold_db = rng.uniform(-36.0, -12.0)
    ratio = rng.uniform(2.0, 10.0)
    attack_ms = rng.uniform(0.5, 35.0)
    release_ms = rng.uniform(40.0, MAX_RELEASE_MS)
    makeup_db = rng.uniform(0.0, 8.0)
    attack = math.exp(-1.0 / (RATE * attack_ms / 1000.0))
    release = math.exp(-1.0 / (RATE * release_ms / 1000.0))
    envelope = 0.0
    wet = np.empty_like(clean, dtype=np.float64)
    for index, sample in enumerate(np.asarray(clean, dtype=np.float64)):
        level = abs(sample)
        coefficient = attack if level > envelope else release
        envelope = coefficient * envelope + (1.0 - coefficient) * level
        level_db = 20.0 * math.log10(max(envelope, 1.0e-7))
        reduction_db = max(0.0, level_db - threshold_db) * (1.0 - 1.0 / ratio)
        wet[index] = sample * 10.0 ** ((makeup_db - reduction_db) / 20.0)
    return wet.astype(np.float32), {
        "threshold_db": threshold_db,
        "ratio": ratio,
        "attack_ms": attack_ms,
        "release_ms": release_ms,
        "makeup_db": makeup_db,
    }


def _delay(clean: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    delay_frames = round(rng.uniform(0.04, MAX_DELAY_SECONDS) * RATE)
    feedback = rng.uniform(0.1, 0.72)
    mix = rng.uniform(0.15, 0.65)
    echo = np.zeros_like(clean, dtype=np.float64)
    for index in range(delay_frames, len(clean)):
        echo[index] = clean[index - delay_frames] + feedback * echo[index - delay_frames]
    wet = (1.0 - mix) * clean + mix * echo
    return wet.astype(np.float32), {
        "kind": "delay",
        "time_ms": delay_frames * 1000.0 / RATE,
        "feedback": feedback,
        "mix": mix,
    }


@lru_cache(maxsize=16)
def _rir(path: Path) -> np.ndarray:
    value, rate = soundfile.read(path, dtype="float32", always_2d=True)
    value = value[:, 0]
    if rate != RATE:
        divisor = math.gcd(rate, RATE)
        value = resample_poly(value, RATE // divisor, rate // divisor).astype(np.float32)
    peak = float(np.max(np.abs(value)))
    active = np.flatnonzero(np.abs(value) >= max(peak * 0.01, 1.0e-8))
    if not len(active):
        raise ValueError(f"silent RIR: {path}")
    value = value[int(active[0]) : RATE * 8]
    value /= math.sqrt(float(np.sum(value.astype(np.float64) ** 2)) + 1.0e-12)
    value.setflags(write=False)
    return value


def rir_splits(workspace: Path) -> dict[str, list[Path]]:
    root = workspace / "data/corpus/aachen-chapel-rir"
    paths = sorted(
        path.resolve() for path in root.rglob("*.wav")
        if "brir" in "/".join(part.lower() for part in path.parts)
    )
    boundaries = (int(len(paths) * 0.60), int(len(paths) * 0.76), int(len(paths) * 0.88))
    return {
        "fit": paths[: boundaries[0]],
        "calibration": paths[boundaries[0] : boundaries[1]],
        "development": paths[boundaries[1] : boundaries[2]],
        "locked-final": paths[boundaries[2] :],
    }


def _reverb(clean: np.ndarray, path: Path, rng: random.Random) -> tuple[np.ndarray, dict]:
    impulse = _rir(path)
    diffuse = fftconvolve(clean, impulse, mode="full")[: len(clean)].astype(np.float64)
    clean_rms = math.sqrt(float(np.mean(clean.astype(np.float64) ** 2)) + 1.0e-12)
    wet_rms = math.sqrt(float(np.mean(diffuse**2)) + 1.0e-12)
    diffuse *= clean_rms / wet_rms
    mix = rng.uniform(0.2, 0.7)
    wet = (1.0 - mix) * clean + mix * diffuse
    return wet.astype(np.float32), {"kind": "reverb", "rir": path.name, "mix": mix}


class ProductPairs(torch.utils.data.Dataset):
    def __init__(
        self,
        workspace: Path,
        mechanism: str,
        split: str,
        samples: int,
        frames: int,
        seed: int,
    ) -> None:
        if mechanism not in {"nonlinear", "dynamics", "temporal"} or split not in SPLITS:
            raise ValueError(f"unsupported product pair selection: {mechanism}/{split}")
        if samples < 1 or frames < 4096:
            raise ValueError("product pair request is too small")
        self.workspace = workspace.resolve()
        self.mechanism = mechanism
        self.split = split
        self.samples = samples
        self.frames = frames
        self.seed = seed
        self.clean = [item for item in discover_clean(self.workspace) if item.split == split]
        if not self.clean:
            raise ValueError(f"no product clean programs for {split}")
        sources = {item.source_id for item in self.clean}
        sources.add("muspector-dsp")
        if mechanism == "temporal":
            sources.add("aachen-chapel-rir")
            self.rirs = rir_splits(self.workspace)[split]
            if not self.rirs:
                raise ValueError(f"no product RIRs for {split}")
        else:
            self.rirs = []
        self.authorization = require_product_weights(
            self.workspace / "remix/data_sources.json", sorted(sources)
        )
        if sources & RESEARCH_SOURCE_IDS:
            raise PermissionError(f"research sources entered product dataset: {sources & RESEARCH_SOURCE_IDS}")

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict:
        rng = random.Random(self.seed + index * 104_729)
        selected = self.clean[index % len(self.clean)]
        clean = _read(selected, self.frames, rng.getrandbits(63))
        if self.mechanism == "nonlinear":
            wet, controls = nonlinear(clean, rng)
        elif self.mechanism == "dynamics":
            wet, controls = dynamics(clean, rng)
        elif index % 2 == 0:
            wet, controls = _delay(clean, rng)
        else:
            wet, controls = _reverb(clean, self.rirs[(index // 2) % len(self.rirs)], rng)
        if wet.shape != clean.shape or not np.isfinite(wet).all():
            raise ValueError("product renderer changed geometry or emitted non-finite audio")
        return {
            "wet": torch.from_numpy(wet.copy()),
            "clean": torch.from_numpy(clean.copy()),
            "source_id": selected.source_id,
            "group": selected.group,
            "controls": controls,
        }


def audit(workspace: Path) -> dict:
    clean = discover_clean(workspace.resolve())
    counts = {split: {} for split in SPLITS}
    for split in SPLITS:
        for source_id in sorted({item.source_id for item in clean}):
            counts[split][source_id] = sum(
                item.split == split and item.source_id == source_id for item in clean
            )
    registry = load_registry(workspace / "remix/data_sources.json")
    authorization = require_product_weights(
        workspace / "remix/data_sources.json",
        (*PRODUCT_CLEAN_SOURCE_IDS, "muspector-dsp", "aachen-chapel-rir"),
    )
    return {
        "schema": 1,
        "status": "product-pair-generation-authorized",
        "clean_programs": len(clean),
        "clean_counts": counts,
        "rir_counts": {name: len(paths) for name, paths in rir_splits(workspace).items()},
        "authorization": authorization,
        "research_sources_present": False,
        "generated_audio_written": False,
        "source_audio_read_only": True,
        "physical_audio_devices_used": False,
        "locked_final_audio_opened": False,
        "registry_schema": registry["schema"],
    }
