"""Product4 Ambience pairs using room-disjoint BUT ReverbDB RIRs."""

import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .ambience2 import AmbiencePairsV2, _rir_decay_seconds
from .product_data import _rir


DECAY_DOMAIN_SECONDS = (0.10, 1.00)
DECAY_STRATA_SECONDS = (0.40, 0.75)
TARGET_FRAMES_MINIMUM = 65_536
DECAY_BANK_VARIANTS = tuple(
    (half_life_ms, ratio_threshold, 0.80)
    for half_life_ms in (150.0, 350.0, 700.0)
    for ratio_threshold in (0.50, 0.65)
)
PROFILE_SHAPING_STRENGTHS = (0.18, 0.22, 0.26, 0.30)
PRODUCT4_CLEAN_SOURCE_IDS = frozenset((
    "dafx25-guitar-effects-chains",
    "egfxset",
    "guitar-techs",
    "guitarjam",
    "longitudinal-guitar-string-ageing",
))


def multiband_decay_suppression(
    wet: torch.Tensor,
    *,
    half_life_ms: float,
    ratio_threshold: float,
    strength: float,
    n_fft: int = 1024,
    hop: int = 256,
) -> torch.Tensor:
    """Attenuate Wet-only band tails detected by a causal exponential peak state."""
    if wet.ndim != 2 or wet.shape[1] < n_fft * 2 or not torch.isfinite(wet).all():
        raise ValueError("decay suppression expects finite [batch,time] ambience audio")
    if not 50.0 <= half_life_ms <= 2_000.0:
        raise ValueError("decay suppression half-life is outside the admitted range")
    if not 0.20 <= ratio_threshold <= 0.90 or not 0.0 <= strength <= 1.0:
        raise ValueError("invalid decay suppression threshold or strength")
    if n_fft < 512 or hop < 64:
        raise ValueError("invalid decay suppression spectral geometry")
    window = torch.hann_window(n_fft, device=wet.device, dtype=wet.dtype)
    spectrum = torch.stft(
        wet, n_fft, hop, window=window, center=True,
        pad_mode="constant", return_complex=True,
    )
    magnitude = spectrum.abs()
    frequencies = torch.linspace(
        0.0, 24_000.0, spectrum.shape[-2], device=wet.device, dtype=wet.dtype
    )
    gain = torch.ones_like(magnitude)
    alpha = math.exp(-hop / (48_000.0 * half_life_ms / 1_000.0))
    edges = (0.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0, 24_001.0)
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        selected = (frequencies >= low) & (frequencies < high)
        if not bool(selected.any()):
            continue
        level = magnitude[:, selected].square().mean(dim=1).clamp_min(1.0e-12).sqrt()
        states = [level[:, 0]]
        for frame in range(1, level.shape[1]):
            states.append(torch.maximum(level[:, frame], states[-1] * alpha))
        state = torch.stack(states, dim=1)
        ratio = level / state.clamp_min(1.0e-8)
        tail = torch.sigmoid((ratio_threshold - ratio) / 0.08)
        active = state >= state.amax(dim=1, keepdim=True) * 0.03
        band_gain = 1.0 - strength * tail * active
        gain[:, selected] = band_gain[:, None, :]
    restored = torch.istft(
        spectrum * gain, n_fft, hop, window=window,
        center=True, length=wet.shape[1],
    )
    return restored


def release_event_frames(audio: np.ndarray) -> int:
    value = np.asarray(audio, dtype=np.float64)
    if value.ndim != 1 or len(value) < 480 or not np.isfinite(value).all():
        return 0
    count = 1 + (len(value) - 480) // 120
    frames = np.lib.stride_tricks.as_strided(
        value,
        shape=(count, 480),
        strides=(value.strides[0] * 120, value.strides[0]),
    )
    envelope = np.sqrt(np.mean(np.square(frames), axis=1) + 1.0e-12)
    active = envelope > max(float(np.quantile(envelope, 0.75)), 0.01 * float(np.max(envelope)))
    recent = np.convolve(active.astype(np.float64), np.ones(40), mode="full")[:len(envelope)] > 0
    quiet = envelope <= np.quantile(envelope, 0.35)
    return int(np.sum(quiet & recent))


class AmbiencePairsV4(AmbiencePairsV2):
    # This is intentionally conservative until an active-frame-only foreground
    # metric is frozen. It must not be mislabeled as foreground-only evidence.
    quality_contract = "tail-removal-with-global-nonregression"
    required_decay_strata = frozenset(("short-tail", "medium-tail", "long-tail"))
    def __init__(
        self,
        workspace: Path,
        split: str,
        samples: int,
        target_frames: int,
        seed: int,
        include_late_base: bool = True,
        rir_source_id: str = "but-reverbdb",
    ) -> None:
        if target_frames < TARGET_FRAMES_MINIMUM:
            raise ValueError(
                f"Product4 target must cover at least {TARGET_FRAMES_MINIMUM} frames"
            )
        super().__init__(
            workspace,
            split,
            samples,
            target_frames,
            seed,
            include_late_base=include_late_base,
            rir_source_id=rir_source_id,
            decay_domain=DECAY_DOMAIN_SECONDS,
            late_suppression_strength=0.375,
            clean_source_ids=PRODUCT4_CLEAN_SOURCE_IDS,
        )
        cells: dict[tuple[str, str], list[Path]] = {}
        if self.rir_buckets is None:
            raise ValueError("Product4 requires room-addressable RIR buckets")
        for room, paths in self.rir_buckets.items():
            for path in paths:
                stratum = self.decay_stratum(_rir_decay_seconds(_rir(path)))
                cells.setdefault((room, stratum), []).append(path)
        self.rir_cells = {
            key: tuple(paths) for key, paths in sorted(cells.items()) if paths
        }
        if set(stratum for _, stratum in self.rir_cells) != set(self.required_decay_strata):
            raise ValueError(f"Product4 split {split} does not cover every decay stratum")

    def _rir_selection(self, index: int) -> Path:
        cells = tuple(self.rir_cells)
        source_index = index % len(self.source_ids)
        cell = cells[(index // len(self.source_ids)) % len(cells)]
        rows = self.rir_cells[cell]
        cycle = index // (len(self.source_ids) * len(cells))
        return rows[(cycle * 104729 + source_index * 15485863 + self.seed) % len(rows)]

    def _select_clean_audio(self, index: int, seed: int):
        from .product_data import _read

        source = self.source_ids[index % len(self.source_ids)]
        rows = self.buckets[source]
        cycle = index // len(self.source_ids)
        start = (cycle * 104729 + self.seed) % len(rows)
        best_score = -1
        searched = min(len(rows), 64)
        for row_offset in range(searched):
            selected = rows[(start + row_offset) % len(rows)]
            for crop_attempt in range(8):
                value = _read(
                    selected,
                    self.total_frames,
                    seed + row_offset * 32452843 + crop_attempt * 49979687,
                )
                score = release_event_frames(value[self.history_frames:])
                best_score = max(best_score, score)
                if score >= 4:
                    return selected, value
        raise ValueError(
            f"Product4 could not find a measurable release event for {source}; "
            f"searched_files={searched} best_frames={best_score}"
        )

    @staticmethod
    def decay_stratum(seconds: float) -> str:
        if seconds < DECAY_STRATA_SECONDS[0]:
            return "short-tail"
        if seconds < DECAY_STRATA_SECONDS[1]:
            return "medium-tail"
        return "long-tail"

    @staticmethod
    def room_group(row: dict) -> str:
        room, separator, _ = str(row["rir"]).partition("/")
        if not separator or not room:
            raise ValueError("Product4 RIR must retain its room-relative path")
        return room


class AmbiencePairsV4WetBase(AmbiencePairsV4):
    """Product4 ablation initialized from Wet instead of blind suppression."""

    def __init__(
        self,
        workspace: Path,
        split: str,
        samples: int,
        target_frames: int,
        seed: int,
    ) -> None:
        super().__init__(
            workspace,
            split,
            samples,
            target_frames,
            seed,
            include_late_base=False,
        )


class AmbiencePairsV4DecayBank(AmbiencePairsV4):
    """Product4 pairs carrying fixed Wet-only exponential-decay candidates."""

    candidate_variants = DECAY_BANK_VARIANTS

    def __init__(
        self,
        workspace: Path,
        split: str,
        samples: int,
        target_frames: int,
        seed: int,
    ) -> None:
        super().__init__(
            workspace, split, samples, target_frames, seed, include_late_base=False
        )

    def __getitem__(self, index: int) -> dict:
        row = super().__getitem__(index)
        if row["late_base"].ndim == 2:
            return row
        with torch.inference_mode():
            candidates = [
                multiband_decay_suppression(
                    row["wet"].unsqueeze(0),
                    half_life_ms=half_life_ms,
                    ratio_threshold=ratio_threshold,
                    strength=strength,
                )[0]
                for half_life_ms, ratio_threshold, strength in self.candidate_variants
            ]
        result = {**row, "late_base": torch.stack(candidates)}
        self._cache[index] = result
        return result


class AmbiencePairsV4ProfileBank(AmbiencePairsV4):
    """Known-profile physical candidates for a bounded gray-box selector."""

    candidate_strengths = PROFILE_SHAPING_STRENGTHS

    def __init__(
        self,
        workspace: Path,
        split: str,
        samples: int,
        target_frames: int,
        seed: int,
    ) -> None:
        super().__init__(
            workspace, split, samples, target_frames, seed, include_late_base=False
        )

    def __getitem__(self, index: int) -> dict:
        from .ambience2 import AmbienceAbstention, restore_known_ambience_profile
        from .ambience4_shaping import profile_shortening_inverse

        row = super().__getitem__(index)
        if row["late_base"].ndim == 2:
            return row
        impulse = _rir(self.rir_root / row["rir"])
        wet = row["wet"].numpy()
        try:
            exact, report = restore_known_ambience_profile(
                wet, impulse, row["control_values"]
            )
            candidates = [exact] * len(self.candidate_strengths)
            profile_mode = "exact"
        except AmbienceAbstention as error:
            candidates = [
                profile_shortening_inverse(
                    wet,
                    impulse,
                    row["control_values"],
                    early_ms=25.0,
                    target_rt60_ms=40.0,
                    maximum_gain=4.0,
                    low_band_strength=strength,
                    high_band_strength=0.0,
                    low_band_end_hz=2_000.0,
                    high_band_start_hz=3_000.0,
                )[0]
                for strength in self.candidate_strengths
            ]
            report = {
                "implementation": "bounded-profile-shortening-bank",
                "exact_abstention": str(error),
                "candidate_strengths": list(self.candidate_strengths),
            }
            profile_mode = "shaping-bank"
        result = {
            **row,
            "late_base": torch.from_numpy(np.stack(candidates).astype(np.float32)),
            "profile_mode": profile_mode,
            "profile_report": report,
        }
        self._cache[index] = result
        return result


class AmbiencePairsV4FrequencyProfileBank(AmbiencePairsV4):
    """Known-profile frequency-decay candidates for a bounded selector."""

    def __init__(
        self,
        workspace: Path,
        split: str,
        samples: int,
        target_frames: int,
        seed: int,
        rir_source_id: str = "but-reverbdb",
    ) -> None:
        from .ambience4_frequency_shaping import FREQUENCY_SHAPING_CANDIDATES

        super().__init__(
            workspace, split, samples, target_frames, seed,
            include_late_base=False, rir_source_id=rir_source_id,
        )
        self.candidate_variants = FREQUENCY_SHAPING_CANDIDATES

    def __getitem__(self, index: int) -> dict:
        from .ambience2 import AmbienceAbstention, restore_known_ambience_profile
        from .ambience4_frequency_shaping import profile_frequency_shortening_inverse

        row = super().__getitem__(index)
        if row["late_base"].ndim == 2:
            return row
        impulse = _rir(self.rir_root / row["rir"])
        wet = row["wet"].numpy()
        abstentions = []
        try:
            exact, report = restore_known_ambience_profile(
                wet, impulse, row["control_values"]
            )
            candidates = [exact] * len(self.candidate_variants)
            profile_mode = "exact"
        except AmbienceAbstention as exact_error:
            candidates = []
            for candidate in self.candidate_variants:
                parameters = {
                    key: value for key, value in candidate.items() if key != "id"
                }
                try:
                    restored, _ = profile_frequency_shortening_inverse(
                        wet, impulse, row["control_values"], **parameters
                    )
                except AmbienceAbstention as error:
                    restored = wet.copy()
                    abstentions.append({"candidate": candidate["id"], "reason": str(error)})
                candidates.append(restored)
            report = {
                "implementation": "bounded-frequency-bin-faded-profile-bank",
                "exact_abstention": str(exact_error),
                "candidate_ids": [row["id"] for row in self.candidate_variants],
                "candidate_abstentions": abstentions,
            }
            profile_mode = "frequency-shaping-bank"
        result = {
            **row,
            "late_base": torch.from_numpy(np.stack(candidates).astype(np.float32)),
            "profile_mode": profile_mode,
            "profile_report": report,
        }
        self._cache[index] = result
        return result


class AmbiencePairsV4OpenSLR26FrequencyProfileBank(AmbiencePairsV4FrequencyProfileBank):
    """Room-disjoint OpenSLR 26 product pairs for selector generalization."""

    def __init__(self, workspace: Path, split: str, samples: int, target_frames: int, seed: int) -> None:
        super().__init__(
            workspace, split, samples, target_frames, seed,
            rir_source_id="openslr26-simulated-rir-external-v1",
        )

    @staticmethod
    def room_group(row: dict) -> str:
        parts = Path(str(row["rir"])).parts
        if len(parts) < 3:
            raise ValueError("OpenSLR 26 RIR must retain category and room path")
        return "/".join(parts[:2])


class AmbiencePairsV4MixedFrequencyProfileBank(torch.utils.data.Dataset):
    """Balanced real-BUT and simulated-OpenSLR26 room-disjoint profile bank."""

    quality_contract = AmbiencePairsV4.quality_contract
    required_decay_strata = AmbiencePairsV4.required_decay_strata
    decay_stratum = staticmethod(AmbiencePairsV4.decay_stratum)

    def __init__(self, workspace: Path, split: str, samples: int, target_frames: int, seed: int) -> None:
        but_samples = (samples + 1) // 2
        simulated_samples = samples // 2
        if not simulated_samples:
            raise ValueError("mixed Product4 dataset needs at least two samples")
        self.but = AmbiencePairsV4FrequencyProfileBank(
            workspace, split, but_samples, target_frames, seed
        )
        self.simulated = AmbiencePairsV4OpenSLR26FrequencyProfileBank(
            workspace, split, simulated_samples, target_frames, seed + 15_485_863
        )
        self.workspace = workspace.resolve()
        self.split = split
        self.samples = samples
        self.target_frames = target_frames
        self.history_frames = self.but.history_frames
        self.total_frames = self.but.total_frames
        sources = sorted(set(self.but.authorization["sources"]) | set(self.simulated.authorization["sources"]))
        attributions = {
            row["source_id"]: row
            for row in self.but.authorization["required_attribution"]
            + self.simulated.authorization["required_attribution"]
        }
        self.authorization = {
            "authorized": True,
            "sources": sources,
            "required_attribution": [attributions[key] for key in sorted(attributions)],
        }

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict:
        if index % 2 == 0:
            return {**self.but[index // 2], "rir_source_id": "but-reverbdb"}
        return {
            **self.simulated[index // 2],
            "rir_source_id": "openslr26-simulated-rir-external-v1",
        }

    def realized_source_counts(self) -> dict[str, int]:
        return dict(Counter(
            self[index]["source_id"] for index in range(len(self))
        ))

    def realized_rir_counts(self) -> dict[str, int]:
        return dict(Counter(
            f"{self[index]['rir_source_id']}:{self[index]['rir']}"
            for index in range(len(self))
        ))

    @staticmethod
    def room_group(row: dict) -> str:
        source = str(row["rir_source_id"])
        parts = Path(str(row["rir"])).parts
        # Hundreds of simulated rooms remain physically split-disjoint, while
        # quality aggregation uses their three frozen size categories so every
        # reported group has enough temporal-tail examples for a stable gate.
        room = parts[0]
        return f"{source}:{room}"
