"""Product4 profile tensors for a direct, order-independent Reverb gray box."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .ambience2 import AmbienceAbstention, restore_known_ambience_profile
from .ambience3 import _transfer
from .ambience3 import regularized_profile_inverse
from .ambience4 import AmbiencePairsV4
from .ambience4_frequency_shaping import _frequency_decay_seconds
from .product_data import _rir


PROFILE_N_FFT = 1_024
PROFILE_CHANNELS = 4


def profile_response_features(
    impulse: np.ndarray,
    controls: dict,
    *,
    n_fft: int = PROFILE_N_FFT,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode the exact current transfer and measured frequency-decay profile."""
    if n_fft < 512 or n_fft % 2:
        raise ValueError("profile response FFT must be even and at least 512")
    transfer = _transfer(impulse, controls).astype(np.float32)
    # Folding computes the long FIR's response exactly at the model STFT bins;
    # unlike np.fft.rfft(..., n=n_fft), it does not silently truncate the RIR.
    folded = np.zeros(n_fft, dtype=np.float64)
    np.add.at(folded, np.arange(len(transfer)) % n_fft, transfer)
    response = np.fft.rfft(folded)
    magnitude = np.maximum(np.abs(response), 1.0e-8)
    unit = response / magnitude
    _, _, _, decay = _frequency_decay_seconds(impulse, n_fft=n_fft)
    features = np.stack((
        np.clip(np.log(magnitude), -4.0, 4.0) / 4.0,
        unit.real,
        unit.imag,
        np.clip((decay - 0.08) / (1.50 - 0.08), 0.0, 1.0) * 2.0 - 1.0,
    )).astype(np.float32)
    if features.shape != (PROFILE_CHANNELS, n_fft // 2 + 1):
        raise ValueError("profile feature geometry changed")
    if not np.isfinite(features).all() or not np.isfinite(transfer).all():
        raise ValueError("profile encoding produced non-finite values")
    return features, transfer


class AmbiencePairsV4ProfileDirect(AmbiencePairsV4):
    """Known-profile pairs without a fixed restoration-candidate bank."""

    def __init__(
        self,
        workspace: Path,
        split: str,
        samples: int,
        target_frames: int,
        seed: int,
        rir_source_id: str = "but-reverbdb",
    ) -> None:
        super().__init__(
            workspace,
            split,
            samples,
            target_frames,
            seed,
            include_late_base=False,
            rir_source_id=rir_source_id,
        )

    def __getitem__(self, index: int) -> dict:
        row = super().__getitem__(index)
        if "profile_features" in row:
            return row
        impulse = _rir(self.rir_root / row["rir"])
        features, transfer = profile_response_features(impulse, row["control_values"])
        try:
            exact, report = restore_known_ambience_profile(
                row["wet"].numpy(), impulse, row["control_values"]
            )
            exact_available = True
            analytic = exact
            analytic_mode = "exact"
        except AmbienceAbstention as error:
            exact = row["wet"].numpy().copy()
            exact_available = False
            try:
                analytic, regularized_report = regularized_profile_inverse(
                    row["wet"].numpy(), impulse, row["control_values"]
                )
                analytic_mode = "regularized"
                report = {**regularized_report, "exact_abstention": str(error)}
            except AmbienceAbstention as regularized_error:
                analytic = row["wet"].numpy().copy()
                analytic_mode = "wet-fallback"
                report = {
                    "exact_abstention": str(error),
                    "regularized_abstention": str(regularized_error),
                }
        result = {
            **row,
            "profile_features": torch.from_numpy(features),
            "transfer": torch.from_numpy(transfer.copy()),
            "exact_base": torch.from_numpy(np.asarray(exact, dtype=np.float32).copy()),
            "exact_available": exact_available,
            "analytic_base": torch.from_numpy(np.asarray(analytic, dtype=np.float32).copy()),
            "analytic_mode": analytic_mode,
            "profile_report": report,
        }
        self._cache[index] = result
        return result


class AmbiencePairsV4OpenSLR26ProfileDirect(AmbiencePairsV4ProfileDirect):
    def __init__(self, workspace: Path, split: str, samples: int, target_frames: int, seed: int) -> None:
        super().__init__(
            workspace,
            split,
            samples,
            target_frames,
            seed,
            rir_source_id="openslr26-simulated-rir-external-v1",
        )

    @staticmethod
    def room_group(row: dict) -> str:
        parts = Path(str(row["rir"])).parts
        if len(parts) < 3:
            raise ValueError("OpenSLR 26 RIR must retain category and room path")
        return "/".join(parts[:2])


class AmbiencePairsV4MixedProfileDirect(torch.utils.data.Dataset):
    """Balanced measured/simulated profile-direct training partition."""

    required_decay_strata = AmbiencePairsV4.required_decay_strata
    decay_stratum = staticmethod(AmbiencePairsV4.decay_stratum)

    def __init__(self, workspace: Path, split: str, samples: int, target_frames: int, seed: int) -> None:
        but_samples = (samples + 1) // 2
        simulated_samples = samples // 2
        if not simulated_samples:
            raise ValueError("mixed profile-direct dataset needs at least two samples")
        self.but = AmbiencePairsV4ProfileDirect(
            workspace, split, but_samples, target_frames, seed
        )
        self.simulated = AmbiencePairsV4OpenSLR26ProfileDirect(
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
        return dict(Counter(self[index]["source_id"] for index in range(len(self))))

    def realized_rir_counts(self) -> dict[str, int]:
        return dict(Counter(
            f"{self[index]['rir_source_id']}:{self[index]['rir']}"
            for index in range(len(self))
        ))

    @staticmethod
    def room_group(row: dict) -> str:
        parts = Path(str(row["rir"])).parts
        return f"{row['rir_source_id']}:{parts[0]}"
