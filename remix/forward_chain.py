"""Unified loss-preserving forward runtime for complete Remix effect chains.

This module deliberately has no audio-device or file-playback code.  It accepts
finite mono float32 analysis copies and applies the frozen Drive, Delay and
capture-fitted Reverb renderers in the exact order described by ``ChainSpec``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .forward_delay import (
    DelayForwardRenderer,
    DelayStreamState,
    normalized_delay_controls,
    stream_delay_block,
)
from .drive_adapter import CalibratedDriveRenderer, DriveAdapterState, load_drive_adapter
from .forward_drive import FORWARD_RATE, DriveForwardRenderer, normalized_drive_controls
from .forward_reverb import (
    ReverbDeviceProfile,
    ReverbStreamState,
    load_reverb_profile,
    stream_reverb_block,
)
from .quality import checked_audio, validate_render
from .spec import ChainSpec, Delay, Drive, Effect, Reverb


ROOT = Path(__file__).resolve().parents[1]
CLIENT = Path(os.environ.get("MUSPECTOR_CLIENT", ROOT.parent / "muspector")).resolve()
LEGACY = CLIENT / "remix/runs"
DRIVE_CHECKPOINT = LEGACY / "drive-forward-pilot/drive-forward-candidate.pt"
DELAY_CHECKPOINT = LEGACY / "delay-forward-pilot/delay-forward-candidate.pt"
REVERB_PROFILE = LEGACY / "reverb-forward-phase1/reverb-device-profile.npz"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_drive(path: Path) -> DriveForwardRenderer:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != 1
        or payload.get("sample_rate") != FORWARD_RATE
        or tuple(payload.get("controls", ())) != ("gain_db", "tone", "level_db")
    ):
        raise ValueError("incompatible Drive forward checkpoint")
    model = DriveForwardRenderer(int(payload["hidden_size"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def _load_delay(path: Path) -> DelayForwardRenderer:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != 1
        or payload.get("sample_rate") != FORWARD_RATE
        or tuple(payload.get("controls", ())) != ("time_ms", "feedback", "mix")
    ):
        raise ValueError("incompatible Delay forward checkpoint")
    model = DelayForwardRenderer(int(payload["fir_taps"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


@dataclass(frozen=True)
class ForwardChainState:
    """State for one fixed chain; entries follow the exact effect order."""

    spec: ChainSpec
    effect_states: tuple[object | None, ...]


class ForwardChainRuntime:
    """Compose the three accepted forward models without hidden audio changes."""

    def __init__(
        self,
        drive_checkpoint: Path = DRIVE_CHECKPOINT,
        delay_checkpoint: Path = DELAY_CHECKPOINT,
        reverb_profile: Path | ReverbDeviceProfile = REVERB_PROFILE,
        drive_adapter: Path | None = None,
    ) -> None:
        self.drive_checkpoint = Path(drive_checkpoint)
        self.delay_checkpoint = Path(delay_checkpoint)
        self.reverb_profile_path = (
            Path(reverb_profile) if isinstance(reverb_profile, (str, Path)) else None
        )
        self.drive = _load_drive(self.drive_checkpoint)
        self.drive_adapter_path = Path(drive_adapter) if drive_adapter is not None else None
        self.calibrated_drive = (
            CalibratedDriveRenderer(self.drive, load_drive_adapter(self.drive_adapter_path)).eval()
            if self.drive_adapter_path is not None
            else None
        )
        self.delay = _load_delay(self.delay_checkpoint)
        self.reverb = (
            load_reverb_profile(self.reverb_profile_path)
            if self.reverb_profile_path is not None
            else reverb_profile
        )
        self.reverb.validate()
        if self.reverb.sample_rate != FORWARD_RATE:
            raise ValueError("Reverb profile sample rate is incompatible")

    def artifact_hashes(self) -> dict[str, str]:
        result = {
            "drive_checkpoint": sha256(self.drive_checkpoint),
            "delay_checkpoint": sha256(self.delay_checkpoint),
        }
        if self.reverb_profile_path is not None:
            result["reverb_profile"] = sha256(self.reverb_profile_path)
        else:
            result["reverb_calibration"] = self.reverb.calibration_hash
        if self.drive_adapter_path is not None:
            result["drive_adapter"] = sha256(self.drive_adapter_path)
        return result

    @staticmethod
    def _drive_controls(effect: Drive) -> torch.Tensor:
        return torch.from_numpy(normalized_drive_controls(effect)).unsqueeze(0)

    @staticmethod
    def _delay_controls(effect: Delay) -> torch.Tensor:
        return torch.from_numpy(normalized_delay_controls(effect)).unsqueeze(0)

    def render(self, dry: np.ndarray, spec: ChainSpec) -> np.ndarray:
        """Render a complete mono buffer. Empty chains are bit-exact bypass."""

        spec.validate()
        source = checked_audio(np.asarray(dry, dtype=np.float32), name="forward chain input")
        if source.ndim != 1:
            raise ValueError("forward chain runtime expects mono [frames] audio")
        if not spec.effects:
            return source.copy()
        value = source.copy()
        with torch.inference_mode():
            for effect in spec.effects:
                if isinstance(effect, Drive):
                    renderer = self.drive if self.calibrated_drive is None else self.calibrated_drive
                    rendered, _ = renderer(
                        torch.from_numpy(value).unsqueeze(0), self._drive_controls(effect)
                    )
                    value = rendered.squeeze(0).numpy().copy()
                elif isinstance(effect, Delay):
                    rendered = self.delay(
                        torch.from_numpy(value).unsqueeze(0), self._delay_controls(effect)
                    )
                    value = rendered.squeeze(0).numpy().copy()
                elif isinstance(effect, Reverb):
                    value = self.reverb.render(value, effect)
                else:  # pragma: no cover - ChainSpec validates the union.
                    raise TypeError(f"unsupported forward effect: {effect!r}")
        validate_render(source, value)
        return np.asarray(value, dtype=np.float32)

    def stream_block(
        self,
        dry: np.ndarray,
        spec: ChainSpec,
        state: ForwardChainState | None = None,
    ) -> tuple[np.ndarray, ForwardChainState]:
        """Render one block while preserving state and the declared order."""

        spec.validate()
        source = checked_audio(np.asarray(dry, dtype=np.float32), name="forward stream block")
        if source.ndim != 1:
            raise ValueError("forward streaming expects mono [frames] blocks")
        if state is not None and state.spec != spec:
            raise ValueError("chain topology or controls changed while reusing stream state")
        previous = (None,) * len(spec.effects) if state is None else state.effect_states
        if len(previous) != len(spec.effects):
            raise ValueError("forward chain stream state has incompatible geometry")
        if not spec.effects:
            return source.copy(), ForwardChainState(spec, ())

        value = source.copy()
        next_states: list[object | None] = []
        with torch.inference_mode():
            for effect, effect_state in zip(spec.effects, previous, strict=True):
                if isinstance(effect, Drive):
                    renderer = self.drive if self.calibrated_drive is None else self.calibrated_drive
                    if self.calibrated_drive is not None and effect_state is not None and not isinstance(effect_state, DriveAdapterState):
                        raise ValueError("calibrated Drive stream state is incompatible")
                    tensor, next_state = renderer(
                        torch.from_numpy(value).unsqueeze(0),
                        self._drive_controls(effect),
                        effect_state,
                    )
                    value = tensor.squeeze(0).numpy().copy()
                elif isinstance(effect, Delay):
                    tensor, next_state = stream_delay_block(
                        self.delay,
                        torch.from_numpy(value).unsqueeze(0),
                        self._delay_controls(effect),
                        effect_state if isinstance(effect_state, DelayStreamState) else None,
                    )
                    value = tensor.squeeze(0).numpy().copy()
                elif isinstance(effect, Reverb):
                    value, next_state = stream_reverb_block(
                        self.reverb,
                        value,
                        effect,
                        effect_state if isinstance(effect_state, ReverbStreamState) else None,
                    )
                else:  # pragma: no cover
                    raise TypeError(f"unsupported forward effect: {effect!r}")
                next_states.append(next_state)
        validate_render(source, value)
        return np.asarray(value, dtype=np.float32), ForwardChainState(spec, tuple(next_states))

    def stream(self, dry: np.ndarray, spec: ChainSpec, block_frames: int = 1_024) -> np.ndarray:
        """Convenience reference stream; native Reverb should use partitioned FIR."""

        if block_frames <= 0 or block_frames > 1_920:
            raise ValueError("stream block must be within [1, 1920] frames")
        source = checked_audio(np.asarray(dry, dtype=np.float32), name="forward stream")
        state = None
        pieces = []
        for start in range(0, len(source), block_frames):
            piece, state = self.stream_block(source[start : start + block_frames], spec, state)
            pieces.append(piece)
        return np.concatenate(pieces) if pieces else source.copy()
