"""Content-reduced paired transfer signature for device-family routing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np

from .packages import digest
from .paired_transfer_features import paired_transfer_features
from .router import level


RATE = 48_000
FFT = 3
BANDS = 48
BLOCK = 582
PAIR = 1_882


def _names() -> tuple[str, ...]:
    names = []
    fields = ("ratio", "coherence", "transfer", "phase")
    for fft in (256, 1024, 4096):
        for view in ("raw", "centered"):
            names.extend(f"fft{fft}.{view}.{field}.{band}" for field in fields for band in range(BANDS))
        names.extend(f"fft{fft}.mean.{field}" for field in fields)
    names.extend(f"audio.wet-dry.{index}" for index in range(27))
    names.extend(f"audio.residual-dry.{index}" for index in range(27))
    for index in range(11):
        names.extend((f"onset.rms.{index}", f"onset.peak.{index}"))
    return tuple(names)


FEATURES = _names()


def select(paired: np.ndarray) -> np.ndarray:
    """Remove absolute Dry/Wet spectra while retaining their transfer relation."""

    source = np.asarray(paired, dtype=np.float32)
    if source.shape != (PAIR,) or not np.isfinite(source).all():
        raise ValueError("signature requires the frozen paired feature geometry")
    pieces = []
    offset = 0
    for _ in range(FFT):
        raw = source[offset:offset + 288].reshape(6, BANDS); offset += 288
        centered = source[offset:offset + 288].reshape(6, BANDS); offset += 288
        means = source[offset:offset + 6]; offset += 6
        pieces.extend((raw[2:].ravel(), centered[2:].ravel(), means[2:]))
    dry = source[offset:offset + 27]; offset += 27
    wet = source[offset:offset + 27]; offset += 27
    residual = source[offset:offset + 27]; offset += 27
    pieces.extend((wet - dry, residual - dry))
    onset = source[offset:].reshape(11, 5)
    pieces.extend((onset[:, 2], np.log(np.maximum(onset[:, 3], 1.0e-8) / np.maximum(onset[:, 4], 1.0e-8))))
    result = np.concatenate(pieces).astype(np.float32)
    if result.shape != (len(FEATURES),) or not np.isfinite(result).all():
        raise ValueError("signature feature contract changed")
    return result


def encode(dry: np.ndarray, wet: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != RATE:
        raise ValueError("signature requires 48 kHz analysis arrays")
    dry_value, wet_value = np.asarray(dry, dtype=np.float32), np.asarray(wet, dtype=np.float32)
    if dry_value.shape != wet_value.shape:
        raise ValueError("signature Dry/Wet geometry differs")
    before = hashlib.sha256(dry_value.tobytes() + wet_value.tobytes()).hexdigest()
    result = select(paired_transfer_features(level(dry_value), level(wet_value), RATE))
    if before != hashlib.sha256(dry_value.tobytes() + wet_value.tobytes()).hexdigest():
        raise ValueError("signature mutated source audio")
    return result


class SignatureRuntime:
    """Route a 48 kHz mono Dry/Wet pair using a grouped-validation artifact."""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        payload = joblib.load(self.path)
        if (payload.get("schema") != 2 or payload.get("device") != "rat"
                or tuple(payload.get("features", ())) != FEATURES
                or not 0.0 <= float(payload.get("threshold", -1.0)) <= 1.0):
            raise ValueError("signature artifact contract changed")
        self.model, self.threshold = payload["model"], float(payload["threshold"])
        self.hash = digest(self.path)

    def infer_pair(self, dry: np.ndarray, wet: np.ndarray, sample_rate: int) -> dict:
        probability = float(self.model.predict_proba(encode(dry, wet, sample_rate)[None])[0, 1])
        accepted = probability >= self.threshold
        return {
            "schema": 2, "decision": "candidate" if accepted else "abstain",
            "label": "RAT" if accepted else None, "candidate": "RAT",
            "score": probability, "threshold": self.threshold,
            "shadow": True, "automatic_delivery": False,
            "content_reduced": True, "analysis_gain_standardization": True,
            "source_audio_modified": False, "physical_audio_devices_used": False,
        }

    def assert_artifacts_unchanged(self) -> None:
        if digest(self.path) != self.hash:
            raise ValueError("signature artifact changed")
