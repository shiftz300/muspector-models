"""Deterministic in-memory probes for classifying downloadable model files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import chirp

from .packages import digest
from .signature import FEATURES as SIGNATURE_FEATURES, RATE, encode


SECONDS = 5
COUNT = 4
FEATURES = SIGNATURE_FEATURES


def _rms(value: np.ndarray, target: float = 0.02) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return (value / max(float(np.sqrt(np.mean(value * value))), 1.0e-8) * target).astype(np.float32)


def probes() -> tuple[np.ndarray, ...]:
    """Return four bounded, reproducible analysis signals without file/device IO."""

    frames = SECONDS * RATE
    time = np.arange(frames, dtype=np.float32) / RATE
    rng = np.random.default_rng(20260901)
    frequencies = np.geomspace(43, 12_000, 48)
    phases = rng.uniform(0, 2 * np.pi, len(frequencies))
    multisine = sum(np.sin(2 * np.pi * frequency * time + phase) for frequency, phase in zip(frequencies, phases))
    sweep = chirp(time, f0=35, f1=16_000, t1=SECONDS, method="logarithmic")
    noise = rng.standard_normal(frames).astype(np.float32)
    envelope = np.tile(np.concatenate((np.linspace(0, 1, RATE // 10), np.ones(RATE // 10), np.linspace(1, 0, RATE // 10), np.zeros(RATE // 5))), 10)[:frames]
    bursts = noise * envelope
    plucks = np.zeros(frames, np.float32)
    for start, root in zip(range(0, frames, RATE), (82.4, 110.0, 146.8, 196.0, 246.9)):
        local = time[:RATE]
        value = sum(np.sin(2 * np.pi * root * harmonic * local) / harmonic for harmonic in range(1, 9))
        plucks[start:start + RATE] = value * np.exp(-local * 4.0)
    result = tuple(_rms(value) for value in (multisine, sweep, bursts, plucks))
    if any(value.shape != (frames,) or not np.isfinite(value).all() or np.max(np.abs(value)) > 0.2 for value in result):
        raise ValueError("probe geometry or safety bound changed")
    return result


def hashes() -> tuple[str, ...]:
    return tuple(hashlib.sha256(value.tobytes()).hexdigest() for value in probes())


class ProbeRuntime:
    """Classify a software model by rendering fixed probes entirely in memory."""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        payload = joblib.load(self.path)
        if (payload.get("schema") != 1 or payload.get("device") != "rat"
                or tuple(payload.get("features", ())) != FEATURES
                or tuple(payload.get("probe_sha256", ())) != hashes()):
            raise ValueError("probe artifact contract changed")
        self.model, self.threshold = payload["model"], float(payload["threshold"])
        self.hash = digest(self.path)

    def infer_model(self, renderer) -> dict:
        scores = []
        for dry in probes():
            before = hashlib.sha256(dry.tobytes()).hexdigest()
            wet = renderer.render(dry)
            scores.append(float(self.model.predict_proba(encode(dry, wet, RATE)[None])[0, 1]))
            if before != hashlib.sha256(dry.tobytes()).hexdigest():
                raise ValueError("probe renderer mutated analysis input")
        score = float(np.median(scores)); accepted = score >= self.threshold
        return {"schema":1,"decision":"candidate" if accepted else "abstain",
                "label":"RAT" if accepted else None,"candidate":"RAT","score":score,
                "probe_scores":scores,"threshold":self.threshold,"software_models_only":True,
                "physical_audio_devices_used":False,"automatic_delivery":False}

    def assert_artifacts_unchanged(self) -> None:
        if digest(self.path) != self.hash: raise ValueError("probe artifact changed")
