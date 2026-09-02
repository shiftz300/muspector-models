"""Feature and runtime contract for chain family-set gating."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import joblib
import numpy as np

from .order_search import rank_topologies
from .spec import KINDS


SUBSETS = tuple(
    subset for size in range(len(KINDS) + 1) for subset in itertools.combinations(KINDS, size)
)
MASKS = tuple(sum(1 << KINDS.index(name) for name in subset) for subset in SUBSETS)
CLASSES = tuple(range(1 << len(KINDS)))
FEATURES = (
    tuple(f"family.score.{name}" for name in KINDS)
    + tuple(f"family.threshold.{name}" for name in KINDS)
    + tuple(f"family.margin.{name}" for name in KINDS)
    + ("family.reverb_verifier",)
    + tuple(f"error.log.{mask}" for mask in MASKS)
    + tuple(f"error.delta.{mask}" for mask in MASKS)
    + tuple(f"control.{index}" for index in range(9))
    + ("pair.rms_ratio", "pair.residual_ratio", "pair.correlation", "pair.peak_ratio")
)


def mask(active) -> int:
    return sum(1 << KINDS.index(name) for name in active)


def active(value: int) -> list[str]:
    return [name for index, name in enumerate(KINDS) if value & (1 << index)]


def vector(family: dict, remix: dict, dry: np.ndarray, wet: np.ndarray, runtime) -> np.ndarray:
    controls = np.asarray(remix["normalized_controls"], dtype=np.float64)
    errors = np.asarray(
        [
            rank_topologies(
                dry,
                wet,
                subset,
                controls,
                runtime.sample_rate,
                tuple(runtime.order_search["renderers"]),
            )[0][1]
            for subset in SUBSETS
        ],
        dtype=np.float64,
    )
    floor = max(float(errors.min()), 1.0e-8)
    dry_rms = max(float(np.sqrt(np.mean(dry * dry))), 1.0e-8)
    wet_rms = float(np.sqrt(np.mean(wet * wet)))
    residual = float(np.sqrt(np.mean((wet - dry) ** 2)))
    correlation = float(np.corrcoef(dry, wet)[0, 1])
    if not np.isfinite(correlation):
        correlation = 0.0
    values = (
        [family["scores"][name] for name in KINDS]
        + [family["thresholds"][name] for name in KINDS]
        + [family["margins"][name] for name in KINDS]
        + [family["reverb_verifier"]]
        + np.log(errors + 1.0e-8).tolist()
        + ((errors - floor) / floor).tolist()
        + controls.tolist()
        + [
            wet_rms / dry_rms,
            residual / dry_rms,
            correlation,
            float(np.max(np.abs(wet))) / max(float(np.max(np.abs(dry))), 1.0e-8),
        ]
    )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(FEATURES),) or not np.isfinite(result).all():
        raise ValueError("chain gate feature contract changed")
    return result


class GateRuntime:
    def __init__(self, artifact: Path):
        self.path = Path(artifact).resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        payload = joblib.load(self.path)
        if payload.get("schema") != 1 or tuple(payload.get("features", ())) != FEATURES:
            raise ValueError("unsupported chain gate")
        if tuple(payload.get("classes", ())) != CLASSES:
            raise ValueError("chain gate class contract changed")
        self.model = payload["model"]
        if "thresholds" in payload:
            self.thresholds = {int(name): float(value) for name, value in payload["thresholds"].items()}
            if set(self.thresholds) != set(CLASSES):
                raise ValueError("chain gate threshold contract changed")
        else:
            threshold = float(payload["threshold"])
            self.thresholds = {value: threshold for value in CLASSES}

    def infer(self, features: np.ndarray) -> dict:
        probabilities = self.model.predict_proba(np.asarray(features, dtype=np.float32)[None])[0]
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        value = int(self.model.classes_[index])
        threshold = self.thresholds[value]
        return {
            "active": active(value),
            "mask": value,
            "confidence": confidence,
            "threshold": threshold,
            "accepted": confidence >= threshold,
            "probabilities": {str(int(label)): float(probability) for label, probability in zip(self.model.classes_, probabilities)},
        }

    def assert_artifact_unchanged(self) -> None:
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256:
            raise ValueError("chain gate changed")
