"""Gain-robust RAT shadow router layered on frozen identity evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np

from .identity import IdentityRuntime, LABELS
from .packages import digest, verify
from .route import features


FEATURES = (
    *(f"level.{label}" for label in LABELS),
    "level.known",
    "level.entropy",
    *(f"level.pair.{index}" for index in range(1_882)),
)


def level(audio: np.ndarray) -> np.ndarray:
    """Remove device Volume from a private analysis copy."""

    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError("level analysis requires finite nonempty mono audio")
    scale = max(float(np.sqrt(np.mean(value * value))), 1.0e-5)
    return value / scale * 0.05


def encode(
    identity: IdentityRuntime,
    report: dict,
    dry: np.ndarray,
    wet: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    dry_level, wet_level = level(dry), level(wet)
    level_report = identity.infer(wet_level, sample_rate)
    result = features(level_report, dry_level, wet_level, sample_rate)
    if result.shape != (len(FEATURES),) or not np.isfinite(result).all():
        raise ValueError("router feature contract changed")
    return result


class RouterRuntime:
    """Classify one admitted Drive as RAT or unknown without delivery."""

    def __init__(self, model: Path, identity: IdentityRuntime):
        self.path = Path(model).resolve()
        payload = joblib.load(self.path)
        if (
            payload.get("schema") != 1
            or payload.get("device") != "rat"
            or tuple(payload.get("features", ())) != FEATURES
            or payload.get("identity_package_sha256") != identity.package_hash
            or not 0.0 <= float(payload.get("threshold", -1.0)) <= 1.0
        ):
            raise ValueError("router artifact contract changed")
        self.identity = identity
        self.model = payload["model"]
        self.threshold = float(payload["threshold"])
        self.hash = digest(self.path)

    @classmethod
    def package(cls, root: Path, identity: IdentityRuntime) -> "RouterRuntime":
        root = Path(root).resolve()
        package = verify(root)
        if package["id"] != "router" or package["version"] != "1.0.0":
            raise ValueError("unsupported router package")
        roles = {row["role"]: root / row["path"] for row in package["artifacts"]}
        if set(roles) != {"head", "fit", "development", "runtime", "cycle"}:
            raise ValueError("router package artifact contract changed")
        runtime = cls(roles["head"], identity)
        runtime.root = root
        runtime.package_hash = digest(root / "package.json")
        runtime.paths = roles
        runtime.hashes = {path: digest(path) for path in roles.values()}
        return runtime

    def infer_pair(self, dry: np.ndarray, wet: np.ndarray, sample_rate: int) -> dict:
        before = hashlib.sha256(np.asarray(dry).tobytes() + np.asarray(wet).tobytes()).hexdigest()
        base = self.identity.infer(wet, sample_rate)
        probability = float(self.model.predict_proba(encode(self.identity, base, dry, wet, sample_rate)[None])[0, 1])
        accepted = probability >= self.threshold
        if before != hashlib.sha256(np.asarray(dry).tobytes() + np.asarray(wet).tobytes()).hexdigest():
            raise ValueError("router runtime mutated source audio")
        return {
            "schema": 1,
            "decision": "candidate" if accepted else "abstain",
            "label": "RAT" if accepted else None,
            "candidate": "RAT",
            "score": probability,
            "threshold": self.threshold,
            "base": base,
            "shadow": True,
            "automatic_delivery": False,
            "analysis_gain_standardization": True,
            "source_audio_modified": False,
            "physical_audio_devices_used": False,
        }

    def assert_artifacts_unchanged(self) -> None:
        self.identity.assert_artifacts_unchanged()
        if digest(self.path) != self.hash:
            raise ValueError("router artifact changed")
        if hasattr(self, "root"):
            if digest(self.root / "package.json") != self.package_hash:
                raise ValueError("router package changed")
            for path, expected in self.hashes.items():
                if digest(path) != expected:
                    raise ValueError(f"router package artifact changed: {path.name}")
