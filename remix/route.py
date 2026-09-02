"""Small device-routing head layered on frozen identity scores."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np

from .identity import IdentityRuntime, LABELS
from .packages import digest, verify
from .paired_transfer_features import paired_transfer_features


PAIR = 1_882
FEATURES = (*LABELS, "known", "entropy", *(f"pair.{index}" for index in range(PAIR)))


def features(
    report: dict, dry: np.ndarray, wet: np.ndarray, sample_rate: int
) -> np.ndarray:
    scores = np.asarray([report["scores"][label] for label in LABELS], dtype=np.float64)
    known = float(scores.sum())
    probability = scores / max(known, 1.0e-12)
    entropy = float(-np.sum(probability * np.log(np.maximum(probability, 1.0e-12))))
    paired = paired_transfer_features(dry, wet, sample_rate)
    if paired.shape != (PAIR,):
        raise ValueError("paired route feature contract changed")
    result = np.concatenate((scores, np.asarray((known, entropy)), paired)).astype(np.float32)
    if result.shape != (len(FEATURES),) or not np.isfinite(result).all():
        raise ValueError("route feature contract changed")
    return result


class RouteRuntime:
    """Classify one admitted Drive as RAT or unknown without delivering controls."""

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
            raise ValueError("route artifact contract changed")
        self.identity = identity
        self.model = payload["model"]
        self.threshold = float(payload["threshold"])
        self.hash = digest(self.path)

    @classmethod
    def package(cls, root: Path, identity: IdentityRuntime) -> "RouteRuntime":
        root = Path(root).resolve()
        package = verify(root)
        if package["id"] != "route" or package["version"] != "1.0.0":
            raise ValueError("unsupported route package")
        roles = {row["role"]: root / row["path"] for row in package["artifacts"]}
        if set(roles) != {"head", "fit", "development", "runtime", "cycle"}:
            raise ValueError("route package artifact contract changed")
        runtime = cls(roles["head"], identity)
        runtime.root = root
        runtime.package_hash = digest(root / "package.json")
        runtime.paths = roles
        runtime.hashes = {path: digest(path) for path in roles.values()}
        return runtime

    def infer_pair(self, dry: np.ndarray, wet: np.ndarray, sample_rate: int) -> dict:
        before = hashlib.sha256(
            np.asarray(dry).tobytes() + np.asarray(wet).tobytes()
        ).hexdigest()
        base = self.identity.infer(wet, sample_rate)
        probability = float(
            self.model.predict_proba(features(base, dry, wet, sample_rate)[None])[0, 1]
        )
        accepted = probability >= self.threshold
        if before != hashlib.sha256(
            np.asarray(dry).tobytes() + np.asarray(wet).tobytes()
        ).hexdigest():
            raise ValueError("route runtime mutated source audio")
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
            "source_audio_modified": False,
            "physical_audio_devices_used": False,
        }

    def assert_artifacts_unchanged(self) -> None:
        self.identity.assert_artifacts_unchanged()
        if digest(self.path) != self.hash:
            raise ValueError("route artifact changed")
        if hasattr(self, "root"):
            if digest(self.root / "package.json") != self.package_hash:
                raise ValueError("route package changed")
            for path, expected in self.hashes.items():
                if digest(path) != expected:
                    raise ValueError(f"route package artifact changed: {path.name}")
