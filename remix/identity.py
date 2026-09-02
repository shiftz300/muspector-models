"""Read-only pedal identity inference and shadow device routing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime
from scipy.signal import resample_poly

from .packages import digest, verify
from .quality import checked_audio


RATE = 48_000
FRAMES = 240_000
WINDOWS = 3
LABELS = (
    "BluesDriver",
    "RAT",
    "TubeScreamer",
    "BigMuff",
    "MetalMuff",
    "FuzzyLogic",
    "SillyFuzz",
)
ROUTES = {"RAT": "rat"}


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0))))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - float(np.max(values))
    result = np.exp(shifted)
    return result / result.sum()


def audio(audio: np.ndarray, source_rate: int) -> np.ndarray:
    """Create a mono 48 kHz analysis copy without changing the source."""

    value = checked_audio(audio, name="identity analysis copy")
    if value.ndim != 1:
        raise ValueError("identity requires mono analysis audio")
    if source_rate <= 0:
        raise ValueError("source sample rate must be positive")
    if source_rate != RATE:
        common = np.gcd(source_rate, RATE)
        value = resample_poly(value, RATE // common, source_rate // common).astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def starts(value: np.ndarray) -> list[int]:
    """Match the bounded energetic-window policy used by the Rust client."""

    if len(value) <= FRAMES:
        return [0]
    result = list(range(0, len(value) - FRAMES + 1, FRAMES // 2))
    final = len(value) - FRAMES
    if result[-1] != final:
        result.append(final)
    result.sort(
        key=lambda start: (-float(np.mean(np.square(value[start : start + FRAMES]))), start)
    )
    return sorted(result[:WINDOWS])


class IdentityRuntime:
    """Load the seven-pedal open-set classifier from a materialized package."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        package = verify(self.root)
        if (
            package["id"] != "identity"
            or package["version"] != "1.0.0"
            or package["sample_rate"] != RATE
        ):
            raise ValueError("unsupported identity package")
        roles = {row["role"]: self.root / row["path"] for row in package["artifacts"]}
        expected = {"identity_model", "identity_config", "license_notice"}
        if set(roles) != expected:
            raise ValueError("identity artifact contract changed")
        self.config = json.loads(roles["identity_config"].read_text())
        if (
            self.config.get("schema") != 1
            or tuple(self.config.get("labels", ())) != LABELS
            or self.config.get("model_sha256") != digest(roles["identity_model"])
            or self.config.get("license_scope") != "non-commercial research"
            or self.config.get("user_gradient_updates") != 0
        ):
            raise ValueError("identity configuration is invalid")
        self.threshold = float(self.config["threshold"])
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("identity threshold is invalid")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            roles["identity_model"], sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = [(item.name, item.shape, item.type) for item in self.session.get_inputs()]
        outputs = [(item.name, item.shape, item.type) for item in self.session.get_outputs()]
        if inputs != [("waveform_48khz", [1, 1, FRAMES], "tensor(float)")]:
            raise ValueError(f"identity input contract changed: {inputs}")
        if outputs != [
            ("identity_logits", [1, len(LABELS)], "tensor(float)"),
            ("known_logit", [1], "tensor(float)"),
        ]:
            raise ValueError(f"identity output contract changed: {outputs}")
        self.paths = roles
        self.hashes = {path: digest(path) for path in roles.values()}
        self.package_hash = digest(self.root / "package.json")

    def infer(self, source: np.ndarray, sample_rate: int) -> dict:
        before = hashlib.sha256(np.asarray(source).tobytes()).hexdigest()
        value = audio(source, sample_rate)
        combined = np.zeros(len(LABELS), dtype=np.float64)
        selected = starts(value)
        for start in selected:
            window = np.zeros(FRAMES, dtype=np.float32)
            available = min(len(value) - start, FRAMES)
            window[:available] = value[start : start + available]
            logits, known = self.session.run(None, {"waveform_48khz": window[None, None]})
            combined += _softmax(logits[0]) * _sigmoid(float(known[0]))
        combined /= len(selected)
        index = int(np.argmax(combined))
        score = float(combined[index])
        accepted = score >= self.threshold
        if before != hashlib.sha256(np.asarray(source).tobytes()).hexdigest():
            raise ValueError("identity runtime mutated source audio")
        return {
            "schema": 1,
            "decision": "candidate" if accepted else "abstain",
            "label": LABELS[index] if accepted else None,
            "candidate": LABELS[index],
            "score": score,
            "scores": {label: float(combined[position]) for position, label in enumerate(LABELS)},
            "threshold": self.threshold,
            "windows": len(selected),
            "shadow": True,
            "automatic_delivery": False,
            "package_sha256": self.package_hash,
            "source_audio_modified": False,
            "physical_audio_devices_used": False,
        }

    def infer_pair(self, dry: np.ndarray, wet: np.ndarray, sample_rate: int) -> dict:
        """Identity baseline ignores Dry while preserving the pair runtime interface."""

        if np.asarray(dry).shape != np.asarray(wet).shape:
            raise ValueError("identity pair geometry differs")
        return self.infer(wet, sample_rate)

    def assert_artifacts_unchanged(self) -> None:
        if digest(self.root / "package.json") != self.package_hash:
            raise ValueError("identity package changed")
        for path, expected in self.hashes.items():
            if digest(path) != expected:
                raise ValueError(f"identity artifact changed: {path.name}")


class Router:
    """Produce a device candidate without authorizing adapter delivery."""

    def __init__(self, identity: IdentityRuntime):
        self.identity = identity

    def infer(
        self, chain: dict, dry: np.ndarray, wet: np.ndarray, sample_rate: int
    ) -> dict:
        if chain.get("decision") != "accepted" or chain.get("active") != ["drive"]:
            return {
                "schema": 1,
                "decision": "abstain",
                "reason": "chain-ineligible",
                "device": None,
                "identity": None,
                "shadow": True,
                "automatic_delivery": False,
            }
        identity = self.identity.infer_pair(dry, wet, sample_rate)
        device = ROUTES.get(identity["label"])
        return {
            "schema": 1,
            "decision": "candidate" if device is not None else "abstain",
            "reason": None if device is not None else "identity-unmapped",
            "device": device,
            "identity": identity,
            "shadow": True,
            "automatic_delivery": False,
        }
