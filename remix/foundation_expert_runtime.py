"""Packaged offline runtime for accepted order-independent foundation experts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .foundation_data import RATE
from .inverse3 import NonlinearInverseV3
from .packages import digest, verify
from .product2 import restore_dynamics, restore_echo
from .spectral3 import normalized_controls
from .spectral_model3 import SpectralInverseV3


PACKAGE_TO_MECHANISM = {
    "foundation-drive": "nonlinear",
    "foundation-dynamics": "dynamics",
    "foundation-echo": "echo",
    "foundation-spectral": "spectral",
}
CONTROL_KEYS = {
    "nonlinear": frozenset({"drive", "bias", "cutoff_hz", "level", "shape"}),
    "dynamics": frozenset(
        {"threshold_db", "ratio", "attack_ms", "release_ms", "makeup_db"}
    ),
    "echo": frozenset({"time_ms", "feedback", "mix"}),
    "spectral": frozenset(
        {"low_gain_db", "mid_gain_db", "mid_hz", "mid_q", "high_gain_db"}
    ),
}
CONTROL_BOUNDS = {
    "nonlinear": {
        "drive": (1.8, 8.0),
        "bias": (-0.12, 0.12),
        "cutoff_hz": (1_800.0, 11_000.0),
        "level": (0.45, 0.85),
        "shape": (0.0, 2.0),
    },
    "dynamics": {
        "threshold_db": (-36.0, -12.0),
        "ratio": (2.0, 10.0),
        "attack_ms": (0.5, 35.0),
        "release_ms": (40.0, 450.0),
        "makeup_db": (0.0, 8.0),
    },
    "echo": {
        "time_ms": (40.0, 650.0),
        "feedback": (0.1, 0.72),
        "mix": (0.15, 0.65),
    },
    "spectral": {
        "low_gain_db": (-8.0, 8.0),
        "mid_gain_db": (-8.0, 8.0),
        "mid_hz": (250.0, 5_200.0),
        "mid_q": (0.45, 3.5),
        "high_gain_db": (-8.0, 8.0),
    },
}


def _checked_audio(value: np.ndarray) -> np.ndarray:
    audio = np.asarray(value, dtype=np.float32)
    if audio.ndim != 1 or audio.size < 1 or not np.isfinite(audio).all():
        raise ValueError("foundation inverse expects finite mono audio")
    return audio.copy()


def _checked_controls(mechanism: str, controls: dict[str, Any]) -> dict[str, float]:
    if not isinstance(controls, dict) or set(controls) != CONTROL_KEYS[mechanism]:
        raise ValueError(f"{mechanism} controls differ from the sealed contract")
    result = {key: float(value) for key, value in controls.items()}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError(f"{mechanism} controls must be finite")
    for key, value in result.items():
        low, high = CONTROL_BOUNDS[mechanism][key]
        if not low <= value <= high:
            raise ValueError(f"{mechanism} control {key} is outside the admitted domain")
    if mechanism == "nonlinear" and result["shape"] not in {0.0, 1.0, 2.0}:
        raise ValueError("nonlinear shape must be the sealed tanh/atan/cubic enum")
    return result


class FoundationExpertRuntime:
    """Run exactly one expert from Wet plus its own controls and state."""

    def __init__(
        self,
        mechanism: str,
        checkpoint: Path | None = None,
    ) -> None:
        if mechanism not in CONTROL_KEYS:
            raise ValueError(f"unsupported foundation expert: {mechanism}")
        self.mechanism = mechanism
        self.checkpoint = Path(checkpoint).resolve() if checkpoint is not None else None
        self.model: SpectralInverseV3 | NonlinearInverseV3 | None = None
        if mechanism == "nonlinear":
            if self.checkpoint is None:
                raise ValueError("nonlinear expert requires its accepted checkpoint")
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            architecture = payload.get("architecture", {})
            if (
                payload.get("schema") != 2
                or payload.get("sample_rate") != RATE
                or architecture.get("schema") != 4
                or architecture.get("architecture")
                != "regularized-deeq-plus-analytic-shape-inverse-plus-local-tcn"
                or architecture.get("graph_order_input") is not False
                or architecture.get("neighbouring_effect_input") is not False
            ):
                raise ValueError("nonlinear checkpoint contract changed")
            self.model = NonlinearInverseV3(
                hidden_size=int(architecture["hidden_size"]),
                layers=int(architecture["layers"]),
            )
            self.model.load_state_dict(payload["state_dict"], strict=True)
            self.model.eval()
        if mechanism == "spectral":
            if self.checkpoint is None:
                raise ValueError("spectral expert requires its accepted checkpoint")
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            architecture = payload.get("architecture", {})
            if (
                payload.get("schema") != 1
                or payload.get("sample_rate") != RATE
                or architecture.get("architecture")
                != "bounded-analytic-three-band-eq-inverse-plus-local-tcn"
                or architecture.get("graph_order_input") is not False
                or architecture.get("neighbouring_effect_input") is not False
            ):
                raise ValueError("spectral checkpoint contract changed")
            self.model = SpectralInverseV3(
                channels=int(architecture["channels"]),
                depth=int(architecture["depth"]),
            )
            self.model.load_state_dict(payload["state_dict"], strict=True)
            self.model.eval()

    @classmethod
    def package(cls, root: Path) -> "FoundationExpertRuntime":
        root = Path(root).resolve()
        package = verify(root)
        mechanism = PACKAGE_TO_MECHANISM.get(package["id"])
        if (
            mechanism is None
            or package["version"] != "1.0.0"
            or package["kind"] != "inverse"
            or package["quality"] != "development"
            or package["audio_behavior"] != "loss_preserving_restore"
            or package["sample_rate"] != RATE
            or package["channels"] != 1
        ):
            raise ValueError("unsupported foundation restoration package")
        roles = {row["role"]: root / row["path"] for row in package["artifacts"]}
        expected = {"runtime", "development", "chain"}
        if mechanism in {"nonlinear", "spectral"}:
            expected.add("checkpoint")
        if set(roles) != expected:
            raise ValueError("foundation package artifact roles changed")
        runtime = cls(mechanism, roles.get("checkpoint"))
        runtime.root = root
        runtime.package_hash = digest(root / "package.json")
        runtime.paths = roles
        runtime.hashes = {path: digest(path) for path in roles.values()}
        return runtime

    def restore(
        self,
        wet: np.ndarray,
        controls: dict[str, Any],
        state: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        audio = _checked_audio(wet)
        values = _checked_controls(self.mechanism, controls)
        before = hashlib.sha256(audio.tobytes()).hexdigest()
        if self.mechanism == "nonlinear":
            if state not in (None, {}):
                raise ValueError("offline nonlinear inverse does not accept external state")
            assert isinstance(self.model, NonlinearInverseV3)
            normalized = np.asarray(
                (
                    (values["drive"] - 1.8) / (8.0 - 1.8),
                    (values["bias"] + 0.12) / 0.24,
                    np.log(values["cutoff_hz"] / 1_800.0) / np.log(11_000.0 / 1_800.0),
                    (values["level"] - 0.45) / 0.40,
                    values["shape"] / 2.0,
                ),
                dtype=np.float32,
            )
            with torch.inference_mode():
                restored_tensor, uncertainty_tensor, _ = self.model(
                    torch.from_numpy(audio[None]),
                    torch.from_numpy(normalized[None]),
                )
            restored = restored_tensor[0].numpy().astype(np.float32, copy=False)
            uncertainty = uncertainty_tensor[0].numpy().astype(np.float32, copy=False)
            next_state = None
        elif self.mechanism == "dynamics":
            if state is None:
                envelope = 0.0
            elif set(state) == {"envelope"} and float(state["envelope"]) >= 0.0:
                envelope = float(state["envelope"])
            else:
                raise ValueError("invalid dynamics state")
            restored, envelope = restore_dynamics(audio, values, envelope)
            next_state: dict[str, float] | None = {"envelope": envelope}
            uncertainty = None
        elif self.mechanism == "echo":
            if state not in (None, {}):
                raise ValueError("offline echo inverse does not accept external state")
            restored = restore_echo(audio, values)
            next_state = None
            uncertainty = None
        else:
            if state not in (None, {}):
                raise ValueError("offline spectral inverse does not accept external state")
            assert self.model is not None
            normalized = normalized_controls(values)
            with torch.inference_mode():
                restored_tensor, uncertainty_tensor = self.model(
                    torch.from_numpy(audio[None]),
                    torch.from_numpy(normalized[None]),
                )
            restored = restored_tensor[0].numpy().astype(np.float32, copy=False)
            uncertainty = uncertainty_tensor[0].numpy().astype(np.float32, copy=False)
            next_state = None
        if hashlib.sha256(audio.tobytes()).hexdigest() != before:
            raise ValueError("foundation inverse mutated its Wet input")
        if restored.shape != audio.shape or not np.isfinite(restored).all():
            raise ValueError("foundation inverse violated output geometry")
        return {
            "audio": restored.copy(),
            "mechanism": self.mechanism,
            "state": next_state,
            "uncertainty": None if uncertainty is None else uncertainty.copy(),
            "source_audio_modified": False,
        }

    def assert_artifacts_unchanged(self) -> None:
        if not hasattr(self, "root"):
            return
        if digest(self.root / "package.json") != self.package_hash:
            raise ValueError("foundation package changed")
        for path, expected in self.hashes.items():
            if digest(path) != expected:
                raise ValueError(f"foundation package artifact changed: {path.name}")


class FoundationChainRuntime:
    """Reverse a known graph while keeping every expert order-agnostic."""

    def __init__(self, experts: dict[str, FoundationExpertRuntime]) -> None:
        if not experts or not set(experts) <= set(CONTROL_KEYS):
            raise ValueError("foundation chain has an empty or unsupported expert registry")
        if any(expert.mechanism != name for name, expert in experts.items()):
            raise ValueError("foundation expert registry is inconsistent")
        self.experts = dict(experts)

    def restore(self, wet: np.ndarray, forward_stages: list[dict[str, Any]]) -> dict[str, Any]:
        current = _checked_audio(wet)
        identifiers = [stage.get("instance_id") for stage in forward_stages]
        if len(set(identifiers)) != len(identifiers) or any(not value for value in identifiers):
            raise ValueError("stage instance ids must be unique and non-empty")
        trace = []
        for stage in reversed(forward_stages):
            if set(stage) != {"instance_id", "mechanism", "controls"}:
                raise ValueError("foundation stage contract changed")
            mechanism = stage["mechanism"]
            if mechanism not in self.experts:
                raise ValueError(f"no admitted expert for {mechanism}")
            result = self.experts[mechanism].restore(current, stage["controls"])
            current = result["audio"]
            trace.append(
                {
                    "instance_id": stage["instance_id"],
                    "mechanism": mechanism,
                    "frames": int(current.size),
                }
            )
        return {
            "audio": current,
            "trace": trace,
            "source_audio_modified": False,
            "expert_order_inputs": False,
        }
