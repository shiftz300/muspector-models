"""Packaged Wet-only generic Drive presence runtime.

This classifier owns neither graph order nor controls. It combines the frozen
nonlinear family expert with the frozen any-effect Clean gate and intentionally
does not expose the shared Ambience output from the historical full stack.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .blind2 import WINDOW, BlindFamilyPresence, aggregate_windows, log_mel
from .packages import digest, verify


PACKAGE_ID = "foundation-drive-presence"
NONLINEAR_THRESHOLD = 0.37
ANY_EFFECT_THRESHOLD = 0.915


def _load_family(path: Path, family: str) -> tuple[BlindFamilyPresence, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    manifest = checkpoint.get("manifest", {})
    if (
        checkpoint.get("schema") != 1
        or manifest.get("architecture") != "compact-audio-resnet18-family-expert"
        or manifest.get("family") != family
        or manifest.get("order_output") is not False
        or manifest.get("controls_output") is not False
        or manifest.get("sample_rate") != 44_100
        or manifest.get("window_frames") != WINDOW
    ):
        raise ValueError(f"{family} presence checkpoint contract changed")
    model = BlindFamilyPresence(family)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def _windows(audio: np.ndarray) -> torch.Tensor:
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 1 or value.size < 1 or not np.isfinite(value).all():
        raise ValueError("Drive presence expects finite mono audio")
    if len(value) < WINDOW:
        value = np.pad(value, (0, WINDOW - len(value)))
    maximum = len(value) - WINDOW
    starts = sorted({0, maximum // 2, maximum})
    return torch.stack([
        torch.from_numpy(value[start : start + WINDOW].copy()) for start in starts
    ])


def _probability(
    model: BlindFamilyPresence,
    checkpoint: dict,
    windows: torch.Tensor,
    aggregation: str,
) -> float:
    with torch.inference_mode():
        logits = model(log_mel(windows))
        calibration = checkpoint.get("calibration")
        if not isinstance(calibration, list) or len(calibration) != 1:
            raise ValueError("presence calibration contract changed")
        scale = float(calibration[0]["scale"])
        bias = float(calibration[0]["bias"])
        calibrated = torch.sigmoid(logits * scale + bias)
        if aggregation == "top-two":
            result = aggregate_windows(calibrated)
        elif aggregation == "mean":
            result = calibrated.mean(dim=0)
        else:
            raise ValueError(f"unsupported presence aggregation: {aggregation}")
    return float(result[0])


class FoundationDrivePresenceRuntime:
    sample_rate = 44_100
    labels = ("nonlinear",)

    def __init__(self, nonlinear_checkpoint: Path, any_gate_checkpoint: Path) -> None:
        self.nonlinear_path = Path(nonlinear_checkpoint).resolve()
        self.gate_path = Path(any_gate_checkpoint).resolve()
        self.nonlinear, self.nonlinear_checkpoint = _load_family(
            self.nonlinear_path, "nonlinear"
        )
        self.gate, self.gate_checkpoint = _load_family(self.gate_path, "any")
        if float(self.nonlinear_checkpoint.get("threshold", -1.0)) != NONLINEAR_THRESHOLD:
            raise ValueError("nonlinear presence threshold changed")
        if float(self.gate_checkpoint.get("threshold", -1.0)) != ANY_EFFECT_THRESHOLD:
            raise ValueError("any-effect threshold changed")
        if self.gate_checkpoint.get("file_aggregation") != "mean":
            raise ValueError("any-effect aggregation changed")
        if "base_fallback_threshold" in self.gate_checkpoint or "family_fallback" in self.gate_checkpoint:
            raise ValueError("foundation Drive presence forbids shared-family gate fallbacks")

    @classmethod
    def package(cls, root: Path) -> "FoundationDrivePresenceRuntime":
        root = Path(root).resolve()
        package = verify(root)
        if (
            package["id"] != PACKAGE_ID
            or package["version"] != "1.0.0"
            or package["kind"] != "classifier"
            or package["quality"] != "development"
            or package["audio_behavior"] != "analysis_only"
            or package["sample_rate"] != cls.sample_rate
            or package["channels"] != 1
            or package["controls"]
        ):
            raise ValueError("unsupported foundation Drive presence package")
        roles = {row["role"]: root / row["path"] for row in package["artifacts"]}
        expected = {
            "runtime", "nonlinear_checkpoint", "any_effect_gate_checkpoint",
            "development", "external_software_chain",
        }
        if set(roles) != expected:
            raise ValueError("foundation Drive presence artifact roles changed")
        runtime = cls(roles["nonlinear_checkpoint"], roles["any_effect_gate_checkpoint"])
        runtime.root = root
        runtime.package_hash = digest(root / "package.json")
        runtime.hashes = {path: digest(path) for path in roles.values()}
        return runtime

    def analyze(self, wet: np.ndarray) -> dict:
        source = np.asarray(wet)
        before = hashlib.sha256(source.tobytes()).hexdigest()
        windows = _windows(source)
        nonlinear_probability = _probability(
            self.nonlinear, self.nonlinear_checkpoint, windows, "top-two"
        )
        any_probability = _probability(
            self.gate, self.gate_checkpoint, windows, "mean"
        )
        gate_active = any_probability >= ANY_EFFECT_THRESHOLD
        detected = gate_active and nonlinear_probability >= NONLINEAR_THRESHOLD
        if hashlib.sha256(source.tobytes()).hexdigest() != before:
            raise ValueError("Drive presence runtime mutated its Wet input")
        return {
            "labels": ["nonlinear"],
            "probabilities": {"nonlinear": nonlinear_probability},
            "thresholds": {"nonlinear": NONLINEAR_THRESHOLD},
            "detected": {"nonlinear": detected},
            "any_effect_probability": any_probability,
            "any_effect_threshold": ANY_EFFECT_THRESHOLD,
            "any_effect_gate_active": gate_active,
            "order_output": False,
            "controls_output": False,
            "source_audio_modified": False,
        }

    def assert_artifacts_unchanged(self) -> None:
        if not hasattr(self, "root"):
            return
        if digest(self.root / "package.json") != self.package_hash:
            raise ValueError("foundation Drive presence package changed")
        for path, expected in self.hashes.items():
            if digest(path) != expected:
                raise ValueError(f"foundation Drive presence artifact changed: {path.name}")
