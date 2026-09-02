"""Explicit named-device adapters layered after the canonical remix chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.signal import resample_poly

from .chain import ChainRuntime
from .drive_model import DriveControlEstimator
from .packages import digest, verify
from .quality import checked_audio
from .semantic_rat_runtime import RATSemanticRuntime


DEVICES = {"rat": "drive"}
RAT_NAMES = ("distortion", "filter", "volume")


def analysis(audio: np.ndarray, source_rate: int, target_rate: int, frames: int) -> np.ndarray:
    value = checked_audio(audio, name="device analysis copy")
    if value.ndim != 1:
        raise ValueError("device adapters require mono analysis audio")
    if source_rate <= 0:
        raise ValueError("source sample rate must be positive")
    if source_rate != target_rate:
        common = np.gcd(source_rate, target_rate)
        value = resample_poly(value, target_rate // common, source_rate // common).astype(np.float32)
    if len(value) < frames:
        value = np.pad(value, (0, frames - len(value)))
    return np.asarray(value[:frames], dtype=np.float32)


class RatAdapter(RATSemanticRuntime):
    """Load the accepted RAT inverse entirely from a materialized package."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        package = verify(self.root)
        if package["id"] != "ratknobs" or package["version"] != "1.0.0":
            raise ValueError("unsupported RAT adapter package")
        roles = {row["role"]: self.root / row["path"] for row in package["artifacts"]}
        expected = {"initializer", "search_renderer", "semantic_heads", "model_card"}
        if set(roles) != expected:
            raise ValueError("RAT adapter artifact contract changed")
        self.manifest = json.loads(roles["model_card"].read_text())
        if (
            not self.manifest.get("accepted")
            or not self.manifest.get("runtime_replay_verified")
            or self.manifest.get("runtime_replay_compute") != "cpu"
        ):
            raise ValueError("RAT adapter evidence is not accepted for CPU replay")
        self.files = {
            "initializer": roles["initializer"],
            "renderer": roles["search_renderer"],
            "semantic_heads": roles["semantic_heads"],
        }
        payload = torch.load(self.files["initializer"], map_location="cpu", weights_only=True)
        self.initializer = DriveControlEstimator()
        self.initializer.load_state_dict(payload["state_dict"])
        self.initializer.eval()
        self.heads = joblib.load(self.files["semantic_heads"])
        self.hashes = {path: digest(path) for path in roles.values()}
        self.package_hash = digest(self.root / "package.json")

    def infer_batch(self, dry: np.ndarray, wet: np.ndarray) -> list[dict]:
        before = hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest()
        result = super().infer(dry, wet)
        identifiable = np.asarray(result["upstream_identifiable"], dtype=bool)
        physical = np.asarray(result["physical_controls"], dtype=np.float32)
        if before != hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest():
            raise ValueError("RAT adapter mutated analysis audio")
        return [
            {
                "schema": 1,
                "device": "rat",
                "decision": "accepted" if active else "abstain",
                "controls": (
                    {name: {"normalized": float(physical[row, index])} for index, name in enumerate(RAT_NAMES)}
                    if active
                    else {}
                ),
                "reason": None if active else "upstream-unidentifiable",
                "package_sha256": self.package_hash,
                "source_audio_modified": False,
                "physical_audio_devices_used": False,
            }
            for row, active in enumerate(identifiable)
        ]

    def infer_controls(self, dry: np.ndarray, wet: np.ndarray) -> dict:
        reports = self.infer_batch(dry, wet)
        if len(reports) != 1:
            raise ValueError("single-pair RAT adapter received a batch")
        return reports[0]

    def assert_artifacts_unchanged(self) -> None:
        if digest(self.root / "package.json") != self.package_hash:
            raise ValueError("RAT adapter package changed")
        for path, expected in self.hashes.items():
            if digest(path) != expected:
                raise ValueError(f"RAT adapter artifact changed: {path.name}")


class DeviceRuntime:
    """Run canonical recognition first, then one explicit named-device adapter."""

    def __init__(self, chain: ChainRuntime, device: str, adapter: RatAdapter):
        if device not in DEVICES or device != adapter.manifest.get("device"):
            raise ValueError(f"unsupported or mismatched device adapter: {device}")
        self.chain = chain
        self.device = device
        self.adapter = adapter

    def infer(self, dry: np.ndarray, wet: np.ndarray, sample_rate: int) -> dict:
        original_dry = checked_audio(dry, name="dry source copy")
        original_wet = checked_audio(wet, name="wet source copy")
        if original_dry.shape != original_wet.shape:
            raise ValueError("device pair geometry differs")
        before = hashlib.sha256(original_dry.tobytes() + original_wet.tobytes()).hexdigest()
        chain_dry = analysis(original_dry, sample_rate, 44_100, 220_500)
        chain_wet = analysis(original_wet, sample_rate, 44_100, 220_500)
        chain = self.chain.infer(chain_dry, chain_wet)
        expected = DEVICES[self.device]
        adapter = None
        decision = "abstain"
        reason = "chain-abstained" if chain["decision"] == "abstain" else "family-mismatch"
        if chain["decision"] == "accepted" and chain["active"] == [expected]:
            device_dry = analysis(original_dry, sample_rate, 48_000, 48_000)
            device_wet = analysis(original_wet, sample_rate, 48_000, 48_000)
            adapter = self.adapter.infer_controls(device_dry, device_wet)
            decision = adapter["decision"]
            reason = adapter["reason"]
        if before != hashlib.sha256(original_dry.tobytes() + original_wet.tobytes()).hexdigest():
            raise ValueError("device runtime mutated source audio")
        return {
            "schema": 1,
            "device": self.device,
            "decision": decision,
            "reason": reason,
            "chain": chain,
            "adapter": adapter,
            "quality": {
                "analysis_only": True,
                "source_audio_modified": False,
                "automatic_normalization": False,
                "automatic_limiting": False,
                "lossy_reencoding": False,
                "physical_audio_devices_used": False,
            },
        }

    def assert_artifacts_unchanged(self) -> None:
        self.chain.assert_artifacts_unchanged()
        self.adapter.assert_artifacts_unchanged()
