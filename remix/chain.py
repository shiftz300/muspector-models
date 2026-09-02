"""Offline, abstention-first runtime for the complete analysis chain."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .family import FamilyRuntime
from .gate import GateRuntime, vector
from .inference import sha256
from .knobs import KnobRuntime
from .order import OrderRuntime
from .quality import checked_audio
from .spec import KINDS


def _audio_hash(dry: np.ndarray, wet: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(dry.tobytes())
    digest.update(wet.tobytes())
    return digest.hexdigest()


class ChainRuntime:
    """Compose family, family-set gate, Order 2, and knob recovery safely."""

    def __init__(
        self,
        family: Path,
        manifest: Path,
        gate: Path,
        order: Path,
        bundle: Path,
        evidence: Path,
        target: torch.device | None = None,
    ) -> None:
        cpu = torch.device("cpu") if target is None else target
        if cpu.type != "cpu":
            raise ValueError("chain runtime is currently sealed for offline CPU inference only")
        self.family = FamilyRuntime(family, manifest)
        self.gate = GateRuntime(gate)
        self.order = OrderRuntime(order)
        self.knobs = KnobRuntime(bundle, evidence, cpu)

    def infer(self, dry: np.ndarray, wet: np.ndarray) -> dict:
        clean = checked_audio(dry, name="dry chain copy")
        affected = checked_audio(wet, name="wet chain copy")
        if clean.shape != affected.shape:
            raise ValueError("chain pair geometry differs")
        before = _audio_hash(clean, affected)

        family = self.family.infer(clean, affected)
        probe = self.knobs.infer(clean, affected, KINDS)
        features = vector(family, probe, clean, affected, self.knobs)
        gate = self.gate.infer(features)
        active = tuple(gate["active"])

        report = {
            "schema": 1,
            "decision": "abstain" if not gate["accepted"] else "bypass",
            "active": list(active) if gate["accepted"] else [],
            "family": family,
            "gate": gate,
            "order": None,
            "knobs": None,
            "quality": {
                "analysis_only": True,
                "source_audio_modified": False,
                "automatic_normalization": False,
                "automatic_limiting": False,
                "lossy_reencoding": False,
                "physical_audio_devices_used": False,
            },
        }
        if gate["accepted"] and active:
            order = self.order.infer(clean, affected, active)
            knobs = self.knobs.controls_from_report(order)
            report.update(decision="accepted", order=order, knobs=knobs)

        if before != _audio_hash(clean, affected):
            raise ValueError("chain inference mutated source audio")
        return report

    def assert_artifacts_unchanged(self) -> None:
        self.family.assert_artifacts_unchanged()
        self.gate.assert_artifact_unchanged()
        self.order.assert_artifacts_unchanged()
        if sha256(self.knobs.bundle_path) != self.knobs.bundle_hash:
            raise ValueError("inverse bundle changed")
        if sha256(self.knobs.evidence_path) != self.knobs.evidence_hash:
            raise ValueError("knob evidence changed")
