"""Control-only runtime for the sealed Drive/Delay/Reverb inverse bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .inference import RemixerRuntime, sha256
from .quality import checked_audio
from .spec import CONTROL_NAMES, KINDS
from .train import physical_controls


UNITS = ("dB", "%", "dB", "ms", "%", "%", "s", "%", "%")


def verify_evidence(path: Path, bundle_sha256: str) -> dict:
    evidence = json.loads(path.read_text())
    capability = evidence.get("capabilities", {}).get("knob", {})
    if not capability.get("accepted") or capability.get("failures"):
        raise ValueError("knob capability is not sealed and accepted")
    if evidence.get("artifacts", {}).get("bundle") != bundle_sha256:
        raise ValueError("sealed evidence belongs to another inverse bundle")
    if not evidence.get("artifacts_unchanged"):
        raise ValueError("inverse bundle changed during sealed evaluation")
    if evidence.get("physical_audio_devices_used"):
        raise ValueError("sealed evidence violates the audio-device boundary")
    return evidence


class KnobRuntime(RemixerRuntime):
    """Recover controls while deliberately discarding the rejected order overlay."""

    def __init__(self, bundle: Path, evidence: Path, target: torch.device | None = None):
        super().__init__(bundle, target)
        self.evidence_path = evidence
        self.evidence_hash = sha256(evidence)
        self.evidence = verify_evidence(evidence, self.bundle_hash)

    def _prepare_analysis_pair(
        self, dry: np.ndarray, wet: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        clean = checked_audio(dry, name="dry analysis copy")
        affected = checked_audio(wet, name="wet analysis copy")
        if clean.shape != affected.shape:
            raise ValueError("paired audio geometry differs")
        return clean, affected

    def _model_analysis_batch(
        self, dry: np.ndarray, wet: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clean = torch.zeros((8, dry.shape[0]), dtype=torch.float32, device=self.target)
        affected = torch.zeros_like(clean)
        clean[0] = torch.from_numpy(dry).to(self.target)
        affected[0] = torch.from_numpy(wet).to(self.target)
        return clean, affected

    @torch.inference_mode()
    def infer_controls(
        self, dry: np.ndarray, wet: np.ndarray, active: Sequence[str]
    ) -> dict:
        report = super().infer(dry, wet, tuple(active))
        return self.controls_from_report(report)

    def controls_from_report(self, report: dict) -> dict:
        """Decode controls from an already-computed inverse report."""

        if report.get("bundle_sha256") != self.bundle_hash:
            raise ValueError("inverse report belongs to another bundle")
        active_set = set(report["active_effects"])
        active = [name for name in KINDS if name in active_set]
        normalized = np.asarray(report.get("normalized_controls", np.zeros(9)), dtype=np.float32)
        if normalized.shape != (len(CONTROL_NAMES),):
            raise ValueError("inverse report has invalid control geometry")
        physical = physical_controls(torch.from_numpy(normalized).unsqueeze(0))[0].numpy()
        controls = {
            name: {
                "normalized": float(normalized[index]),
                "value": float(physical[index]),
                "unit": UNITS[index],
            }
            for index, name in enumerate(CONTROL_NAMES)
            if name.split(".", 1)[0] in active_set
        }
        return {
            "schema": 1,
            "active": active,
            "controls": controls,
            "bundle_sha256": self.bundle_hash,
            "evidence_sha256": self.evidence_hash,
            "order_used": False,
            "source_audio_modified": False,
            "physical_audio_devices_used": False,
        }
