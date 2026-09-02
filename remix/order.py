"""Audio-safe wrapper for the frozen order-transfer runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .order_search import decode_controls
from .order_transfer_runtime import TransferRemixerRuntime
from .spec import ChainSpec


def guard(report: dict) -> dict:
    """Fall back when the proposed topology has higher diagnostic error."""

    transfer = report.get("transfer")
    if transfer is None:
        report["order2"] = {
            "guarded": False,
            "reason": "single-family",
            "order_used": False,
        }
        return report
    original = tuple(transfer["original_selected"])
    proposed = tuple(report["order"]["selected"])
    errors = {
        tuple(row["topology"]): float(row["reconstruction_error"])
        for row in report["order"]["ranked"]
    }
    original_error = errors[original]
    proposed_error = errors[proposed]
    blocked = proposed_error > original_error
    if blocked:
        normalized = np.asarray(report["normalized_controls"], dtype=np.float32)
        report["chain"] = ChainSpec(decode_controls(original, normalized)).document()
        report["decision"] = "order-fallback"
        report["order"]["selected"] = list(original)
        report["order"]["selected_reconstruction_error"] = original_error
    report["order2"] = {
        "guarded": blocked,
        "reason": "audio-regression" if blocked else "accepted",
        "order_used": not blocked and proposed != original,
        "original": list(original),
        "proposed": list(proposed),
        "selected": list(original if blocked else proposed),
        "original_error": original_error,
        "proposed_error": proposed_error,
        "selected_error": min(original_error, proposed_error),
    }
    return report


class OrderRuntime:
    def __init__(self, run: Path):
        self.base = TransferRemixerRuntime(run)

    def infer(self, dry: np.ndarray, wet: np.ndarray, active: Sequence[str]) -> dict:
        return guard(self.base.infer(dry, wet, active))

    def assert_artifacts_unchanged(self) -> None:
        self.base.assert_artifacts_unchanged()
