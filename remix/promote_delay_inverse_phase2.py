#!/usr/bin/env python3
"""Promote the accepted Delay inverse candidate with an auditable rollback copy."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/order-control-paired-public"
PHASE = ROOT / "remix/runs/delay-inverse-phase2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    acceptance = json.loads((PHASE / "acceptance.json").read_text())
    training = json.loads((PHASE / "metrics.json").read_text())
    if not acceptance["accepted"]:
        raise ValueError("Delay inverse Phase 2 candidate has not passed acceptance")
    source = RUN / "delay-estimator.pt"
    candidate = Path(acceptance["checkpoint"])
    source_hash = _sha256(source)
    candidate_hash = _sha256(candidate)
    if source_hash != training["source_checkpoint_sha256"]:
        raise ValueError("canonical Delay checkpoint changed after Phase 2 training")
    if candidate_hash != acceptance["checkpoint_sha256"]:
        raise ValueError("accepted Delay candidate hash changed before promotion")
    rollback = PHASE / "delay-estimator-before-phase2.pt"
    if rollback.exists() and _sha256(rollback) != source_hash:
        raise ValueError("existing Phase 2 rollback checkpoint has an unexpected hash")
    if not rollback.exists():
        shutil.copy2(source, rollback)
    temporary = source.with_suffix(".phase2.tmp")
    shutil.copy2(candidate, temporary)
    os.replace(temporary, source)
    promoted_hash = _sha256(source)
    if promoted_hash != candidate_hash:
        raise RuntimeError("promoted Delay checkpoint failed its hash verification")
    report = {
        "schema": 1,
        "status": "promoted",
        "source_checkpoint_sha256": source_hash,
        "rollback_checkpoint": str(rollback),
        "rollback_checkpoint_sha256": _sha256(rollback),
        "promoted_checkpoint": str(source),
        "promoted_checkpoint_sha256": promoted_hash,
        "physical_audio_devices_used": False,
        "locked_tele_test_opened": False,
    }
    (PHASE / "promotion.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
