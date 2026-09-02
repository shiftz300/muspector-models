#!/usr/bin/env python3
"""Finalize the synthetic per-device Drive adapter proof."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "remix/runs/drive-adapter-pilot"


def _load(name: str) -> dict:
    return json.loads((RUN / name).read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    training = _load("metrics.json")
    chain = _load("chain-acceptance.json")
    runtime = _load("runtime-contract.json")
    checkpoint = Path(training["checkpoint"])
    onnx = Path(runtime["onnx"])
    gates = {
        "isolated_validation_promotable": bool(training["promotable"]),
        "adapter_final_mean_improvement_at_least_20_percent": training[
            "adapter_final"
        ]["mean_relative_improvement"]
        >= 0.20,
        "adapter_final_peak_gate": training["adapter_final"]["peak_ratio_p95"] <= 1.35,
        "chain_validation_accepted": bool(chain["accepted"]),
        "all_chain_gates_passed": all(chain["gates"].values()),
        "runtime_passed": bool(runtime["passed"]),
        "checkpoint_hash_matches": (
            training["checkpoint_sha256"]
            == chain["adapter_sha256"]
            == runtime["checkpoint_sha256"]
            == _sha256(checkpoint)
        ),
        "onnx_hash_matches": runtime["onnx_sha256"] == _sha256(onnx),
        "onnx_torch_parity": runtime["parity"]["onnx_vs_torch_chunked"][
            "max_absolute_error"
        ]
        <= 2.0e-5,
        "silence_bit_exact": (
            training["runtime"]["silence_max_absolute_output"] == 0.0
            and runtime["parity"]["silence_max_absolute_output"] == 0.0
        ),
        "source_files_read_only": bool(training["policy"]["source_files_read_only"]),
        "no_automatic_normalization": not training["policy"]["automatic_normalization"],
        "locked_tele_test_closed": not training["policy"]["locked_tele_test_opened"],
        "no_physical_audio_devices": not training["policy"]["physical_audio_devices_used"],
    }
    accepted = all(gates.values())
    report = {
        "schema": 1,
        "status": "accepted-synthetic-device-adapter" if accepted else "rejected",
        "accepted": accepted,
        "model_family": "drive-device-causal-residual-adapter",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "onnx": str(onnx),
        "onnx_sha256": _sha256(onnx),
        "gates": gates,
        "scope": {
            "ready_for_aligned_capture_kit_trial": accepted,
            "ready_for_ui_integration": False,
            "real_hardware_fidelity_claim_allowed": False,
            "synthetic_device_domain_only": True,
            "locked_tele_test_opened": False,
            "physical_audio_devices_used": False,
        },
    }
    (RUN / "acceptance.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
