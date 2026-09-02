#!/usr/bin/env python3
"""Freeze the multi-device stable-forward experiment and cleanup decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing Phase-4 evidence: {path}")
    return json.loads(path.read_text())


def summarize(workspace: Path) -> dict:
    runs = workspace / "remix/runs"
    dfz_root = runs / "asrnn-dfz-stable-pilot"
    cs3_root = runs / "asrnn-cs3-stable-pilot"
    dfz = _read(dfz_root / "candidate-comparison.json")
    cs3 = _read(cs3_root / "candidate-comparison.json")
    dfz_gain = _read(dfz_root / "gain-audit.json")
    cs3_gain = _read(cs3_root / "gain-audit.json")
    dfz_refit = _read(dfz_root / "output-refit.json")
    cs3_refit = _read(cs3_root / "output-refit.json")
    cs3_refit_eval = _read(cs3_root / "metrics-output-refit.json")

    def device_result(base, gain, refit, refit_eval=None):
        return {
            "data_audit": base["data_audit"],
            "best_source": base["selected_source_name"],
            "best_source_sha256": base["selected_source_sha256"],
            "base_official_eval": base["official_eval"],
            "base_accepted": base["accepted"],
            "failed_base_gate": "absolute_peak_error_p95 <= 0.02",
            "train_setting_gain_upper_bound": gain,
            "output_refit": refit,
            "output_refit_official_eval": refit_eval,
            "promoted": False,
            "checkpoint_retained": False,
        }

    return {
        "schema": 1,
        "phase": "phase-4-multi-device-stable-forward-pilot",
        "status": "completed-no-additional-device-promoted",
        "phase_4_evaluation_complete": True,
        "phase_4_runtime_architecture_complete": True,
        "accepted_devices": ["rat"],
        "rejected_devices": ["dfz", "cs3"],
        "devices": {
            "dfz": device_result(dfz, dfz_gain, dfz_refit),
            "cs3": device_result(cs3, cs3_gain, cs3_refit, cs3_refit_eval),
        },
        "architecture": {
            "runtime": "control-count-agnostic stable conditioned LSTM",
            "supported_control_widths_exercised": [1, 2, 3],
            "recurrent_state_explicit": True,
            "bias_free_output": True,
            "source_audio_processing": "direct lossless PCM read; no normalization",
        },
        "decision": {
            "dfz": (
                "reject: excellent waveform/spectral metrics but peak P95 remains 0.113; "
                "train-only gain and frozen-core output refit do not pass calibration"
            ),
            "cs3": (
                "reject: frozen-core output refit passed train calibration but one-shot "
                "official-eval peak P95 remains 0.0264"
            ),
            "next_model_work": (
                "do not add post-hoc gain patches; future DFZ/CS-3 work requires peak-aware "
                "recurrent training and a new locked-final capture pack"
            ),
        },
        "quality_contract": {
            "source_files_read_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
        },
        "license_scope": "internal non-commercial development only (CC-BY-NC-4.0 data)",
        "official_eval_is_locked_final": False,
        "ui_integration_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("remix/runs/asrnn-multi-device-phase4-summary.json"),
    )
    args = parser.parse_args()
    report = summarize(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
