"""Audit whether missed synthetic Drive stages are genuinely safe to bypass.

The audit is diagnostic only. It evaluates the currently packaged nonlinear
presence expert and any-effect gate, then inspects immediate Drive stage pairs
only for missed synthetic positives. It never trains, selects a checkpoint,
changes thresholds, consumes order labels, or opens locked-final.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .blind2 import LABELS, BlindFamilyPresence, log_mel
from .blind_data2 import BlindPresenceData
from .restoration_quality import (
    _band_envelope,
    _crest_error,
    _relative_error,
    _spectral_error,
    _transient_envelope,
)
from .train_blind2 import SEED, _device, _loader


FROZEN_THRESHOLDS = {
    "maximum_scale_invariant_residual_db": -20.0,
    "maximum_absolute_gain_shift_db": 1.0,
    "maximum_spectral_error": 0.05,
    "maximum_high_band_error": 0.10,
    "maximum_transient_error": 0.10,
    "maximum_dynamics_error": 0.10,
    "minimum_safe_fraction": 0.95,
    "minimum_missed_examples": 10,
}


def _calibrated(logits: torch.Tensor, checkpoint: dict) -> torch.Tensor:
    calibration = checkpoint.get("calibration")
    if not isinstance(calibration, list) or len(calibration) != 1:
        raise ValueError("family checkpoint calibration changed")
    return torch.sigmoid(
        logits * float(calibration[0]["scale"]) + float(calibration[0]["bias"])
    )


def _load(path: Path, family: str, device: torch.device) -> tuple[BlindFamilyPresence, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    manifest = checkpoint.get("manifest", {})
    if manifest.get("family") != family or manifest.get("labels") != [family]:
        raise ValueError(f"expected standalone {family} family checkpoint")
    model = BlindFamilyPresence(family).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def _scale_invariant_residual_db(predecessor: np.ndarray, wet: np.ndarray) -> float:
    left = predecessor.astype(np.float64)
    right = wet.astype(np.float64)
    scale = float(np.dot(left, right) / (np.dot(left, left) + 1.0e-12))
    residual = right - scale * left
    ratio = math.sqrt(float(np.mean(residual**2)) + 1.0e-12) / (
        math.sqrt(float(np.mean((scale * left) ** 2)) + 1.0e-12)
    )
    return 20.0 * math.log10(max(ratio, 1.0e-12))


def _bypass_metrics(predecessor: np.ndarray, wet: np.ndarray, sample_rate: int) -> dict:
    if sample_rate != 48_000:
        raise ValueError("Drive bypass audit expects the 48-kHz product renderer")
    if predecessor.shape != wet.shape or predecessor.ndim != 1 or not predecessor.size:
        raise ValueError("Drive bypass audit requires aligned nonempty mono audio")
    if not np.isfinite(predecessor).all() or not np.isfinite(wet).all():
        raise ValueError("Drive bypass audit received non-finite audio")
    predecessor_rms = math.sqrt(float(np.mean(predecessor.astype(np.float64) ** 2)) + 1.0e-12)
    wet_rms = math.sqrt(float(np.mean(wet.astype(np.float64) ** 2)) + 1.0e-12)
    metrics = {
        "scale_invariant_residual_db": _scale_invariant_residual_db(predecessor, wet),
        "gain_shift_db": 20.0 * math.log10(wet_rms / predecessor_rms),
        "spectral_error": _spectral_error(wet, predecessor),
        "high_band_error": _relative_error(
            _band_envelope(wet, 3_000.0, 12_000.0),
            _band_envelope(predecessor, 3_000.0, 12_000.0),
        ),
        "transient_error": _relative_error(
            _transient_envelope(wet), _transient_envelope(predecessor)
        ),
        "dynamics_error": _crest_error(wet, predecessor),
        "wet_peak": float(np.max(np.abs(wet))),
    }
    gates = {
        "scale_invariant_residual": metrics["scale_invariant_residual_db"]
        <= FROZEN_THRESHOLDS["maximum_scale_invariant_residual_db"],
        "gain_shift": abs(metrics["gain_shift_db"])
        <= FROZEN_THRESHOLDS["maximum_absolute_gain_shift_db"],
        "spectrum": metrics["spectral_error"]
        <= FROZEN_THRESHOLDS["maximum_spectral_error"],
        "high_band": metrics["high_band_error"]
        <= FROZEN_THRESHOLDS["maximum_high_band_error"],
        "transient": metrics["transient_error"]
        <= FROZEN_THRESHOLDS["maximum_transient_error"],
        "dynamics": metrics["dynamics_error"]
        <= FROZEN_THRESHOLDS["maximum_dynamics_error"],
        "no_clipping": metrics["wet_peak"] <= 1.0,
    }
    return {"metrics": metrics, "gates": gates, "safe_to_bypass": all(gates.values())}


def _aggregate(rows: list[dict]) -> dict:
    safe_fraction = float(np.mean([row["safe_to_bypass"] for row in rows])) if rows else 0.0
    metric_names = (
        "scale_invariant_residual_db",
        "gain_shift_db",
        "spectral_error",
        "high_band_error",
        "transient_error",
        "dynamics_error",
    )
    gates = {
        "enough_missed_examples": len(rows) >= FROZEN_THRESHOLDS["minimum_missed_examples"],
        "safe_fraction": safe_fraction >= FROZEN_THRESHOLDS["minimum_safe_fraction"],
    }
    return {
        "missed_examples": len(rows),
        "safe_examples": sum(row["safe_to_bypass"] for row in rows),
        "safe_fraction": safe_fraction,
        "metric_medians": {
            name: float(np.median([row["metrics"][name] for row in rows])) if rows else None
            for name in metric_names
        },
        "metric_worst": {
            "scale_invariant_residual_db": (
                float(max(row["metrics"]["scale_invariant_residual_db"] for row in rows))
                if rows
                else None
            ),
            "absolute_gain_shift_db": (
                float(max(abs(row["metrics"]["gain_shift_db"]) for row in rows))
                if rows
                else None
            ),
            **{
                name: (
                    float(max(row["metrics"][name] for row in rows)) if rows else None
                )
                for name in metric_names
                if name not in {"scale_invariant_residual_db", "gain_shift_db"}
            },
        },
        "gates": gates,
        "negative_detection_may_bypass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nonlinear", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = _device(args.device)
    nonlinear, nonlinear_checkpoint = _load(args.nonlinear, "nonlinear", device)
    gate, gate_checkpoint = _load(args.gate, "any", device)
    dataset = BlindPresenceData(Path.cwd(), "development", 640, SEED + 37)
    loader = _loader(dataset, 16, False)
    predictions: list[bool] = []
    targets: list[bool] = []
    sources: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            features = log_mel(batch["audio"].to(device))
            nonlinear_probability = _calibrated(nonlinear(features), nonlinear_checkpoint)
            gate_probability = _calibrated(gate(features), gate_checkpoint)
            detected = (
                (nonlinear_probability[:, 0] >= float(nonlinear_checkpoint["threshold"]))
                & (gate_probability[:, 0] >= float(gate_checkpoint["threshold"]))
            )
            predictions.extend(detected.cpu().tolist())
            nonlinear_index = LABELS.index("nonlinear")
            targets.extend((batch["target"][:, nonlinear_index] >= 0.5).tolist())
            sources.extend(batch["source"])

    missed_indices = [
        index
        for index, (predicted, expected, source) in enumerate(
            zip(predictions, targets, sources, strict=True)
        )
        if expected
        and not predicted
        and source.startswith("synthetic-product-render:nonlinear:")
    ]
    audit_dataset = BlindPresenceData(
        Path.cwd(),
        "development",
        640,
        SEED + 37,
        include_nonlinear_audit_pair=True,
    )
    rows = []
    for index in missed_indices:
        row = audit_dataset[index]
        if row["source"] != sources[index]:
            raise ValueError("deterministic audit replay changed source identity")
        result = _bypass_metrics(
            row["nonlinear_predecessor"].numpy(),
            row["nonlinear_wet"].numpy(),
            int(row["nonlinear_pair_sample_rate"]),
        )
        rows.append({"index": index, "source": row["source"], **result})

    aggregate = _aggregate(rows)
    report = {
        "schema": 1,
        "status": (
            "development-bypass-safe-not-promoted"
            if aggregate["negative_detection_may_bypass"]
            else "development-bypass-rejected"
        ),
        "scope": "impact audit for missed synthetic Drive stages; no training or threshold selection",
        "device": str(device),
        "frozen_thresholds": FROZEN_THRESHOLDS,
        "development": {
            "examples": len(dataset),
            "nonlinear_positives": sum(targets),
            "synthetic_nonlinear_misses": len(missed_indices),
            "aggregate": aggregate,
            "rows": rows,
        },
        "checkpoints": {
            "nonlinear": {
                "path": str(args.nonlinear.resolve()),
                "sha256": hashlib.sha256(args.nonlinear.read_bytes()).hexdigest(),
            },
            "any_effect_gate": {
                "path": str(args.gate.resolve()),
                "sha256": hashlib.sha256(args.gate.read_bytes()).hexdigest(),
            },
        },
        "boundaries": {
            "training": False,
            "threshold_selection": False,
            "order_labels_used": False,
            "controls_used": False,
            "amp_used": False,
            "reverb_model_used": False,
            "locked_final_opened": False
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "examples": report["development"]["examples"],
        "nonlinear_positives": report["development"]["nonlinear_positives"],
        "synthetic_nonlinear_misses": report["development"]["synthetic_nonlinear_misses"],
        "aggregate": aggregate,
    }, indent=2))


if __name__ == "__main__":
    main()
