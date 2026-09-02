"""Cross-domain EGFx RAT restoration audit without retaining rendered audio."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import soundfile
import torch
from scipy.signal import correlate, correlation_lags

from .clean import _align, _sidr
from .confidence import features, load_restorer


def pairs(root: Path, limit: int) -> list[tuple[Path, Path]]:
    dry_root, wet_root = root / "Clean/Clean", root / "RAT/RAT"
    wet = sorted(
        wet_root.rglob("*.wav"),
        key=lambda path: hashlib.sha256(str(path.relative_to(wet_root)).encode()).digest(),
    )
    result = [(dry_root / path.relative_to(wet_root), path) for path in wet[:limit]]
    if len(result) != limit or not all(dry.is_file() for dry, _ in result):
        raise ValueError("EGFx RAT cross-domain pairs are incomplete")
    return result


def align(dry: np.ndarray, wet: np.ndarray, maximum: int) -> tuple[np.ndarray, np.ndarray, int, float]:
    count = min(len(dry), len(wet)); dry, wet = dry[:count], wet[:count]
    correlation = correlate(wet, dry, mode="full", method="fft")
    lags = correlation_lags(len(wet), len(dry), mode="full")
    eligible = np.abs(lags) <= maximum
    index = np.flatnonzero(eligible)[np.argmax(np.abs(correlation[eligible]))]
    lag = int(lags[index])
    if lag >= 0: clean, affected = dry[: count - lag], wet[lag:count]
    else: clean, affected = dry[-lag:count], wet[: count + lag]
    left, right = clean.astype(np.float64), affected.astype(np.float64)
    score = float(abs(np.dot(left, right)) / np.sqrt(max(np.dot(left, left) * np.dot(right, right), 1e-24)))
    return clean.copy(), affected.copy(), lag, score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/corpus/egfxset"))
    parser.add_argument("--cycle", type=Path, default=Path("cycles/clean6.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/clean/model4/clean.pt"))
    parser.add_argument("--gate", type=Path, default=Path("runs/clean/model5/gate.joblib"))
    parser.add_argument("--gate-report", type=Path, default=Path("runs/clean/model5/valid.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/clean/model6"))
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to replace cross-domain run {args.output}")
    cycle_bytes = args.cycle.read_bytes(); cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "clean1" or cycle.get("round") != 6 or cycle.get("status") != "planned": raise ValueError("invalid clean6 cycle")
    args.output.mkdir(parents=True)
    gate = joblib.load(args.gate); gate_threshold = json.loads(args.gate_report.read_text())["threshold"]
    model = load_restorer(args.checkpoint, torch.device("cpu"))
    rows = []; mutations = nonfinite = geometry = 0
    for index, (dry_path, wet_path) in enumerate(pairs(args.data, cycle["data"]["examples"])):
        dry, dry_rate = soundfile.read(dry_path, dtype="float32"); wet, wet_rate = soundfile.read(wet_path, dtype="float32")
        if dry_rate != 48000 or wet_rate != 48000 or dry.ndim != 1 or wet.ndim != 1: raise ValueError("EGFx pair geometry differs")
        before = hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest()
        clean, affected, lag, alignment = align(dry, wet, cycle["alignment"]["maximum_lag_frames"])
        probability = float(gate.predict_proba(features(affected)[None])[0, 1]); selected = probability >= gate_threshold
        prediction = model(torch.from_numpy(affected)[None])[0].detach().numpy() if selected else affected.copy()
        nonfinite += int(not np.isfinite(prediction).all()); geometry += int(prediction.shape != affected.shape)
        mutations += int(before != hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest())
        baseline = _align(affected, clean); restored = _align(prediction, clean); energy = max(float(np.mean(np.square(clean, dtype=np.float64))), 1e-12)
        rows.append({
            "pair": str(wet_path.relative_to(args.data)), "lag": lag, "alignment": alignment, "selected": selected, "confidence": probability,
            "baseline_esr": float(np.mean(np.square(baseline - clean, dtype=np.float64)) / energy),
            "restored_esr": float(np.mean(np.square(restored - clean, dtype=np.float64)) / energy),
            "baseline_mae": float(np.mean(np.abs(baseline.astype(np.float64) - clean))),
            "restored_mae": float(np.mean(np.abs(restored.astype(np.float64) - clean))),
            "sidr_improvement_db": _sidr(prediction, clean) - _sidr(affected, clean),
            "baseline_peak_error": abs(float(np.max(np.abs(baseline))) - float(np.max(np.abs(clean)))),
            "restored_peak_error": abs(float(np.max(np.abs(restored))) - float(np.max(np.abs(clean)))),
        })
        if index % 24 == 23: print(json.dumps({"stage": "cross", "completed": index + 1, "total": cycle["data"]["examples"]}), flush=True)
    aligned = [row for row in rows if row["alignment"] >= cycle["alignment"]["minimum_absolute_correlation"] and row["selected"]]
    def mean(name): return float(np.mean([row[name] for row in aligned]))
    def p95(name): return float(np.quantile([row[name] for row in aligned], .95))
    metrics = {
        "examples": len(rows), "aligned_selected": len(aligned), "coverage": len(aligned) / len(rows),
        "alignment_coverage": sum(row["alignment"] >= cycle["alignment"]["minimum_absolute_correlation"] for row in rows) / len(rows),
        "confidence_coverage": sum(row["selected"] for row in rows) / len(rows),
        "aligned_esr_improvement": 1 - mean("restored_esr") / max(mean("baseline_esr"), 1e-12),
        "aligned_mae_improvement": 1 - mean("restored_mae") / max(mean("baseline_mae"), 1e-12),
        "sidr_improvement_db": mean("sidr_improvement_db"),
        "improved_fraction": sum(row["restored_esr"] < row["baseline_esr"] for row in aligned) / len(aligned),
        "p95_ratio": p95("restored_esr") / max(p95("baseline_esr"), 1e-12),
        "peak_p95_ratio": p95("restored_peak_error") / max(p95("baseline_peak_error"), 1e-12),
        "source_mutations": mutations, "nonfinite_outputs": nonfinite, "geometry_errors": geometry,
    }
    failures = []
    for name, minimum in cycle["gates"]["minimum"].items():
        if metrics[name] < minimum: failures.append(name)
    for name, maximum in cycle["gates"]["maximum"].items():
        if metrics[name] > maximum: failures.append(name)
    for name in ("source_mutations", "nonfinite_outputs", "geometry_errors"):
        if metrics[name]: failures.append(name)
    result = {
        "schema": 1, "status": "accepted-cross-development" if not failures else "rejected", "accepted": not failures, "failures": failures,
        "metrics": metrics, "artifacts": {"restorer": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "gate": hashlib.sha256(args.gate.read_bytes()).hexdigest()},
        "data": {"source": "EGFxSet RAT", "license": "CC-BY-4.0", "source_read_only": True, "physical_audio_devices_used": False},
        "quality": {"rendered_audio_retained": False, "source_audio_modified": False, "automatic_normalization": False, "automatic_limiting": False, "lossy_reencoding": False},
        "limitations": ["deterministic development sample", "oracle pair alignment is evaluation-only", "single EGFx RAT capture domain"],
        "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest(), "rows": rows,
    }
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "accepted", "failures", "metrics")}, sort_keys=True))


if __name__ == "__main__":
    main()
