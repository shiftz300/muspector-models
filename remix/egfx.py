"""Archived EGFx RAT inverse experiment; product execution is fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .asrnn_data import rat_files
from .clean import SpectralNet, _align, _sidr, accepted, eligible, evaluate, loss
from .cross import align
from .license_gate import require_product_uses


def partition(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
    return "calibration" if bucket == 0 else "development" if bucket == 1 else "fit"


def audit(root: Path, maximum_lag: int, minimum_alignment: float) -> tuple[list[dict], dict]:
    dry_root, wet_root = root / "Clean/Clean", root / "RAT/RAT"
    rows = []
    for wet_path in sorted(wet_root.rglob("*.wav")):
        relative = wet_path.relative_to(wet_root); dry_path = dry_root / relative
        dry, dry_rate = soundfile.read(dry_path, dtype="float32"); wet, wet_rate = soundfile.read(wet_path, dtype="float32")
        if dry_rate != 48000 or wet_rate != 48000 or dry.ndim != 1 or wet.ndim != 1: raise ValueError("EGFx pair geometry differs")
        _, _, lag, score = align(dry, wet, maximum_lag)
        rows.append({"group": relative.stem, "pickup": relative.parent.name, "relative": str(relative), "split": partition(relative.stem), "lag": lag, "alignment": score, "eligible": score >= minimum_alignment})
    groups = {name: {row["group"] for row in rows if row["split"] == name} for name in ("fit", "calibration", "development")}
    if any(groups[left] & groups[right] for left in groups for right in groups if left < right): raise RuntimeError("EGFx performance group leaked across splits")
    return rows, {
        "files": len(rows), "eligible": sum(row["eligible"] for row in rows), "performance_groups": len({row["group"] for row in rows}),
        "splits": {name: {"files": sum(row["eligible"] and row["split"] == name for row in rows), "groups": len(groups[name])} for name in groups},
        "group_overlap": 0, "source_read_only": True, "physical_audio_devices_used": False,
    }


def audio(root: Path, row: dict) -> tuple[np.ndarray, np.ndarray]:
    dry, _ = soundfile.read(root / "Clean/Clean" / row["relative"], dtype="float32")
    wet, _ = soundfile.read(root / "RAT/RAT" / row["relative"], dtype="float32")
    count = min(len(dry), len(wet)); lag = row["lag"]
    if lag >= 0: return dry[: count - lag].copy(), wet[lag:count].copy()
    return dry[-lag:count].copy(), wet[: count + lag].copy()


class Windows(Dataset):
    def __init__(self, root: Path, rows: list[dict], frames: int, clips: int):
        self.root, self.rows, self.frames, self.clips = root, rows, frames, clips

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        dry, wet = audio(self.root, self.rows[index])
        starts = np.linspace(0, len(dry) - self.frames, self.clips * 4, dtype=int)
        energy = [np.mean(np.square(wet[start : start + self.frames], dtype=np.float64)) for start in starts]
        chosen = np.sort(starts[np.argsort(energy)[-self.clips :]])
        return torch.from_numpy(np.stack([wet[start : start + self.frames] for start in chosen]).copy()), torch.from_numpy(np.stack([dry[start : start + self.frames] for start in chosen]).copy())


@torch.inference_mode()
def evaluate_rows(model: SpectralNet, root: Path, rows: list[dict], strength: float = 1.0) -> dict:
    values = []; mutations = nonfinite = geometry = 0; model.eval()
    for index, row in enumerate(rows):
        dry_path, wet_path = root / "Clean/Clean" / row["relative"], root / "RAT/RAT" / row["relative"]
        dry_source, _ = soundfile.read(dry_path, dtype="float32"); wet_source, _ = soundfile.read(wet_path, dtype="float32")
        before = hashlib.sha256(dry_source.tobytes() + wet_source.tobytes()).hexdigest()
        dry, wet = audio(root, row); prediction = model(torch.from_numpy(wet)[None], strength=strength)[0].numpy()
        nonfinite += int(not np.isfinite(prediction).all()); geometry += int(prediction.shape != wet.shape)
        mutations += int(before != hashlib.sha256(dry_source.tobytes() + wet_source.tobytes()).hexdigest())
        baseline, restored = _align(wet, dry), _align(prediction, dry); energy = max(float(np.mean(np.square(dry, dtype=np.float64))), 1e-12)
        values.append({
            "baseline_esr": float(np.mean(np.square(baseline - dry, dtype=np.float64)) / energy), "restored_esr": float(np.mean(np.square(restored - dry, dtype=np.float64)) / energy),
            "baseline_mae": float(np.mean(np.abs(baseline.astype(np.float64) - dry))), "restored_mae": float(np.mean(np.abs(restored.astype(np.float64) - dry))),
            "sidr_improvement_db": _sidr(prediction, dry) - _sidr(wet, dry),
            "baseline_peak_error": abs(float(np.max(np.abs(baseline))) - float(np.max(np.abs(dry)))), "restored_peak_error": abs(float(np.max(np.abs(restored))) - float(np.max(np.abs(dry)))),
        })
        if index % 24 == 23: print(json.dumps({"stage": "egfx-evaluate", "completed": index + 1, "total": len(rows)}), flush=True)
    def mean(name): return float(np.mean([row[name] for row in values]))
    def p95(name): return float(np.quantile([row[name] for row in values], .95))
    return {
        "examples": len(values), "coverage": 1.0,
        "aligned_esr_improvement": 1 - mean("restored_esr") / max(mean("baseline_esr"), 1e-12),
        "aligned_mae_improvement": 1 - mean("restored_mae") / max(mean("baseline_mae"), 1e-12),
        "sidr_improvement_db": mean("sidr_improvement_db"),
        "baseline_aligned_esr_p95": p95("baseline_esr"), "restored_aligned_esr_p95": p95("restored_esr"),
        "baseline_aligned_peak_error_p95": p95("baseline_peak_error"), "restored_aligned_peak_error_p95": p95("restored_peak_error"),
        "source_mutations": mutations, "nonfinite_outputs": nonfinite, "geometry_errors": geometry,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/corpus/egfxset")); parser.add_argument("--cycle", type=Path, default=Path("cycles/clean7.json")); parser.add_argument("--output", type=Path, default=Path("runs/clean/model7"))
    parser.add_argument("--epochs", type=int, default=6); parser.add_argument("--batch", type=int, default=8); parser.add_argument("--clips", type=int, default=1); parser.add_argument("--frames", type=int, default=16384); parser.add_argument("--channels", type=int, default=12); parser.add_argument("--rate", type=float, default=1e-3)
    args = parser.parse_args()
    require_product_uses(
        Path(__file__).with_name("data_sources.json"),
        {"egfxset": "train-restoration"},
    )
    if args.output.exists(): raise FileExistsError(f"refusing to replace EGFx run {args.output}")
    cycle_bytes = args.cycle.read_bytes(); cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "clean1" or cycle.get("round") != 7 or cycle.get("status") != "planned": raise ValueError("invalid clean7 cycle")
    args.output.mkdir(parents=True)
    rows, data_audit = audit(args.data, cycle["alignment"]["maximum_lag_frames"], cycle["alignment"]["minimum_absolute_correlation"])
    (args.output / "pairs.json").write_text(json.dumps({"schema": 1, "audit": data_audit, "rows": rows}, indent=2) + "\n")
    selected = {name: [row for row in rows if row["eligible"] and row["split"] == name] for name in ("fit", "calibration", "development")}
    torch.manual_seed(20260901); np.random.seed(20260901); model = SpectralNet(args.channels); optimizer = torch.optim.AdamW(model.parameters(), lr=args.rate, weight_decay=1e-5)
    loader = DataLoader(Windows(args.data, selected["fit"], args.frames, args.clips), batch_size=args.batch, shuffle=True, num_workers=0)
    history = []; best = float("inf"); state = None
    for epoch in range(args.epochs):
        model.train(); totals = []
        for wet, dry in loader:
            wet, dry = wet.flatten(0, 1), dry.flatten(0, 1); value, _ = loss(model, wet, dry); optimizer.zero_grad(set_to_none=True); value.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); totals.append(float(value.detach()))
        calibration = evaluate_rows(model, args.data, selected["calibration"]); score = -calibration["aligned_esr_improvement"] - .01 * calibration["sidr_improvement_db"]
        if score < best: best = score; state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append({"epoch": epoch + 1, "loss": float(np.mean(totals)), "calibration": calibration}); (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n"); print(json.dumps({"stage": "egfx-train", **history[-1]}), flush=True)
    model.load_state_dict(state); checkpoint = args.output / "clean.pt"; torch.save({"schema": 1, "architecture": "complex-stft", "sample_rate": 48000, "channels": args.channels, "n_fft": model.n_fft, "hop": model.hop, "device": "egfx-rat", "kind": "clean", "state_dict": state, "dataset": "https://zenodo.org/records/7044411", "license": "CC-BY-4.0"}, checkpoint)
    development = evaluate_rows(model, args.data, selected["development"]); passed, failures = accepted(development, cycle["gates"])
    asrnn_paths, asrnn_eligibility = eligible(rat_files(Path("data/corpus/asrnn-physical-effects"), "eval"), .03); cross = evaluate(model, asrnn_paths, torch.device("cpu"))["metrics"]
    result = {"schema": 1, "status": "accepted-device-development" if passed else "rejected", "accepted": passed, "failures": failures, "model": {"parameters": sum(p.numel() for p in model.parameters()), "checkpoint": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "trained_from_scratch": True}, "data": {**data_audit, "license": "CC-BY-4.0", "splits": {name: len(value) for name, value in selected.items()}}, "development": development, "asrnn_cross": {**cross, "oracle_eligibility": asrnn_eligibility}, "gates": cycle["gates"], "quality": {"source_audio_modified": False, "physical_audio_devices_used": False, "automatic_normalization": False, "automatic_limiting": False, "automatic_dither": False, "lossy_reencoding": False}, "limitations": ["one EGFx RAT capture domain", "oracle evaluation alignment", "output level remains unresolved", "not a multi-device seal"], "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest()}
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
