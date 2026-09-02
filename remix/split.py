"""Train independent, position-agnostic Drive and Reverb inverse stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from torch import nn
from torch.utils.data import DataLoader

from .chain_report import load_model, source_domains
from .clean import _pre, _spectrum
from .data import _pair, discover_order_records
from .forward_chain import ForwardChainRuntime
from .forward_drive import _segment
from .net import SpectralNet
from .spec import ChainSpec
from .stages import (
    CORPUS,
    INITIAL,
    REVERB_DILATIONS,
    clean_audit,
    effects,
    evaluate,
    examples,
    external,
    gates,
    load_initial,
    metric,
    portable,
    sha256,
    state,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "runs/stage/model2"
RUN = ROOT / "runs/stage/model3"


@torch.inference_mode()
def synthetic(runtime: ForwardChainRuntime, corpus: Path, split: str, count: int, frames: int, seed: int) -> dict[str, list[dict]]:
    domains = source_domains(corpus, split)
    names = sorted(domains)
    if not names:
        raise RuntimeError(f"no clean source domains for {split}")
    rows = {"drive": [], "reverb": []}
    for index in range(count):
        rng = random.Random(seed + index * 104_729)
        name = names[index % len(names)]
        paths = domains[name]
        clean = _segment(paths[rng.randrange(len(paths))], frames, rng)
        peak = max(float(np.max(np.abs(clean))), 1.0e-6)
        clean = np.asarray(clean * (rng.uniform(0.05, 0.28) / peak), dtype=np.float32)
        reverb, drive = effects(rng)
        drive_target = clean if index % 2 == 0 else runtime.render(clean, ChainSpec((reverb,)))
        drive_source = runtime.render(drive_target, ChainSpec((drive,)))
        reverb_target = clean if (index // 2) % 2 == 0 else runtime.render(clean, ChainSpec((drive,)))
        reverb_source = runtime.render(reverb_target, ChainSpec((reverb,)))
        rows["drive"].append({"source": torch.from_numpy(drive_source.copy()), "target": torch.from_numpy(drive_target.copy()), "domain": name})
        rows["reverb"].append({"source": torch.from_numpy(reverb_source.copy()), "target": torch.from_numpy(reverb_target.copy()), "domain": name})
    return rows


@torch.inference_mode()
def real(corpus: Path, kind: str, split: str, count: int, frames: int, seed: int) -> list[dict]:
    records = [row for row in discover_order_records(corpus) if row.split == split and row.order == (kind,)]
    if not records:
        raise RuntimeError(f"no {kind}-only DAFx records in {split}")
    rng = random.Random(seed)
    indices = list(range(len(records))); rng.shuffle(indices)
    rows = []
    for index in range(count):
        record = records[indices[index % len(indices)]]
        clean, wet = _pair(record)
        clean = resample_poly(clean, 160, 147).astype(np.float32)
        wet = resample_poly(wet, 160, 147).astype(np.float32)
        length = min(len(clean), len(wet)); clean, wet = clean[:length], wet[:length]
        offset = (seed + index * 104_729) % max(1, length - frames + 1)
        clean, wet = clean[offset : offset + frames], wet[offset : offset + frames]
        if len(clean) < frames:
            clean = np.pad(clean, (0, frames - len(clean))); wet = np.pad(wet, (0, frames - len(wet)))
        peak = max(float(np.max(np.abs(clean))), float(np.max(np.abs(wet))), 1.0e-6)
        scale = 0.22 / peak
        rows.append({"source": torch.from_numpy((wet * scale).astype(np.float32)), "target": torch.from_numpy((clean * scale).astype(np.float32)), "domain": "dafx-real"})
    return rows


def loss(model: SpectralNet, source: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    prediction = model(source)
    target_delta = target - source
    prediction_delta = prediction - source
    delta_scale = target_delta.abs().mean(1).clamp_min(1e-4)
    delta_l1 = ((prediction_delta - target_delta).abs().mean(1) / delta_scale).mean()
    target_norm = target_delta.square().sum(1).sqrt().clamp_min(1e-4)
    prediction_norm = prediction_delta.square().sum(1).sqrt().clamp_min(1e-4)
    delta_direction = (prediction_delta * target_delta).sum(1) / (prediction_norm * target_norm)
    delta_magnitude = (prediction_norm / target_norm - 1.0).abs().mean()
    scale = ((prediction * target).sum(1) / prediction.square().sum(1).clamp_min(1e-6)).detach()
    aligned = prediction * scale[:, None]
    aligned_mae = torch.nn.functional.l1_loss(aligned, target)
    direct_mae = torch.nn.functional.l1_loss(prediction, target)
    aligned_esr = (aligned - target).square().sum() / target.square().sum().clamp_min(1e-6)
    direct_esr = (prediction - target).square().sum() / target.square().sum().clamp_min(1e-6)
    correlation = (prediction * target).sum(1) / (prediction.square().sum(1).sqrt() * target.square().sum(1).sqrt()).clamp_min(1e-6)
    emphasized = torch.nn.functional.l1_loss(_pre(aligned), _pre(target))
    spectral = _spectrum(aligned, target)
    peak = (prediction.abs().amax(1) - target.abs().amax(1)).abs().mean()
    base = (
        0.20 * aligned_mae + 0.10 * emphasized + 0.10 * spectral
        + 0.05 * aligned_esr + 0.20 * (1.0 - correlation).mean()
        + 0.50 * direct_mae + 0.20 * direct_esr + 0.10 * peak
        + 0.50 * delta_l1 + 0.25 * (1.0 - delta_direction).mean()
        + 0.10 * delta_magnitude
    )
    if kind == "reverb":
        predicted_envelope = torch.nn.functional.avg_pool1d(prediction.abs().unsqueeze(1), 1024, 512).squeeze(1)
        target_envelope = torch.nn.functional.avg_pool1d(target.abs().unsqueeze(1), 1024, 512).squeeze(1)
        base = base + 0.20 * torch.nn.functional.l1_loss(predicted_envelope, target_envelope)
    return base


@torch.inference_mode()
def stage_metric(model: SpectralNet, rows: list[dict], target: torch.device) -> dict:
    model.eval(); baselines, predictions, targets = [], [], []
    for first in range(0, len(rows), 8):
        batch = rows[first : first + 8]
        source = torch.stack([row["source"] for row in batch]).to(target)
        prediction = model(source).cpu().numpy()
        baselines.extend(row["source"].numpy() for row in batch)
        predictions.extend(prediction)
        targets.extend(row["target"].numpy() for row in batch)
    return metric(baselines, predictions, targets)


def rank(report: dict) -> float:
    """Prefer material movement toward Clean, not tiny aligned gains."""

    return (
        report["aligned_esr_improvement"]
        + 0.10 * report["sidr_improvement_db"]
        + 0.50 * report["closure"]
        + 0.10 * min(report["correction_ratio"], 1.0)
        + 0.10 * report["correction_direction"]
        + min(report["esr_improvement"], 0.0)
    )


def train(kind: str, model: SpectralNet, rows: list[dict], calibration: list[dict], output: Path, target: torch.device, epochs: int, batch: int, rate: float) -> tuple[SpectralNet, list[dict]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=1e-5)
    loader = DataLoader(rows, batch_size=batch, shuffle=True, num_workers=0)
    initial = stage_metric(model, calibration, target)
    best = rank(initial)
    best_state, stale = state(model), 0
    history = [{"epoch": 0, "loss": None, "score": best, "calibration": initial, "frozen_baseline": True}]
    (output / f"{kind}.history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps({"stage": kind, "epoch": 0, "score": best, "calibration": initial, "frozen_baseline": True}), flush=True)
    for epoch in range(epochs):
        model.train(); values = []
        for item in loader:
            value = loss(model, item["source"].to(target), item["target"].to(target), kind)
            optimizer.zero_grad(set_to_none=True); value.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); values.append(float(value.detach().cpu()))
        report = stage_metric(model, calibration, target)
        score = rank(report)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(values)), "score": score, "calibration": report})
        (output / f"{kind}.history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps({"stage": kind, "epoch": epoch + 1, "loss": history[-1]["loss"], "score": score, "calibration": report}), flush=True)
        if score > best:
            best, best_state, stale = score, state(model), 0
        else:
            stale += 1
        if epoch + 1 >= 10 and stale >= 4:
            break
    model.load_state_dict(best_state)
    return model, history


def score(report: dict) -> float:
    return rank(report["chain"])


def save_stage(model: SpectralNet, path: Path, kind: str, source_hash: str) -> None:
    torch.save({
        "schema": 1,
        "architecture": "complex-stft",
        "sample_rate": 48_000,
        "channels": model.channels,
        "n_fft": model.n_fft,
        "hop": model.hop,
        "dilations": [list(value) for value in model.dilations],
        "kind": f"{kind}-inverse",
        "stage": kind,
        "position_agnostic": True,
        "state_dict": state(model),
        "initialization_sha256": source_hash,
        "training_target": "immediate-predecessor",
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, default=Path("cycles/stage3.json")); parser.add_argument("--corpus", type=Path, default=CORPUS); parser.add_argument("--baseline", type=Path, default=BASELINE); parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--train", type=int, default=960); parser.add_argument("--real", type=int, default=480); parser.add_argument("--calibrate", type=int, default=128); parser.add_argument("--develop", type=int, default=160); parser.add_argument("--frames", type=int, default=16_384); parser.add_argument("--drive-epochs", type=int, default=18); parser.add_argument("--reverb-epochs", type=int, default=22); parser.add_argument("--batch", type=int, default=8); parser.add_argument("--rate", type=float, default=2e-4); parser.add_argument("--threads", type=int, default=5); parser.add_argument("--seal-start", type=int, default=192)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to replace {args.output}")
    cycle_bytes = args.cycle.read_bytes(); cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "stage" or cycle.get("round") not in (2, 3) or cycle.get("status") != "planned": raise ValueError("invalid stage cycle")
    if sha256(args.baseline / "drive.pt") != cycle["baseline"]["drive_sha256"] or sha256(args.baseline / "reverb.pt") != cycle["baseline"]["reverb_sha256"]: raise ValueError("baseline changed")
    args.output.mkdir(parents=True)
    (args.output / "lock.json").write_text(json.dumps({"schema": 1, "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest(), "development_opened": False, "seal_opened": False}, indent=2) + "\n")
    torch.manual_seed(20260926); np.random.seed(20260926); torch.set_num_threads(args.threads)
    target = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); runtime = ForwardChainRuntime()
    fit = synthetic(runtime, args.corpus, "train", args.train, args.frames, 20260926)
    calibration = synthetic(runtime, args.corpus, "calibrate", args.calibrate, args.frames, 20260927)
    for kind in ("drive", "reverb"):
        fit[kind].extend(real(args.corpus, kind, "train", args.real, args.frames, 20260928 + (kind == "reverb")))
        calibration[kind].extend(real(args.corpus, kind, "calibrate", 64, args.frames, 20260930 + (kind == "reverb")))
    drive = load_model(args.baseline / "drive.pt").to(target); reverb = load_model(args.baseline / "reverb.pt").to(target)
    drive, drive_history = train("drive", drive, fit["drive"], calibration["drive"], args.output, target, args.drive_epochs, args.batch, args.rate)
    reverb, reverb_history = train("reverb", reverb, fit["reverb"], calibration["reverb"], args.output, target, args.reverb_epochs, args.batch, args.rate)
    valid_stage = synthetic(runtime, args.corpus, "valid", 96, args.frames * 2, 20261001)
    for kind in ("drive", "reverb"):
        valid_stage[kind].extend(real(args.corpus, kind, "valid", 64, args.frames * 2, 20261002 + (kind == "reverb")))
    stage_reports = {"drive": stage_metric(drive, valid_stage["drive"], target), "reverb": stage_metric(reverb, valid_stage["reverb"], target)}
    development_rows = examples(runtime, sorted({path for paths in source_domains(args.corpus, "valid").values() for path in paths}), args.develop, args.frames * 2, 20261004)
    candidate = evaluate(drive, reverb, development_rows, target, 1.0, 1.0)
    baseline_drive, baseline_reverb = load_model(args.baseline / "drive.pt").to(target), load_model(args.baseline / "reverb.pt").to(target)
    baseline = evaluate(baseline_drive, baseline_reverb, development_rows, target, 1.0, 1.0)
    development_gates = gates(candidate)
    comparison_gate = score(candidate) >= score(baseline) - 0.001
    stage_gates = {kind: values["esr_improvement"] >= 0.0 and values["aligned_esr_improvement"] >= 0.0 for kind, values in stage_reports.items()}
    lock = json.loads((args.output / "lock.json").read_text()); lock["development_opened"] = True; (args.output / "lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    if not all(development_gates.values()) or not comparison_gate or not all(stage_gates.values()):
        report = {"schema": 1, "status": "rejected-development", "accepted": False, "stage": stage_reports, "stage_gates": stage_gates, "development": candidate, "development_gates": development_gates, "baseline": baseline, "comparison_gate": comparison_gate, "scores": {"candidate": score(candidate), "baseline": score(baseline)}}
        (args.output / "valid.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n"); print(json.dumps(report, sort_keys=True)); raise SystemExit(2)
    drive_source_hash, reverb_source_hash = sha256(args.baseline / "drive.pt"), sha256(args.baseline / "reverb.pt"); drive_path, reverb_path = args.output / "drive.pt", args.output / "reverb.pt"; save_stage(drive, drive_path, "drive", drive_source_hash); save_stage(reverb, reverb_path, "reverb", reverb_source_hash)
    lock["seal_opened"] = True; (args.output / "lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    outside = external(drive, reverb, args.corpus, 64, target, 1.0, 1.0, "test", start=args.seal_start)
    baseline_outside = external(baseline_drive, baseline_reverb, args.corpus, 64, target, 1.0, 1.0, "test", start=args.seal_start)
    external_gates = {"absolute_esr": outside["esr_improvement"] >= 0.15, "aligned_esr": outside["aligned_esr_improvement"] >= 0.15, "sidr": outside["sidr_improvement_db"] >= 1.00, "closure": outside["closure"] >= 0.15, "correction": outside["correction_ratio"] >= 0.25, "direction": outside["correction_direction"] >= 0.50, "audible": outside["audible_fraction"] >= 0.50, "tail": outside["restored_aligned_esr_p95"] <= outside["baseline_aligned_esr_p95"], "peak": outside["restored_aligned_peak_p95"] <= outside["baseline_aligned_peak_p95"], "baseline_score": rank(outside) >= rank(baseline_outside) - 0.001}
    runtime_report = portable(drive_path, reverb_path, args.output, 1.0, 1.0)
    accepted = all(external_gates.values()) and runtime_report["accepted"]
    report = {"schema": 1, "status": "accepted-stage-development" if accepted else "rejected-seal", "accepted": accepted, "models": {"drive": {"path": str(drive_path), "sha256": sha256(drive_path), "parameters": sum(p.numel() for p in drive.parameters())}, "reverb": {"path": str(reverb_path), "sha256": sha256(reverb_path), "parameters": sum(p.numel() for p in reverb.parameters())}}, "training": {"shared_optimizer": False, "chain_gradient": False, "synthetic_fit_per_stage": args.train, "real_single_effect_fit_per_stage": args.real, "clean_sampling": "uniform-domain"}, "stage": stage_reports, "stage_gates": stage_gates, "development": candidate, "development_gates": development_gates, "baseline_development": baseline, "comparison_gate": comparison_gate, "seal": outside, "baseline_seal": baseline_outside, "seal_gates": external_gates, "runtime": runtime_report, "order": {"stage_order_fixed": False, "forward_supported": [["reverb", "drive"], ["drive", "reverb"]], "restore": "reverse runtime forward order"}, "data": {"source_read_only": True, "physical_audio_devices_used": False, "rendered_audio_retained": False, "clean_sources": clean_audit(args.corpus)}, "quality": cycle["quality"], "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest()}
    (args.output / "valid.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n"); print(json.dumps({"accepted": accepted, "seal": outside, "baseline_seal": baseline_outside, "runtime": runtime_report}, sort_keys=True))
    if not accepted:
        drive_path.unlink(missing_ok=True); reverb_path.unlink(missing_ok=True); (args.output / "drive.onnx").unlink(missing_ok=True); (args.output / "reverb.onnx").unlink(missing_ok=True); raise SystemExit(2)


if __name__ == "__main__":
    main()
