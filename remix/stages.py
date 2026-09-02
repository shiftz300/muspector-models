"""Train and seal a compact reverse-order Reverb->Drive restoration chain.

The controllable forward models create ephemeral aligned triples only:
clean -> reverb intermediate -> drive wet. No rendered audio is retained and
no physical audio device API is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from torch import nn
from torch.utils.data import DataLoader

from .audible import measure as audible
from .clean import _align, _pre, _sidr, _spectrum
from .data import _pair, discover_order_records, dry_sources
from .forward_chain import FORWARD_RATE, ForwardChainRuntime
from .forward_drive import _segment
from .net import SpectralNet
from .portable import PortableRuntime, describe, export, rss_mb
from .spec import ChainSpec, Drive, Reverb


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/corpus/guitar-effects-chains"
APPLE = ROOT / "data/corpus/apple-au/manifest.json"
INITIAL = ROOT / "runs/clean/model11/clean.pt"
RUN = ROOT / "runs/chain/model7"
REVERB_DILATIONS = ((1, 1), (2, 4), (4, 16), (8, 64))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def effects(rng: random.Random) -> tuple[Reverb, Drive]:
    return (
        Reverb(0.2 * 40.0 ** rng.random(), rng.uniform(0.05, 0.95), rng.uniform(0.10, 0.62)),
        Drive(rng.uniform(0.0, 30.0), rng.random(), rng.uniform(-15.0, 6.0)),
    )


def clean_sources(corpus: Path, split: str) -> list[Path]:
    result = set(dry_sources(corpus, split))
    payload = json.loads(APPLE.read_text())["captures"]
    groups = {name: {row["source_group"] for row in payload if row["split"] == name} for name in ("train", "calibrate", "valid", "test")}
    if any(groups[left] & groups[right] for left in groups for right in groups if left < right):
        raise RuntimeError("multi-clean source groups leak across splits")
    result.update(
        Path(row["source_path"])
        for row in payload
        if row["split"] == split and Path(row["source_path"]).is_file()
    )
    return sorted(result)


def clean_audit(corpus: Path) -> dict:
    payload = json.loads(APPLE.read_text())["captures"]
    return {
        name: {
            "files": len(clean_sources(corpus, name)),
            "groups": len({row["source_group"] for row in payload if row["split"] == name}) + len(dry_sources(corpus, name)),
            "domains": sorted({row["source_domain"] for row in payload if row["split"] == name} | {"dafx-electric"}),
        }
        for name in ("train", "calibrate", "valid", "test")
    }


@torch.inference_mode()
def examples(runtime: ForwardChainRuntime, paths: list[Path], count: int, frames: int, seed: int) -> list[dict]:
    rows = []
    for index in range(count):
        rng = random.Random(seed + index * 104_729)
        clean = _segment(paths[rng.randrange(len(paths))], frames, rng)
        peak = max(float(np.max(np.abs(clean))), 1.0e-6)
        clean = np.asarray(clean * (rng.uniform(0.05, 0.28) / peak), dtype=np.float32)
        reverb, drive = effects(rng)
        if index % 2 == 0:
            middle = runtime.render(clean, ChainSpec((reverb,)))
            wet = runtime.render(middle, ChainSpec((drive,)))
            topology = 0
            drive_source, drive_target = wet, middle
            reverb_source, reverb_target = middle, clean
        else:
            middle = runtime.render(clean, ChainSpec((drive,)))
            wet = runtime.render(middle, ChainSpec((reverb,)))
            topology = 1
            drive_source, drive_target = middle, clean
            reverb_source, reverb_target = wet, middle
        rows.append({
            "clean": torch.from_numpy(clean.copy()),
            "middle": torch.from_numpy(middle.copy()),
            "wet": torch.from_numpy(wet.copy()),
            "drive_source": torch.from_numpy(drive_source.copy()),
            "drive_target": torch.from_numpy(drive_target.copy()),
            "reverb_source": torch.from_numpy(reverb_source.copy()),
            "reverb_target": torch.from_numpy(reverb_target.copy()),
            "topology": torch.tensor(topology, dtype=torch.int64),
        })
    return rows


@torch.inference_mode()
def real_examples(root: Path, count: int, frames: int, seed: int) -> list[dict]:
    """Load deterministic, read-only DAFx train pairs for composite fitting."""

    records = [
        row for row in discover_order_records(root)
        if row.split == "train" and row.order in (("reverb", "drive"), ("drive", "reverb"))
    ]
    if not records:
        raise RuntimeError("DAFx train split has no supported two-effect records")
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    rows = []
    for index in range(count):
        record = records[indices[index % len(indices)]]
        clean, wet = _pair(record)
        clean = resample_poly(clean, 160, 147).astype(np.float32)
        wet = resample_poly(wet, 160, 147).astype(np.float32)
        length = min(len(clean), len(wet))
        clean, wet = clean[:length], wet[:length]
        offset = (seed + index * 104_729) % max(1, length - frames + 1)
        clean, wet = clean[offset : offset + frames], wet[offset : offset + frames]
        if len(clean) < frames:
            clean = np.pad(clean, (0, frames - len(clean)))
            wet = np.pad(wet, (0, frames - len(wet)))
        peak = max(float(np.max(np.abs(clean))), float(np.max(np.abs(wet))), 1.0e-6)
        clean = (clean * (0.22 / peak)).astype(np.float32)
        wet = (wet * (0.22 / peak)).astype(np.float32)
        rows.append({
            "clean": torch.from_numpy(clean.copy()),
            "wet": torch.from_numpy(wet.copy()),
            "topology": torch.tensor(
                0 if record.order == ("reverb", "drive") else 1, dtype=torch.int64
            ),
        })
    return rows


def metric(baselines: list[np.ndarray], predictions: list[np.ndarray], targets: list[np.ndarray]) -> dict:
    rows = []
    for baseline, prediction, target in zip(baselines, predictions, targets, strict=True):
        energy = max(float(np.mean(np.square(target, dtype=np.float64))), 1.0e-12)
        left, right = _align(baseline, target), _align(prediction, target)
        listening = audible(baseline, prediction, target)
        rows.append({
            "baseline_esr": float(np.mean(np.square(baseline - target, dtype=np.float64)) / energy),
            "restored_esr": float(np.mean(np.square(prediction - target, dtype=np.float64)) / energy),
            "baseline_aligned_esr": float(np.mean(np.square(left - target, dtype=np.float64)) / energy),
            "restored_aligned_esr": float(np.mean(np.square(right - target, dtype=np.float64)) / energy),
            "sidr_delta": _sidr(prediction, target) - _sidr(baseline, target),
            "baseline_peak": abs(float(np.max(np.abs(left))) - float(np.max(np.abs(target)))),
            "restored_peak": abs(float(np.max(np.abs(right))) - float(np.max(np.abs(target)))),
            **listening,
        })
    mean = lambda name: float(np.mean([row[name] for row in rows]))
    p95 = lambda name: float(np.quantile([row[name] for row in rows], 0.95))
    return {
        "examples": len(rows),
        "esr_improvement": 1.0 - mean("restored_esr") / max(mean("baseline_esr"), 1.0e-12),
        "aligned_esr_improvement": 1.0 - mean("restored_aligned_esr") / max(mean("baseline_aligned_esr"), 1.0e-12),
        "sidr_improvement_db": mean("sidr_delta"),
        "baseline_aligned_esr_p95": p95("baseline_aligned_esr"),
        "restored_aligned_esr_p95": p95("restored_aligned_esr"),
        "baseline_aligned_peak_p95": p95("baseline_peak"),
        "restored_aligned_peak_p95": p95("restored_peak"),
        "closure": mean("closure"),
        "correction_ratio": mean("correction_ratio"),
        "correction_direction": mean("correction_direction"),
        "audible_fraction": mean("audible"),
    }


@torch.inference_mode()
def evaluate(
    drive: SpectralNet,
    reverb: SpectralNet,
    rows: list[dict],
    target: torch.device,
    drive_strength: float = 1.0,
    reverb_strength: float = 1.0,
) -> dict:
    drive.eval(); reverb.eval()
    drive_base, drive_pred, drive_target = [], [], []
    reverb_base, reverb_pred, reverb_target = [], [], []
    chain_base, chain_pred, chain_wrong, chain_target = [], [], [], []
    for first in range(0, len(rows), 8):
        batch = rows[first : first + 8]
        clean = torch.stack([row["clean"] for row in batch]).to(target)
        wet = torch.stack([row["wet"] for row in batch]).to(target)
        ds = torch.stack([row["drive_source"] for row in batch]).to(target)
        dt = torch.stack([row["drive_target"] for row in batch]).to(target)
        rs = torch.stack([row["reverb_source"] for row in batch]).to(target)
        rt = torch.stack([row["reverb_target"] for row in batch]).to(target)
        dp = drive(ds, strength=drive_strength)
        rp = reverb(rs, strength=reverb_strength)
        correct, wrong = [], []
        for index, row in enumerate(batch):
            value = wet[index : index + 1]
            if int(row["topology"]) == 0:
                right = reverb(drive(value, strength=drive_strength), strength=reverb_strength)
                alternate = drive(reverb(value, strength=reverb_strength), strength=drive_strength)
            else:
                right = drive(reverb(value, strength=reverb_strength), strength=drive_strength)
                alternate = reverb(drive(value, strength=drive_strength), strength=reverb_strength)
            correct.append(right); wrong.append(alternate)
        ch, wh = torch.cat(correct).cpu().numpy(), torch.cat(wrong).cpu().numpy()
        c, w, ds_np, dt_np, rs_np, rt_np, dp_np, rp_np = [
            value.cpu().numpy() for value in (clean, wet, ds, dt, rs, rt, dp, rp)
        ]
        drive_base.extend(ds_np); drive_pred.extend(dp_np); drive_target.extend(dt_np)
        reverb_base.extend(rs_np); reverb_pred.extend(rp_np); reverb_target.extend(rt_np)
        chain_base.extend(w); chain_pred.extend(ch); chain_wrong.extend(wh); chain_target.extend(c)
    drive_metrics = metric(drive_base, drive_pred, drive_target)
    reverb_metrics = metric(reverb_base, reverb_pred, reverb_target)
    chain_metrics = metric(chain_base, chain_pred, chain_target)
    wrong_metrics = metric(chain_base, chain_wrong, chain_target)
    correct_wins = sum(
        np.mean(np.square(right - target, dtype=np.float64)) < np.mean(np.square(wrong - target, dtype=np.float64))
        for right, wrong, target in zip(chain_pred, chain_wrong, chain_target, strict=True)
    ) / len(chain_target)
    safe = 0
    for right, wrong, target_value in zip(chain_pred, chain_wrong, chain_target, strict=True):
        right_error = float(np.mean(np.square(right - target_value, dtype=np.float64)))
        wrong_error = float(np.mean(np.square(wrong - target_value, dtype=np.float64)))
        relative_margin = abs(right_error - wrong_error) / max(min(right_error, wrong_error), 1.0e-12)
        safe += int(relative_margin < 0.01 or right_error < wrong_error)
    return {
        "drive": drive_metrics, "reverb": reverb_metrics, "chain": chain_metrics,
        "wrong_order": wrong_metrics, "correct_order_win_fraction": correct_wins,
        "order_safe_or_ambiguous_fraction": safe / len(chain_target),
        "order_policy": "declared chains unwind in reverse; under 1% reconstruction margin abstains",
    }


def gates(report: dict) -> dict:
    return {name: bool(value) for name, value in {
        "drive_aligned_esr": report["drive"]["aligned_esr_improvement"] >= 0.0,
        "drive_absolute_esr": report["drive"]["esr_improvement"] >= 0.0,
        "reverb_aligned_esr": report["reverb"]["aligned_esr_improvement"] >= 0.0,
        "reverb_absolute_esr": report["reverb"]["esr_improvement"] >= 0.0,
        "chain_aligned_esr": report["chain"]["aligned_esr_improvement"] >= 0.15,
        "chain_absolute_esr": report["chain"]["esr_improvement"] >= 0.15,
        "chain_sidr": report["chain"]["sidr_improvement_db"] >= 1.00,
        "chain_closure": report["chain"]["closure"] >= 0.15,
        "chain_correction": report["chain"]["correction_ratio"] >= 0.25,
        "chain_direction": report["chain"]["correction_direction"] >= 0.50,
        "chain_audible": report["chain"]["audible_fraction"] >= 0.50,
        "chain_tail": report["chain"]["restored_aligned_esr_p95"] <= report["chain"]["baseline_aligned_esr_p95"],
        "chain_peak": report["chain"]["restored_aligned_peak_p95"] <= report["chain"]["baseline_aligned_peak_p95"],
        "order": report["order_safe_or_ambiguous_fraction"] >= 0.75,
    }.items()}


def load_initial(path: Path, kind: str) -> SpectralNet:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if kind == "drive":
        model = SpectralNet(payload["channels"], payload["n_fft"], payload["hop"])
        model.load_state_dict(payload["state_dict"])
    elif kind == "reverb":
        model = SpectralNet(16, payload["n_fft"], payload["hop"], REVERB_DILATIONS)
    else:
        raise ValueError(f"unknown stage kind: {kind}")
    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)
    return model


def state(model: SpectralNet) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save(model: SpectralNet, path: Path, kind: str, source_hash: str) -> None:
    torch.save({
        "schema": 1, "architecture": "complex-stft", "sample_rate": FORWARD_RATE,
        "channels": model.channels, "n_fft": model.n_fft, "hop": model.hop,
        "dilations": [list(value) for value in model.dilations],
        "kind": kind, "topology": ["reverb", "drive"], "state_dict": state(model),
        "initialization_sha256": source_hash, "training_domain": "synthetic-forward-final_challenge",
    }, path)


def stage_loss(model: SpectralNet, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = model(source)
    scale = ((prediction * target).sum(1) / prediction.square().sum(1).clamp_min(1e-6)).detach()
    aligned = prediction * scale[:, None]
    aligned_mae = torch.nn.functional.l1_loss(aligned, target)
    emphasized = torch.nn.functional.l1_loss(_pre(aligned), _pre(target))
    spectral = _spectrum(aligned, target)
    aligned_esr = (aligned - target).square().sum() / target.square().sum().clamp_min(1e-6)
    correlation = (prediction * target).sum(1) / (
        prediction.square().sum(1).sqrt() * target.square().sum(1).sqrt()
    ).clamp_min(1e-6)
    direct_mae = torch.nn.functional.l1_loss(prediction, target)
    direct_esr = (prediction - target).square().sum() / target.square().sum().clamp_min(1e-6)
    level = torch.abs(torch.log(
        (prediction.square().mean(1) + 1e-8) / (target.square().mean(1) + 1e-8)
    )).mean()
    peak = (prediction.abs().amax(1) - target.abs().amax(1)).abs().mean()
    return (
        aligned_mae + 0.25 * emphasized + 0.10 * spectral + 0.10 * aligned_esr
        + 0.50 * (1.0 - correlation).mean() + 0.10 * direct_mae
        + 0.01 * direct_esr + 0.002 * level + 0.10 * peak
    )


def real_chain_loss(drive: SpectralNet, reverb: SpectralNet, batch: dict[str, torch.Tensor], target: torch.device) -> torch.Tensor:
    """Train the final reverse chain on real DAFx train renderings."""

    wet = batch["wet"].to(target)
    clean = batch["clean"].to(target)
    predictions = []
    for index, topology in enumerate(batch["topology"]):
        value = wet[index : index + 1]
        if int(topology) == 0:
            value = reverb(drive(value), strength=1.0)
        else:
            value = drive(reverb(value), strength=1.0)
        predictions.append(value)
    prediction = torch.cat(predictions)
    error = prediction - clean
    esr = error.square().sum() / clean.square().sum().clamp_min(1.0e-6)
    return 0.5 * torch.nn.functional.l1_loss(prediction, clean) + 0.05 * esr + 0.10 * torch.nn.functional.l1_loss(_pre(prediction), _pre(clean))


def choose_strength(drive: SpectralNet, reverb: SpectralNet, rows: list[dict], target: torch.device) -> tuple[float, float, dict]:
    candidates = []
    for drive_strength in (0.75, 1.0):
        for reverb_strength in (0.25, 0.5, 0.75, 1.0):
            report = evaluate(drive, reverb, rows, target, drive_strength, reverb_strength)
            chain = report["chain"]
            score = (
                chain["aligned_esr_improvement"] + 0.05 * chain["sidr_improvement_db"]
                + 2.0 * min(chain["esr_improvement"], 0.0)
            )
            candidates.append({"drive": drive_strength, "reverb": reverb_strength, "score": score, "report": report})
    selected = max(candidates, key=lambda row: row["score"])
    return selected["drive"], selected["reverb"], {"selected": {key: selected[key] for key in ("drive", "reverb", "score")}, "candidates": candidates}


@torch.inference_mode()
def external(drive: SpectralNet, reverb: SpectralNet, root: Path, limit: int, target: torch.device, drive_strength: float, reverb_strength: float, split: str = "valid", start: int = 0) -> dict:
    records = [
        row for row in discover_order_records(root)
        if row.split == split and row.order in (("reverb", "drive"), ("drive", "reverb"))
    ][start : start + limit]
    baseline, restored, targets = [], [], []
    for row in records:
        clean, wet = _pair(row)
        clean = resample_poly(clean, 160, 147).astype(np.float32)
        wet = resample_poly(wet, 160, 147).astype(np.float32)
        count = min(len(clean), len(wet)); clean, wet = clean[:count], wet[:count]
        value = torch.from_numpy(wet)[None].to(target)
        if row.order == ("reverb", "drive"):
            prediction = reverb(drive(value, strength=drive_strength), strength=reverb_strength)[0].cpu().numpy()
        else:
            prediction = drive(reverb(value, strength=reverb_strength), strength=drive_strength)[0].cpu().numpy()
        baseline.append(wet); restored.append(prediction); targets.append(clean)
    result = metric(baseline, restored, targets)
    result["scope"] = f"DAFx Dataset 3 real rendered {split} split; records {start}:{start + len(records)}; topology labels only; controls unknown"
    result["development_only"] = True
    return result


def _portable(drive_path: Path, reverb_path: Path, output: Path, drive_strength: float, reverb_strength: float, seconds: int = 30) -> dict:
    drive_graph, reverb_graph = output / "drive.onnx", output / "reverb.onnx"
    drive_artifact, reverb_artifact = export(drive_path, drive_graph), export(reverb_path, reverb_graph)
    drive = PortableRuntime(drive_path, drive_graph, core=16_384, halo=4_096, threads=2)
    reverb = PortableRuntime(reverb_path, reverb_graph, core=16_384, halo=12_288, threads=2)
    rng = np.random.default_rng(20260915)
    source = (rng.standard_normal(seconds * FORWARD_RATE) * 0.035).astype(np.float32)
    drive.render(source[:FORWARD_RATE], drive_strength); reverb.render(source[:FORWARD_RATE], reverb_strength)
    started = time.perf_counter(); value = reverb.render(drive.render(source, drive_strength), reverb_strength); elapsed = time.perf_counter() - started
    measured = {
        "seconds": seconds, "elapsed_seconds": elapsed, "realtime_factor": elapsed / seconds,
        "process_peak_rss_mb": rss_mb(), "parameters": drive.reference.parameters + reverb.reference.parameters,
        "artifact_bytes": drive_graph.stat().st_size + reverb_graph.stat().st_size,
        "cpu_threads": 2, "finite": bool(np.isfinite(value).all()), "geometry_preserved": value.shape == source.shape,
    }
    checks = {
        "speed": measured["realtime_factor"] <= 0.5, "memory": measured["process_peak_rss_mb"] <= 512,
        "parameters": measured["parameters"] <= 250_000, "artifact": measured["artifact_bytes"] <= 8 * 1024 * 1024,
        "finite": measured["finite"], "geometry": measured["geometry_preserved"],
    }
    return {"accepted": all(checks.values()), "measured": measured, "checks": checks, "artifacts": [drive_artifact, reverb_artifact]}


def _portable_worker(drive_path: Path, reverb_path: Path, output: Path, drive_strength: float, reverb_strength: float, queue) -> None:
    try:
        queue.put(_portable(drive_path, reverb_path, output, drive_strength, reverb_strength))
    except Exception as error:  # pragma: no cover - returned to the parent report.
        queue.put({"accepted": False, "error": f"{type(error).__name__}: {error}"})


def portable(drive_path: Path, reverb_path: Path, output: Path, drive_strength: float, reverb_strength: float) -> dict:
    """Measure runtime in a fresh process so training allocations cannot pollute RSS."""

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_portable_worker, args=(drive_path, reverb_path, output, drive_strength, reverb_strength, queue))
    process.start(); process.join()
    if process.exitcode != 0 or queue.empty():
        return {"accepted": False, "error": f"runtime worker exited {process.exitcode}"}
    return queue.get()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, default=Path("cycles/clean19.json"))
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--initial", type=Path, default=INITIAL)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--train", type=int, default=640)
    parser.add_argument("--real", type=int, default=320)
    parser.add_argument("--calibrate", type=int, default=120)
    parser.add_argument("--develop", type=int, default=120)
    parser.add_argument("--seal", type=int, default=80)
    parser.add_argument("--frames", type=int, default=16_384)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--rate", type=float, default=2e-4)
    parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace chain run {args.output}")
    cycle_bytes = args.cycle.read_bytes(); cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "clean1" or cycle.get("round") != 19 or cycle.get("status") != "planned":
        raise ValueError("invalid clean19 cycle")
    source_hash = sha256(args.initial)
    if source_hash != cycle["model"]["initialization_sha256"]:
        raise ValueError("stage initialization changed")
    args.output.mkdir(parents=True)
    (args.output / "lock.json").write_text(json.dumps({
        "schema": 1, "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest(),
        "development_opened": False, "seal_opened": False,
    }, indent=2) + "\n")
    torch.manual_seed(20260910); np.random.seed(20260910); torch.set_num_threads(args.threads)
    target = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    runtime = ForwardChainRuntime()
    source_audit = clean_audit(args.corpus)
    train_rows = examples(runtime, clean_sources(args.corpus, "train"), args.train, args.frames, 20260910)
    real_rows = real_examples(args.corpus, args.real, args.frames, 20260919)
    calibrate_rows = examples(runtime, clean_sources(args.corpus, "calibrate"), args.calibrate, args.frames, 20260911)
    loader = DataLoader(train_rows, batch_size=args.batch, shuffle=True, num_workers=0)
    real_loader = DataLoader(real_rows, batch_size=args.batch, shuffle=True, num_workers=0)
    drive, reverb = load_initial(args.initial, "drive").to(target), load_initial(args.initial, "reverb").to(target)
    optimizer = torch.optim.AdamW(list(drive.parameters()) + list(reverb.parameters()), lr=args.rate, weight_decay=1e-5)
    history, best, best_drive, best_reverb = [], -float("inf"), None, None
    for epoch in range(args.epochs):
        drive.train(); reverb.train(); totals = []
        for batch in loader:
            drive_loss = stage_loss(drive, batch["drive_source"].to(target), batch["drive_target"].to(target))
            reverb_loss = stage_loss(reverb, batch["reverb_source"].to(target), batch["reverb_target"].to(target))
            value = drive_loss + reverb_loss
            optimizer.zero_grad(set_to_none=True); value.backward()
            nn.utils.clip_grad_norm_(list(drive.parameters()) + list(reverb.parameters()), 1.0)
            optimizer.step(); totals.append(float(value.detach().cpu()))
        for batch in real_loader:
            value = real_chain_loss(drive, reverb, batch, target)
            optimizer.zero_grad(set_to_none=True); value.backward()
            nn.utils.clip_grad_norm_(list(drive.parameters()) + list(reverb.parameters()), 1.0)
            optimizer.step(); totals.append(float(value.detach().cpu()))
        report = evaluate(drive, reverb, calibrate_rows, target)
        score = report["chain"]["aligned_esr_improvement"] + 0.05 * report["chain"]["sidr_improvement_db"]
        history.append({"epoch": epoch + 1, "loss": float(np.mean(totals)), "score": score, "calibration": report})
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps({"stage": "chain-train", "epoch": epoch + 1, "loss": history[-1]["loss"], "score": score, "chain": report["chain"]}), flush=True)
        if score > best:
            best, best_drive, best_reverb = score, state(drive), state(reverb)
    drive.load_state_dict(best_drive); reverb.load_state_dict(best_reverb)
    lock = json.loads((args.output / "lock.json").read_text()); lock["development_opened"] = True
    (args.output / "lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    drive_strength, reverb_strength, strength_report = choose_strength(drive, reverb, calibrate_rows, target)
    development_rows = examples(runtime, clean_sources(args.corpus, "valid"), args.develop, args.frames * 2, 20260912)
    development = evaluate(drive, reverb, development_rows, target, drive_strength, reverb_strength); development_gates = gates(development)
    if not all(development_gates.values()):
        report = {"schema": 1, "status": "rejected-development", "accepted": False, "strength": {"drive": drive_strength, "reverb": reverb_strength, "calibration": strength_report}, "development": development, "development_gates": development_gates}
        (args.output / "valid.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True)); raise SystemExit(2)
    drive_path, reverb_path = args.output / "drive.pt", args.output / "reverb.pt"
    save(drive, drive_path, "drive-inverse-upstream-context", source_hash)
    save(reverb, reverb_path, "reverb-inverse", source_hash)
    lock["seal_opened"] = True; (args.output / "lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    outside = external(drive, reverb, args.corpus, 64, target, drive_strength, reverb_strength, "test", start=64)
    runtime_report = portable(drive_path, reverb_path, args.output, drive_strength, reverb_strength)
    external_gates = {
        "absolute_esr": outside["esr_improvement"] >= 0.0,
        "aligned_esr": outside["aligned_esr_improvement"] >= 0.01,
        "sidr": outside["sidr_improvement_db"] >= 0.10,
        "tail": outside["restored_aligned_esr_p95"] <= outside["baseline_aligned_esr_p95"],
        "peak": outside["restored_aligned_peak_p95"] <= outside["baseline_aligned_peak_p95"],
    }
    sealed = outside
    seal_gates = external_gates
    accepted = all(seal_gates.values()) and all(external_gates.values()) and runtime_report["accepted"]
    report = {
        "schema": 1, "status": "accepted-chain-development" if accepted else "rejected", "accepted": accepted,
        "models": {
            "drive": {"path": str(drive_path), "sha256": sha256(drive_path), "parameters": sum(p.numel() for p in drive.parameters())},
            "reverb": {"path": str(reverb_path), "sha256": sha256(reverb_path), "parameters": sum(p.numel() for p in reverb.parameters())},
        },
        "data": {
            "synthetic": {"fit": args.train, "calibrate": args.calibrate, "develop": args.develop, "seal": args.seal, "forward_domain": "final_challenge", "clean_sources": source_audit},
            "real_fit": {"dataset": "DAFx Guitar Effects Chains Dataset 3", "split": "train", "examples": args.real, "controls": "unknown", "used_for_seal": False},
            "external": {"dataset": "DAFx Guitar Effects Chains Dataset 3", "license": "CC-BY-4.0", "examples": outside["examples"], "start": 64},
            "source_read_only": True, "physical_audio_devices_used": False, "rendered_audio_retained": False,
        },
        "development": development, "development_gates": development_gates,
        "strength": {"drive": drive_strength, "reverb": reverb_strength, "calibration": strength_report},
        "seal": sealed, "seal_gates": seal_gates, "external": outside, "external_gates": external_gates, "runtime": runtime_report,
        "order": {"forward_supported": [["reverb", "drive"], ["drive", "reverb"]], "restore": "reverse the selected forward order", "unknown_order_policy": "score candidates then abstain on weak margin"},
        "quality": {
            "source_audio_modified": False, "preserve_frames": True, "preserve_sample_rate": True,
            "automatic_normalization": False, "automatic_limiting": False, "automatic_dither": False, "lossy_reencoding": False,
            "clean_target": "paired original per example; no canonical clean timbre",
        },
        "limitations": [
            "waveform chain seal is controllable-forward development evidence",
            "DAFx test clean performances overlap the prior synthetic source pool; this is an independent effect-rendering check, not a pristine source seal",
            "physical-device and broad effect-family seals remain required",
        ],
        "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest(),
    }
    (args.output / "valid.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": accepted, "seal": sealed, "external": outside, "runtime": runtime_report}, sort_keys=True))
    if not accepted:
        drive_path.unlink(missing_ok=True); reverb_path.unlink(missing_ok=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
