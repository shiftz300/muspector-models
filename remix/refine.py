"""Refine Stage-5 candidates against aligned quality and hard tails."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .clean import _pre, _spectrum
from .cnet import ConditionalNet
from .forward_chain import ForwardChainRuntime
from .oracle import evaluate, examples, gates, loss as direct_loss
from .stages import CORPUS, sha256, state
from .tnet import TimeNet


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "cycles/stage6.json"
BASELINE = ROOT / "runs/stage/model5"
RUN = ROOT / "runs/stage/model6"


def load(path: Path) -> nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    architecture = payload.get("architecture")
    if architecture == "conditional-time-tcn":
        net = TimeNet(int(payload["channels"]), tuple(payload["dilations"]))
    elif architecture == "conditional-complex-stft":
        net = ConditionalNet(int(payload["channels"]), int(payload["n_fft"]), int(payload["hop"]), tuple(tuple(value) for value in payload["dilations"]))
    else:
        raise ValueError(f"unsupported candidate architecture: {architecture}")
    net.load_state_dict(payload["state_dict"])
    return net


def loss(net: nn.Module, source: torch.Tensor, target: torch.Tensor, controls: torch.Tensor, kind: str) -> torch.Tensor:
    prediction = net(source, controls)
    direct = direct_loss(net, source, target, controls, kind)
    scale = ((prediction * target).sum(1) / prediction.square().sum(1).clamp_min(1.0e-6)).detach()
    aligned = prediction * scale[:, None]
    energy = target.square().mean(1).clamp_min(1.0e-7)
    aligned_rows = (aligned - target).square().mean(1) / energy
    hard = torch.topk(aligned_rows, max(1, len(aligned_rows) // 2)).values.mean()
    correlation = (aligned * target).sum(1) / (aligned.square().sum(1).sqrt() * target.square().sum(1).sqrt()).clamp_min(1.0e-6)
    aligned_l1 = torch.nn.functional.l1_loss(aligned, target)
    emphasized = torch.nn.functional.l1_loss(_pre(aligned), _pre(target))
    spectral = _spectrum(aligned, target)
    peak_rows = (aligned.abs().amax(1) - target.abs().amax(1)).abs()
    hard_peak = torch.topk(peak_rows, max(1, len(peak_rows) // 2)).values.mean()
    direct_weight = 0.35 if kind == "drive" else 0.65
    return (
        direct_weight * direct + 0.25 * aligned_rows.mean() + 0.15 * hard
        + 0.20 * (1.0 - correlation).mean() + 0.30 * aligned_l1
        + 0.10 * emphasized + 0.10 * spectral + 0.50 * peak_rows.mean() + 0.50 * hard_peak
    )


def rank(report: dict) -> float:
    """Select for the worst normalized admission margin, then overall quality."""

    ratios = (
        report["esr_improvement"] / 0.15,
        report["aligned_esr_improvement"] / 0.15,
        report["sidr_improvement_db"] / 1.0,
        report["closure"] / 0.15,
        report["correction_ratio"] / 0.25,
        report["correction_direction"] / 0.50,
        report["audible_fraction"] / 0.50,
        report["baseline_aligned_esr_p95"] / max(report["restored_aligned_esr_p95"], 1.0e-12),
        report["baseline_aligned_peak_p95"] / max(report["restored_aligned_peak_p95"], 1.0e-12),
    )
    return min(ratios) + 0.02 * sum(min(value, 2.0) for value in ratios)


def train(kind: str, net: nn.Module, fit: list[dict], calibration: list[dict], target: torch.device, epochs: int, batch: int, rate: float) -> tuple[nn.Module, list[dict]]:
    optimizer = torch.optim.AdamW(net.parameters(), lr=rate, weight_decay=1.0e-5)
    loader = DataLoader(fit, batch_size=batch, shuffle=True, num_workers=0)
    initial = evaluate(net, calibration, target); best_score, best_state, stale = rank(initial), state(net), 0
    history = [{"epoch": 0, "loss": None, "score": best_score, "calibration": initial}]
    print(json.dumps({"stage": kind, **history[-1]}), flush=True)
    for epoch in range(1, epochs + 1):
        net.train(); values = []
        for item in loader:
            value = loss(net, item["source"].to(target), item["target"].to(target), item["controls"].to(target), kind)
            if not torch.isfinite(value):
                raise FloatingPointError(f"{kind} refine loss became non-finite")
            optimizer.zero_grad(set_to_none=True); value.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); optimizer.step()
            values.append(float(value.detach().cpu()))
        report = evaluate(net, calibration, target); score = rank(report)
        history.append({"epoch": epoch, "loss": float(np.mean(values)), "score": score, "calibration": report})
        print(json.dumps({"stage": kind, **history[-1]}), flush=True)
        if score > best_score:
            best_score, best_state, stale = score, state(net), 0
        else:
            stale += 1
        if epoch >= 8 and stale >= 4:
            break
    net.load_state_dict(best_state)
    return net, history


def save(net: nn.Module, source: Path, output: Path) -> None:
    payload = torch.load(source, map_location="cpu", weights_only=True)
    payload["state_dict"] = state(net)
    payload["initialization_sha256"] = sha256(source)
    payload["refinement"] = "aligned-hard-tail"
    torch.save(payload, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, default=CYCLE); parser.add_argument("--corpus", type=Path, default=CORPUS); parser.add_argument("--baseline", type=Path, default=BASELINE); parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--train", type=int, default=640); parser.add_argument("--calibrate", type=int, default=128); parser.add_argument("--valid", type=int, default=160); parser.add_argument("--drive-epochs", type=int, default=14); parser.add_argument("--reverb-epochs", type=int, default=14); parser.add_argument("--batch", type=int, default=8); parser.add_argument("--drive-rate", type=float, default=2e-5); parser.add_argument("--reverb-rate", type=float, default=1e-4); parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to replace refine run {args.output}")
    cycle_bytes = args.cycle.read_bytes(); cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "stage" or cycle.get("round") != 6 or cycle.get("status") != "planned": raise ValueError("invalid stage6 cycle")
    for kind in ("drive", "reverb"):
        if sha256(args.baseline / f"{kind}.candidate.pt") != cycle["baseline"][f"{kind}_sha256"]: raise ValueError(f"{kind} candidate changed")
    args.output.mkdir(parents=True); torch.manual_seed(20261030); np.random.seed(20261030); torch.set_num_threads(args.threads)
    target = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); runtime = ForwardChainRuntime()
    candidates, reports, checks = {}, {}, {}
    for offset, kind in enumerate(("drive", "reverb")):
        frames = 8192 if kind == "drive" else 16384; profile = "tone" if kind == "drive" else "broad"
        fit = examples(runtime, args.corpus, kind, "train", args.train, frames, 20261070 + offset, profile)
        calibration = examples(runtime, args.corpus, kind, "calibrate", args.calibrate, frames, 20261080 + offset, profile)
        source = args.baseline / f"{kind}.candidate.pt"; net = load(source).to(target)
        net, history = train(kind, net, fit, calibration, target, args.drive_epochs if kind == "drive" else args.reverb_epochs, args.batch, args.drive_rate if kind == "drive" else args.reverb_rate)
        valid = examples(runtime, args.corpus, kind, "valid", args.valid, frames * 2, 20261090 + offset, profile)
        report = evaluate(net, valid, target); check = gates(report)
        candidates[kind], reports[kind], checks[kind] = net.cpu(), report, check
        (args.output / f"{kind}.history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
        save(candidates[kind], source, args.output / f"{kind}.candidate.pt")
        print(json.dumps({"stage": kind, "valid": report, "gates": check}), flush=True)
    accepted = all(all(values.values()) for values in checks.values())
    if accepted:
        for kind in candidates: (args.output / f"{kind}.candidate.pt").rename(args.output / f"{kind}.pt")
    result = {"schema": 1, "status": "accepted-oracle" if accepted else "rejected-oracle", "accepted": accepted, "reports": reports, "gates": checks, "artifacts": {kind: {"path": f"{kind}.{'pt' if accepted else 'candidate.pt'}", "sha256": sha256(args.output / f"{kind}.{'pt' if accepted else 'candidate.pt'}")} for kind in candidates}, "training": {"drive_profile": "tone-balanced", "loss": "aligned-hard-tail", "shared_optimizer": False, "chain_gradient": False}, "data": {"source_read_only": True, "physical_audio_devices_used": False, "rendered_audio_retained": False}, "quality": cycle["quality"], "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest()}
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps({"accepted": accepted, "output": str(args.output)}), flush=True)
    if not accepted: raise SystemExit(2)


if __name__ == "__main__": main()
