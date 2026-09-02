"""Train the Stage-4 control-conditioned inverse oracle on labeled renders."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .chain_report import source_domains
from .clean import _pre, _spectrum
from .cnet import ConditionalNet
from .forward_chain import ForwardChainRuntime
from .forward_drive import _segment
from .net import DILATIONS
from .spec import ChainSpec, Drive, Reverb
from .stages import CORPUS, REVERB_DILATIONS, metric, sha256, state


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "cycles/stage4.json"
BASELINE = ROOT / "runs/stage/model2"
RUN = ROOT / "runs/stage/model4"


def effect(kind: str, rng: random.Random, profile: str = "broad") -> tuple[Drive | Reverb, np.ndarray]:
    """Sample an intentionally audible effect and its normalized controls."""

    if kind == "drive":
        if profile == "tone":
            level_db = rng.uniform(-1.0, 1.0) if rng.random() < 0.75 else rng.uniform(-6.0, 6.0)
            value = Drive(rng.uniform(10.0, 30.0), rng.uniform(0.05, 0.95), level_db)
        elif profile == "broad":
            value = Drive(rng.uniform(6.0, 30.0), rng.uniform(0.05, 0.95), rng.uniform(-6.0, 6.0))
        else:
            raise ValueError(f"unknown oracle profile: {profile}")
        controls = (value.gain_db / 30.0, value.tone, (value.level_db + 18.0) / 30.0)
    elif kind == "reverb":
        value = Reverb(0.3 * (6.0 / 0.3) ** rng.random(), rng.uniform(0.05, 0.95), rng.uniform(0.20, 0.65))
        controls = (math.log(value.decay_s / 0.2) / math.log(8.0 / 0.2), value.damping, value.mix / 0.7)
    else:
        raise ValueError(f"unknown oracle stage: {kind}")
    return value, np.asarray(controls, dtype=np.float32)


@torch.inference_mode()
def examples(
    runtime: ForwardChainRuntime,
    corpus: Path,
    kind: str,
    split: str,
    count: int,
    frames: int,
    seed: int,
    profile: str = "broad",
) -> list[dict[str, torch.Tensor | str]]:
    domains = source_domains(corpus, split)
    names = sorted(domains)
    if not names:
        raise RuntimeError(f"no clean source domains for {split}")
    rows = []
    for index in range(count):
        rng = random.Random(seed + index * 104_729)
        domain = names[index % len(names)]
        paths = domains[domain]
        clean = _segment(paths[rng.randrange(len(paths))], frames, rng)
        peak = max(float(np.max(np.abs(clean))), 1.0e-6)
        clean = np.asarray(clean * (rng.uniform(0.08, 0.28) / peak), dtype=np.float32)
        selected, controls = effect(kind, rng, profile)
        wet = runtime.render(clean, ChainSpec((selected,)))
        rows.append({
            "source": torch.from_numpy(wet.copy()),
            "target": torch.from_numpy(clean.copy()),
            "controls": torch.from_numpy(controls),
            "domain": domain,
        })
    return rows


def model(kind: str, baseline: Path) -> ConditionalNet:
    payload = torch.load(baseline / f"{kind}.pt", map_location="cpu", weights_only=True)
    dilations = DILATIONS if kind == "drive" else REVERB_DILATIONS
    result = ConditionalNet(int(payload["channels"]), int(payload["n_fft"]), int(payload["hop"]), dilations)
    old = payload["state_dict"]
    current = result.state_dict()
    with torch.no_grad():
        current["stem.weight"][:, :3].copy_(old["stem.weight"])
        current["stem.weight"][:, 3:].zero_()
        current["stem.bias"].copy_(old["stem.bias"])
        for name in current:
            if name.startswith("blocks.") or name.startswith("head."):
                current[name].copy_(old[name])
    result.load_state_dict(current)
    return result


def loss(net: ConditionalNet, source: torch.Tensor, target: torch.Tensor, controls: torch.Tensor, kind: str) -> torch.Tensor:
    prediction = net(source, controls)
    wanted = target - source
    correction = prediction - source
    wanted_l1 = wanted.abs().mean(1).clamp_min(1.0e-4)
    wanted_energy = wanted.square().mean(1).clamp_min(1.0e-7)
    delta_l1 = ((correction - wanted).abs().mean(1) / wanted_l1).mean()
    delta_esr = ((correction - wanted).square().mean(1) / wanted_energy).mean()
    target_energy = target.square().mean(1).clamp_min(1.0e-7)
    direct_esr = ((prediction - target).square().mean(1) / target_energy).mean()
    direct_l1 = torch.nn.functional.l1_loss(prediction, target)
    spectral = _spectrum(prediction, target)
    emphasized = torch.nn.functional.l1_loss(_pre(prediction), _pre(target))
    value = (
        0.50 * delta_l1 + 0.20 * torch.log1p(delta_esr)
        + 0.20 * torch.log1p(direct_esr) + 0.50 * direct_l1
        + 0.10 * spectral + 0.10 * emphasized
    )
    if kind == "reverb":
        predicted_envelope = torch.nn.functional.avg_pool1d(prediction.abs().unsqueeze(1), 1024, 512).squeeze(1)
        target_envelope = torch.nn.functional.avg_pool1d(target.abs().unsqueeze(1), 1024, 512).squeeze(1)
        value = value + 0.10 * torch.nn.functional.l1_loss(predicted_envelope, target_envelope)
    return value


@torch.inference_mode()
def evaluate(net: ConditionalNet, rows: list[dict], target: torch.device, strength: float = 1.0) -> dict:
    net.eval(); baselines, predictions, targets = [], [], []
    for first in range(0, len(rows), 8):
        batch = rows[first:first + 8]
        source = torch.stack([row["source"] for row in batch]).to(target)
        controls = torch.stack([row["controls"] for row in batch]).to(target)
        prediction = net(source, controls, strength=strength).cpu().numpy()
        baselines.extend(row["source"].numpy() for row in batch)
        predictions.extend(prediction)
        targets.extend(row["target"].numpy() for row in batch)
    report = metric(baselines, predictions, targets)
    report["domains"] = sorted({str(row["domain"]) for row in rows})
    report["strength"] = float(strength)
    return report


def rank(report: dict) -> float:
    return (
        report["esr_improvement"] + report["aligned_esr_improvement"]
        + 0.10 * report["sidr_improvement_db"] + report["closure"]
        + 0.25 * report["correction_direction"] + 0.10 * min(report["correction_ratio"], 1.0)
    )


def train(
    kind: str,
    net: ConditionalNet,
    fit: list[dict],
    calibration: list[dict],
    target: torch.device,
    epochs: int,
    batch: int,
    rate: float,
) -> tuple[ConditionalNet, list[dict]]:
    optimizer = torch.optim.AdamW(net.parameters(), lr=rate, weight_decay=1.0e-5)
    loader = DataLoader(fit, batch_size=batch, shuffle=True, num_workers=0)
    initial = evaluate(net, calibration, target)
    best_score, best_state, stale = rank(initial), state(net), 0
    history = [{"epoch": 0, "loss": None, "score": best_score, "calibration": initial}]
    print(json.dumps({"stage": kind, **history[-1]}), flush=True)
    for epoch in range(1, epochs + 1):
        net.train(); values = []
        for item in loader:
            value = loss(net, item["source"].to(target), item["target"].to(target), item["controls"].to(target), kind)
            if not torch.isfinite(value):
                raise FloatingPointError(f"{kind} loss became non-finite")
            optimizer.zero_grad(set_to_none=True); value.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step()
            values.append(float(value.detach().cpu()))
        if not all(torch.isfinite(parameter).all() for parameter in net.parameters()):
            net.load_state_dict(best_state)
            history.append({"epoch": epoch, "loss": float(np.mean(values)), "score": None, "nonfinite": True})
            print(json.dumps({"stage": kind, **history[-1]}), flush=True)
            break
        report = evaluate(net, calibration, target); score = rank(report)
        history.append({"epoch": epoch, "loss": float(np.mean(values)), "score": score, "calibration": report})
        print(json.dumps({"stage": kind, **history[-1]}), flush=True)
        if score > best_score:
            best_score, best_state, stale = score, state(net), 0
        else:
            stale += 1
        if epoch >= 10 and stale >= 5:
            break
    net.load_state_dict(best_state)
    return net, history


def gates(report: dict) -> dict[str, bool]:
    return {
        "absolute_esr": report["esr_improvement"] >= 0.15,
        "aligned_esr": report["aligned_esr_improvement"] >= 0.15,
        "sidr": report["sidr_improvement_db"] >= 1.0,
        "closure": report["closure"] >= 0.15,
        "correction": report["correction_ratio"] >= 0.25,
        "direction": report["correction_direction"] >= 0.50,
        "audible": report["audible_fraction"] >= 0.50,
        "tail": report["restored_aligned_esr_p95"] <= report["baseline_aligned_esr_p95"],
        "peak": report["restored_aligned_peak_p95"] <= report["baseline_aligned_peak_p95"],
    }


def save(net: ConditionalNet, path: Path, kind: str, baseline: Path) -> None:
    torch.save({
        "schema": 1,
        "architecture": "conditional-complex-stft",
        "sample_rate": 48_000,
        "channels": net.channels,
        "n_fft": net.n_fft,
        "hop": net.hop,
        "dilations": [list(value) for value in net.dilations],
        "stage": kind,
        "controls": [f"{kind}.{name}" for name in (("gain_db", "tone", "level_db") if kind == "drive" else ("decay_s", "damping", "mix"))],
        "target": "immediate-predecessor",
        "state_dict": state(net),
        "initialization_sha256": sha256(baseline / f"{kind}.pt"),
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, default=CYCLE); parser.add_argument("--corpus", type=Path, default=CORPUS); parser.add_argument("--baseline", type=Path, default=BASELINE); parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--train", type=int, default=640); parser.add_argument("--calibrate", type=int, default=128); parser.add_argument("--valid", type=int, default=160); parser.add_argument("--frames", type=int, default=16_384); parser.add_argument("--drive-epochs", type=int, default=18); parser.add_argument("--reverb-epochs", type=int, default=22); parser.add_argument("--batch", type=int, default=8); parser.add_argument("--rate", type=float, default=3.0e-4); parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace oracle run {args.output}")
    cycle_bytes = args.cycle.read_bytes(); cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "stage" or cycle.get("round") != 4 or cycle.get("status") != "planned":
        raise ValueError("invalid stage4 cycle")
    for kind in ("drive", "reverb"):
        if sha256(args.baseline / f"{kind}.pt") != cycle["baseline"][f"{kind}_sha256"]:
            raise ValueError(f"{kind} baseline changed")
    args.output.mkdir(parents=True)
    torch.manual_seed(20261010); np.random.seed(20261010); torch.set_num_threads(args.threads)
    target = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    runtime = ForwardChainRuntime(); models, histories, reports, checks = {}, {}, {}, {}
    for offset, kind in enumerate(("drive", "reverb")):
        fit = examples(runtime, args.corpus, kind, "train", args.train, args.frames, 20261010 + offset)
        calibration = examples(runtime, args.corpus, kind, "calibrate", args.calibrate, args.frames, 20261020 + offset)
        candidate, history = train(kind, model(kind, args.baseline).to(target), fit, calibration, target, args.drive_epochs if kind == "drive" else args.reverb_epochs, args.batch, args.rate)
        valid = examples(runtime, args.corpus, kind, "valid", args.valid, args.frames * 2, 20261030 + offset)
        report = evaluate(candidate, valid, target); check = gates(report)
        models[kind], histories[kind], reports[kind], checks[kind] = candidate.cpu(), history, report, check
        (args.output / f"{kind}.history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"stage": kind, "valid": report, "gates": check}), flush=True)
    accepted = all(all(value.values()) for value in checks.values())
    if accepted:
        for kind, candidate in models.items():
            save(candidate, args.output / f"{kind}.pt", kind, args.baseline)
    result = {
        "schema": 1,
        "status": "accepted-oracle" if accepted else "rejected-oracle",
        "accepted": accepted,
        "models": {kind: {"path": f"{kind}.pt", "sha256": sha256(args.output / f"{kind}.pt")} for kind in models} if accepted else {},
        "reports": reports,
        "gates": checks,
        "training": {"controls": "oracle", "shared_optimizer": False, "chain_gradient": False, "fit_per_stage": args.train, "calibration_per_stage": args.calibrate},
        "data": {"source_read_only": True, "physical_audio_devices_used": False, "rendered_audio_retained": False},
        "quality": cycle["quality"],
        "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest(),
    }
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": accepted, "output": str(args.output)}), flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
