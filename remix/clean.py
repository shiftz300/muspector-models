"""Train and evaluate the first wet-only RAT-to-clean restoration model."""

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

from .asrnn_data import LICENSE, RECORD_URL, audit_asrnn, rat_files, read_rat_pair
from .net import SpectralNet
from .train_asrnn_rat_adapter import _partition


RATE = 48_000
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)


class Block(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.filter = nn.Conv1d(channels, channels * 2, 9, padding=dilation * 4, dilation=dilation)
        self.mix = nn.Conv1d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        left, right = self.filter(value).chunk(2, 1)
        return value + self.mix(torch.tanh(left) * torch.sigmoid(right))


class CleanNet(nn.Module):
    """Offline residual TCN; strength zero is an exact bit-preserving bypass."""

    def __init__(self, channels: int = 24, dilations: tuple[int, ...] = DILATIONS):
        super().__init__()
        self.channels = int(channels)
        self.dilations = tuple(map(int, dilations))
        self.stem = nn.Conv1d(1, channels, 15, padding=7)
        self.blocks = nn.ModuleList(Block(channels, value) for value in self.dilations)
        self.head = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, wet: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        if wet.ndim != 2 or not torch.isfinite(wet).all():
            raise ValueError("wet audio must be finite [batch,time]")
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError("restoration strength must be between zero and one")
        if strength == 0.0:
            return wet
        value = self.stem(wet.unsqueeze(1))
        for block in self.blocks:
            value = block(value)
        residual = self.head(torch.tanh(value)).squeeze(1)
        return wet + residual * float(strength)


class Windows(Dataset):
    def __init__(self, paths: list[Path], frames: int, clips: int):
        if not paths or frames <= 0 or clips <= 0:
            raise ValueError("invalid restoration dataset")
        self.paths, self.frames, self.clips = paths, frames, clips

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        dry, wet, _ = read_rat_pair(self.paths[index])
        if len(dry) < self.frames:
            raise ValueError("RAT pair is shorter than one restoration window")
        starts = np.linspace(0, len(dry) - self.frames, max(self.clips * 4, self.clips), dtype=int)
        energy = np.asarray([np.mean(np.square(wet[start : start + self.frames], dtype=np.float64)) for start in starts])
        selected = np.sort(starts[np.argsort(energy)[-self.clips :]])
        return (
            torch.from_numpy(np.stack([wet[start : start + self.frames] for start in selected]).copy()),
            torch.from_numpy(np.stack([dry[start : start + self.frames] for start in selected]).copy()),
        )


def _pre(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[:, :1], value[:, 1:] - 0.95 * value[:, :-1]), 1)


def _spectrum(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = prediction.square().mean() * 0.0
    for frames in (256, 1024):
        count = prediction.shape[1] // frames
        left = torch.fft.rfft(prediction[:, : count * frames].reshape(-1, frames))
        right = torch.fft.rfft(target[:, : count * frames].reshape(-1, frames))
        result = result + torch.nn.functional.l1_loss(torch.log1p(left.abs()), torch.log1p(right.abs()))
    return result / 2.0


def loss(model: CleanNet, wet: torch.Tensor, dry: torch.Tensor) -> tuple[torch.Tensor, dict]:
    prediction = model(wet)
    scale = ((prediction * dry).sum(1) / prediction.square().sum(1).clamp_min(1e-6)).detach()
    aligned = prediction * scale[:, None]
    mae = torch.nn.functional.l1_loss(aligned, dry)
    emphasized = torch.nn.functional.l1_loss(_pre(aligned), _pre(dry))
    spectral = _spectrum(aligned, dry)
    global_esr = (aligned - dry).square().sum() / dry.square().sum().clamp_min(1e-6)
    dot = (prediction * dry).sum(1)
    correlation = dot / (prediction.square().sum(1).sqrt() * dry.square().sum(1).sqrt()).clamp_min(1e-6)
    correlation_loss = (1.0 - correlation).mean()
    level = torch.abs(
        torch.log((prediction.square().mean(1) + 1e-8) / (dry.square().mean(1) + 1e-8))
    ).mean()
    peak = (aligned.abs().amax(1) - dry.abs().amax(1)).abs().mean()
    total = mae + 0.25 * emphasized + 0.10 * spectral + 0.10 * global_esr + 0.50 * correlation_loss + 0.001 * level + 0.10 * peak
    return total, {"mae": mae, "pre": emphasized, "spectral": spectral, "esr": global_esr, "correlation": correlation.mean(), "level": level, "peak": peak}


def eligible(paths: list[Path], minimum_ratio: float) -> tuple[list[Path], dict]:
    accepted_paths, ratios = [], []
    for path in paths:
        dry, wet, _ = read_rat_pair(path)
        dry_rms = float(np.sqrt(np.mean(np.square(dry[1024:], dtype=np.float64))))
        wet_rms = float(np.sqrt(np.mean(np.square(wet[1024:], dtype=np.float64))))
        ratio = wet_rms / max(dry_rms, 1e-12)
        ratios.append(ratio)
        if ratio >= minimum_ratio:
            accepted_paths.append(path)
    return accepted_paths, {
        "total": len(paths),
        "eligible": len(accepted_paths),
        "coverage": len(accepted_paths) / len(paths),
        "minimum_wet_to_dry_rms_ratio": minimum_ratio,
        "ratio_minimum": min(ratios),
        "ratio_median": float(np.median(ratios)),
        "ratio_maximum": max(ratios),
        "runtime_available": False,
    }


def _sidr(prediction: np.ndarray, target: np.ndarray) -> float:
    target64, prediction64 = target.astype(np.float64), prediction.astype(np.float64)
    projection = np.dot(prediction64, target64) / max(np.dot(target64, target64), 1e-12) * target64
    noise = prediction64 - projection
    return float(10.0 * np.log10(max(np.dot(projection, projection), 1e-12) / max(np.dot(noise, noise), 1e-12)))


def _align(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    left, right = prediction.astype(np.float64), target.astype(np.float64)
    gain = np.dot(left, right) / max(np.dot(left, left), 1e-12)
    return (left * gain).astype(np.float32)


@torch.inference_mode()
def evaluate(model: CleanNet, paths: list[Path], target: torch.device, strength: float = 1.0) -> dict:
    rows = []
    source_mutations = nonfinite = geometry = 0
    model.eval()
    for index, path in enumerate(paths):
        dry, wet, _ = read_rat_pair(path)
        before = hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest()
        prediction = model(torch.from_numpy(wet).to(target).unsqueeze(0), strength=strength)[0].cpu().numpy()
        baseline_aligned = _align(wet, dry)
        restored_aligned = _align(prediction, dry)
        if prediction.shape != wet.shape:
            geometry += 1
        nonfinite += int(not np.isfinite(prediction).all())
        source_mutations += int(before != hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest())
        energy = max(float(np.mean(np.square(dry, dtype=np.float64))), 1e-12)
        rows.append({
            "file": path.name,
            "baseline_esr": float(np.mean(np.square(wet - dry, dtype=np.float64)) / energy),
            "restored_esr": float(np.mean(np.square(prediction - dry, dtype=np.float64)) / energy),
            "baseline_mae": float(np.mean(np.abs(wet.astype(np.float64) - dry))),
            "restored_mae": float(np.mean(np.abs(prediction.astype(np.float64) - dry))),
            "baseline_sidr": _sidr(wet, dry),
            "restored_sidr": _sidr(prediction, dry),
            "baseline_peak_error": abs(float(np.max(np.abs(wet))) - float(np.max(np.abs(dry)))),
            "restored_peak_error": abs(float(np.max(np.abs(prediction))) - float(np.max(np.abs(dry)))),
            "baseline_aligned_esr": float(np.mean(np.square(baseline_aligned - dry, dtype=np.float64)) / energy),
            "restored_aligned_esr": float(np.mean(np.square(restored_aligned - dry, dtype=np.float64)) / energy),
            "baseline_aligned_mae": float(np.mean(np.abs(baseline_aligned.astype(np.float64) - dry))),
            "restored_aligned_mae": float(np.mean(np.abs(restored_aligned.astype(np.float64) - dry))),
            "baseline_aligned_peak_error": abs(float(np.max(np.abs(baseline_aligned))) - float(np.max(np.abs(dry)))),
            "restored_aligned_peak_error": abs(float(np.max(np.abs(restored_aligned))) - float(np.max(np.abs(dry)))),
        })
        if index % 32 == 31:
            print(json.dumps({"stage": "evaluate", "completed": index + 1, "total": len(paths)}), flush=True)
    def mean(name): return float(np.mean([row[name] for row in rows]))
    def p95(name): return float(np.quantile([row[name] for row in rows], 0.95))
    metrics = {
        "examples": len(rows),
        "baseline_global_esr": mean("baseline_esr"),
        "restored_global_esr": mean("restored_esr"),
        "esr_improvement": 1.0 - mean("restored_esr") / max(mean("baseline_esr"), 1e-12),
        "baseline_mae": mean("baseline_mae"),
        "restored_mae": mean("restored_mae"),
        "mae_improvement": 1.0 - mean("restored_mae") / max(mean("baseline_mae"), 1e-12),
        "sidr_improvement_db": mean("restored_sidr") - mean("baseline_sidr"),
        "baseline_esr_p95": p95("baseline_esr"),
        "restored_esr_p95": p95("restored_esr"),
        "baseline_peak_error_p95": p95("baseline_peak_error"),
        "restored_peak_error_p95": p95("restored_peak_error"),
        "baseline_aligned_esr": mean("baseline_aligned_esr"),
        "restored_aligned_esr": mean("restored_aligned_esr"),
        "aligned_esr_improvement": 1.0 - mean("restored_aligned_esr") / max(mean("baseline_aligned_esr"), 1e-12),
        "baseline_aligned_mae": mean("baseline_aligned_mae"),
        "restored_aligned_mae": mean("restored_aligned_mae"),
        "aligned_mae_improvement": 1.0 - mean("restored_aligned_mae") / max(mean("baseline_aligned_mae"), 1e-12),
        "baseline_aligned_esr_p95": p95("baseline_aligned_esr"),
        "restored_aligned_esr_p95": p95("restored_aligned_esr"),
        "baseline_aligned_peak_error_p95": p95("baseline_aligned_peak_error"),
        "restored_aligned_peak_error_p95": p95("restored_aligned_peak_error"),
        "source_mutations": source_mutations,
        "nonfinite_outputs": nonfinite,
        "geometry_errors": geometry,
    }
    return {"metrics": metrics, "rows": rows}


def accepted(metrics: dict, gates: dict) -> tuple[bool, list[str]]:
    failures = []
    for name in ("esr_improvement", "mae_improvement", "aligned_esr_improvement", "aligned_mae_improvement", "sidr_improvement_db"):
        if name in gates and metrics[name] < gates[name]: failures.append(name)
    if "p95_ratio_max" in gates and metrics["restored_esr_p95"] > metrics["baseline_esr_p95"] * gates["p95_ratio_max"]: failures.append("p95_esr")
    if "peak_ratio_max" in gates and metrics["restored_peak_error_p95"] > metrics["baseline_peak_error_p95"] * gates["peak_ratio_max"]: failures.append("peak_p95")
    if "aligned_p95_ratio_max" in gates and metrics["restored_aligned_esr_p95"] > metrics["baseline_aligned_esr_p95"] * gates["aligned_p95_ratio_max"]: failures.append("aligned_p95_esr")
    if "aligned_peak_ratio_max" in gates and metrics["restored_aligned_peak_error_p95"] > metrics["baseline_aligned_peak_error_p95"] * gates["aligned_peak_ratio_max"]: failures.append("aligned_peak_p95")
    for name in ("source_mutations", "nonfinite_outputs", "geometry_errors"):
        if metrics[name] != 0: failures.append(name)
    if metrics["coverage"] < gates["coverage"]: failures.append("coverage")
    return not failures, failures


def train(args) -> dict:
    cycle_bytes = args.cycle.read_bytes()
    cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "clean1" or cycle.get("status") != "planned":
        raise ValueError("clean1 cycle must be predeclared and planned")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace restoration run {args.output}")
    args.output.mkdir(parents=True)
    (args.output / "lock.json").write_text(json.dumps({"schema": 1, "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest(), "official_eval_opened": False}, indent=2) + "\n")

    audit = audit_asrnn(args.corpus, scan_audio=True)
    fit_all, calibration_all = _partition(rat_files(args.corpus, "train"))
    fit, fit_eligibility = eligible(fit_all, cycle["eligibility"]["minimum_wet_to_dry_rms_ratio"])
    calibration, calibration_eligibility = eligible(calibration_all, cycle["eligibility"]["minimum_wet_to_dry_rms_ratio"])
    target = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else "cpu" if args.device == "auto" else args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    architecture = cycle["architecture"]["model"]
    if architecture == "residual-tcn":
        model = CleanNet(args.channels).to(target)
    elif architecture == "complex-stft":
        model = SpectralNet(args.channels).to(target)
    else:
        raise ValueError(f"unsupported restoration architecture: {architecture}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.rate, weight_decay=1e-5)
    loader = DataLoader(Windows(fit, args.frames, args.clips), batch_size=args.batch, shuffle=True, num_workers=0)
    history = []
    best, state = float("inf"), None
    for epoch in range(args.epochs):
        model.train(); totals = {}
        for wet, dry in loader:
            wet = wet.flatten(0, 1).to(target); dry = dry.flatten(0, 1).to(target)
            value, parts = loss(model, wet, dry)
            optimizer.zero_grad(set_to_none=True); value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            for name, item in {"loss": value, **parts}.items(): totals.setdefault(name, []).append(float(item.detach().cpu()))
        report = evaluate(model, calibration, target)["metrics"]
        score = report["restored_aligned_esr"] - 0.01 * report["sidr_improvement_db"]
        if score < best:
            best = score; state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        row = {"epoch": epoch + 1, "train": {name: float(np.mean(values)) for name, values in totals.items()}, "calibration": report}
        history.append(row); (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps({"stage": "train", **row}), flush=True)
    if state is None: raise RuntimeError("restoration training produced no checkpoint")
    model.load_state_dict(state)
    payload = {"schema": 1, "architecture": architecture, "sample_rate": RATE, "channels": args.channels, "device": "rat", "kind": "clean", "state_dict": state, "dataset": RECORD_URL, "license": LICENSE}
    if isinstance(model, CleanNet): payload["dilations"] = list(model.dilations)
    if isinstance(model, SpectralNet): payload.update({"n_fft": model.n_fft, "hop": model.hop})
    checkpoint = args.output / "clean.pt"; torch.save(payload, checkpoint)
    calibration_report = evaluate(model, calibration, target)
    lock = json.loads((args.output / "lock.json").read_text()); lock["official_eval_opened"] = True; (args.output / "lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    development_all = rat_files(args.corpus, "eval")
    development_paths, development_eligibility = eligible(development_all, cycle["eligibility"]["minimum_wet_to_dry_rms_ratio"])
    development = evaluate(model, development_paths, target)
    development["metrics"]["coverage"] = development_eligibility["coverage"]
    passed, failures = accepted(development["metrics"], cycle["gates"])
    report = {
        "schema": 1, "status": "accepted-development" if passed else "rejected", "accepted": passed, "failures": failures,
        "model": {"architecture": architecture, "parameters": sum(p.numel() for p in model.parameters()), "checkpoint": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
        "data": {"fit": len(fit), "calibration": len(calibration), "development": len(development["rows"]), "record": RECORD_URL, "license": LICENSE, "source_read_only": True, "physical_audio_devices_used": False, "eligibility": {"fit": fit_eligibility, "calibration": calibration_eligibility, "development": development_eligibility}},
        "calibration": calibration_report["metrics"], "development": development["metrics"], "gates": cycle["gates"],
        "quality": {"source_audio_modified": False, "preserve_frames": True, "preserve_channels": True, "preserve_sample_rate": True, "automatic_normalization": False, "automatic_limiting": False, "automatic_dither": False, "lossy_reencoding": False},
        "limitations": ["ASRNN ProCo RAT only", "wet-only development result", "oracle Dry/Wet energy eligibility is not a deployable runtime confidence gate", "official eval is development evidence, not a locked final seal", "no cross-device restoration claim"],
        "audit": audit,
    }
    (args.output / "valid.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/asrnn-physical-effects"))
    parser.add_argument("--cycle", type=Path, default=Path("cycles/clean1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/clean/model1"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--clips", type=int, default=4)
    parser.add_argument("--frames", type=int, default=8192)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    print(json.dumps(train(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
