"""Train a compact anti-blur inverse on real paired ProCo RAT recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .asrnn_data import LICENSE, RECORD_URL, rat_files, read_rat_pair
from .net import SpectralNet
from .clean import eligible
from .restoration_quality import summarize
from .train_asrnn_rat_adapter import _partition


ROOT = Path(__file__).resolve().parents[1]
RATE = 48_000
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


class Block(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.filter = nn.Conv1d(channels, channels * 2, 5, padding=dilation * 2, dilation=dilation, groups=channels)
        self.film = nn.Linear(3, channels * 2)
        self.mix = nn.Conv1d(channels, channels, 1)

    def forward(self, value: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        left, right = self.filter(value).chunk(2, 1)
        scale, bias = self.film(controls).unsqueeze(-1).chunk(2, 1)
        gated = torch.tanh(left * (1.0 + 0.25 * torch.tanh(scale)) + bias) * torch.sigmoid(right)
        return value + self.mix(gated)


class RATNet(nn.Module):
    """Control-conditioned residual; zero head is exact Wet bypass."""

    def __init__(self, channels: int = 24, dilations: tuple[int, ...] = DILATIONS):
        super().__init__()
        self.channels, self.dilations = int(channels), tuple(map(int, dilations))
        self.stem = nn.Conv1d(3, channels, 15, padding=7)
        self.blocks = nn.ModuleList(Block(channels, value) for value in self.dilations)
        self.head = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, wet: torch.Tensor, prior: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        if wet.ndim != 2 or prior.shape != wet.shape or controls.shape != (wet.shape[0], 3):
            raise ValueError("RAT inverse expects wet/prior [batch,time] and controls [batch,3]")
        features = torch.stack((wet, prior, prior - wet), 1)
        value = self.stem(features)
        for block in self.blocks:
            value = block(value, controls)
        return wet + self.head(torch.tanh(value)).squeeze(1)


class Block2(nn.Module):
    def __init__(self, channels: int, dilation: int, groups: int):
        super().__init__()
        self.filter = nn.Conv1d(channels, channels * 2, 5, padding=dilation * 2, dilation=dilation, groups=groups)
        self.film = nn.Linear(3, channels * 2)
        self.mix = nn.Conv1d(channels, channels, 1)

    def forward(self, value: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        left, right = self.filter(value).chunk(2, 1)
        scale, bias = self.film(controls).unsqueeze(-1).chunk(2, 1)
        return value + self.mix(torch.tanh(left * (1.0 + 0.25 * torch.tanh(scale)) + bias) * torch.sigmoid(right))


class RATNet2(nn.Module):
    """Separate level inversion from normalized temporal detail synthesis."""

    def __init__(self, channels: int = 32, groups: int = 4, dilations: tuple[int, ...] = DILATIONS):
        super().__init__()
        if channels % groups or (channels * 2) % groups:
            raise ValueError("RAT v2 channels must be divisible by convolution groups")
        self.channels, self.groups, self.dilations = int(channels), int(groups), tuple(map(int, dilations))
        self.stem = nn.Conv1d(3, channels, 15, padding=7)
        self.blocks = nn.ModuleList(Block2(channels, value, groups) for value in self.dilations)
        self.head = nn.Conv1d(channels, 1, 1)
        self.level = nn.Sequential(nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.level[-1].weight); nn.init.zeros_(self.level[-1].bias)

    def forward(self, wet: torch.Tensor, prior: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        if wet.ndim != 2 or prior.shape != wet.shape or controls.shape != (wet.shape[0], 3):
            raise ValueError("RAT v2 expects wet/prior [batch,time] and controls [batch,3]")
        rms = wet.square().mean(1, keepdim=True).sqrt().clamp_min(1.0e-5)
        features = torch.stack((wet / rms, prior / rms, (prior - wet) / rms), 1)
        value = self.stem(features)
        for block in self.blocks:
            value = block(value, controls)
        gain = torch.exp(self.level(controls).clamp(-4.0, 4.0))
        detail = self.head(torch.tanh(value)).squeeze(1)
        return wet * gain + detail * rms * gain


class Windows(Dataset):
    def __init__(self, paths: list[Path], frames: int, clips: int):
        self.paths, self.frames, self.clips = paths, frames, clips

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        clean, wet, controls = read_rat_pair(self.paths[index])
        starts = np.linspace(0, len(clean) - self.frames, max(12, self.clips * 4), dtype=int)
        pre = np.concatenate((clean[:1], clean[1:] - 0.95 * clean[:-1]))
        scores = np.asarray([np.mean(np.square(pre[start : start + self.frames], dtype=np.float64)) for start in starts])
        selected = np.sort(starts[np.argsort(scores)[-self.clips :]])
        return (
            torch.from_numpy(np.stack([wet[start : start + self.frames] for start in selected]).copy()),
            torch.from_numpy(np.stack([clean[start : start + self.frames] for start in selected]).copy()),
            torch.from_numpy(controls.copy()),
        )


def _prior(path: Path) -> SpectralNet:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = SpectralNet(payload["channels"], payload["n_fft"], payload["hop"])
    model.load_state_dict(payload["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def _pre(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[:, :1], value[:, 1:] - 0.95 * value[:, :-1]), 1)


def _spectrum(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    total = prediction.square().mean() * 0.0
    for size in (256, 1024, 4096):
        count = prediction.shape[1] // size
        left = torch.fft.rfft(prediction[:, : count * size].reshape(-1, size))
        right = torch.fft.rfft(target[:, : count * size].reshape(-1, size))
        scale = right.abs().mean(1, keepdim=True).clamp_min(1.0e-6)
        total = total + torch.nn.functional.l1_loss(torch.log1p(left.abs() / scale), torch.log1p(right.abs() / scale))
    return total / 3.0


def loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    waveform = torch.nn.functional.l1_loss(prediction, target)
    emphasized = torch.nn.functional.l1_loss(_pre(prediction), _pre(target))
    derivative = torch.nn.functional.l1_loss(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1])
    spectral = _spectrum(prediction, target)
    esr = (prediction - target).square().sum() / target.square().sum().clamp_min(1.0e-6)
    peak = (prediction.abs().amax(1) - target.abs().amax(1)).abs().mean()
    total = waveform + 1.0 * emphasized + 0.75 * derivative + 0.20 * spectral + 0.05 * esr + 0.10 * peak
    return total, {"waveform": waveform, "pre": emphasized, "derivative": derivative, "spectral": spectral, "esr": esr, "peak": peak}


def relative_loss(prediction: torch.Tensor, target: torch.Tensor, wet: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Balance audible objectives relative to the unprocessed Wet error."""

    def ratio(value: torch.Tensor, baseline: torch.Tensor, minimum: float) -> torch.Tensor:
        return value / baseline.detach().clamp_min(minimum)

    waveform = ratio(torch.nn.functional.l1_loss(prediction, target), torch.nn.functional.l1_loss(wet, target), 1.0e-4)
    emphasized = ratio(torch.nn.functional.l1_loss(_pre(prediction), _pre(target)), torch.nn.functional.l1_loss(_pre(wet), _pre(target)), 1.0e-5)
    derivative = ratio(
        torch.nn.functional.l1_loss(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1]),
        torch.nn.functional.l1_loss(wet[:, 1:] - wet[:, :-1], target[:, 1:] - target[:, :-1]),
        1.0e-5,
    )
    spectral = ratio(_spectrum(prediction, target), _spectrum(wet, target), 1.0e-3)
    esr = ratio((prediction - target).square().sum() / target.square().sum().clamp_min(1.0e-6), (wet - target).square().sum() / target.square().sum().clamp_min(1.0e-6), 1.0e-3)
    peak_error = (prediction.abs().amax(1) - target.abs().amax(1)).abs().mean()
    wet_peak_error = (wet.abs().amax(1) - target.abs().amax(1)).abs().mean()
    peak = ratio(peak_error, wet_peak_error, 1.0e-4)
    total = waveform + emphasized + 2.0 * derivative + spectral + esr + 0.5 * peak
    return total, {"waveform": waveform, "pre": emphasized, "derivative": derivative, "spectral": spectral, "esr": esr, "peak": peak}


def _attacks(value: torch.Tensor) -> torch.Tensor:
    envelope = torch.nn.functional.avg_pool1d(value.square().unsqueeze(1), 240, 120).clamp_min(1.0e-10).sqrt().squeeze(1)
    return torch.relu(torch.diff(torch.log(envelope + 1.0e-6), dim=1))


def attack_loss(prediction: torch.Tensor, target: torch.Tensor, wet: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Match the actual acceptance metric's pick-attack representation."""

    _, parts = relative_loss(prediction, target, wet)
    attack_error = torch.nn.functional.l1_loss(_attacks(prediction), _attacks(target))
    wet_attack_error = torch.nn.functional.l1_loss(_attacks(wet), _attacks(target))
    attack = attack_error / wet_attack_error.detach().clamp_min(1.0e-5)
    total = 0.5 * parts["waveform"] + parts["pre"] + parts["derivative"] + 3.0 * attack + 0.5 * parts["spectral"] + 0.5 * parts["esr"] + 0.25 * parts["peak"]
    return total, {**parts, "attack": attack}


@torch.inference_mode()
def audit(model: RATNet, prior: SpectralNet, paths: list[Path], device: torch.device, strength: float, limit: int | None = None) -> dict:
    wet_rows, restored_rows, clean_rows = [], [], []
    selected = paths if limit is None else paths[:limit]
    model.eval(); prior.eval()
    for path in selected:
        clean, wet, controls = read_rat_pair(path)
        wet_tensor = torch.from_numpy(wet).to(device).unsqueeze(0)
        base = prior(wet_tensor, strength=strength)
        restored = model(wet_tensor, base, torch.from_numpy(controls).to(device).unsqueeze(0))[0].cpu().numpy()
        wet_rows.append(wet); restored_rows.append(restored); clean_rows.append(clean)
    return summarize(wet_rows, restored_rows, clean_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, default=ROOT / "cycles/rat1.json")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/corpus/asrnn-physical-effects")
    parser.add_argument("--prior", type=Path, default=ROOT / "runs/clean/model11/clean.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/rat/model1")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--clips", type=int, default=2)
    parser.add_argument("--frames", type=int, default=16_384)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--version", choices=(1, 2), type=int, default=1)
    parser.add_argument("--loss-version", choices=(1, 2, 3), type=int, default=1)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--rate", type=float, default=2.0e-4)
    parser.add_argument("--strength", type=float, default=0.875)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    cycle_bytes = args.cycle.read_bytes(); cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "rat" or cycle.get("status") != "planned":
        raise ValueError("rat1 cycle must be frozen and planned")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace RAT restoration run {args.output}")
    args.output.mkdir(parents=True)
    target = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(2); torch.manual_seed(20260902); np.random.seed(20260902)
    fit_all, calibration_all = _partition(rat_files(args.corpus, "train"))
    fit, fit_audit = eligible(fit_all, 0.03)
    calibration, calibration_audit = eligible(calibration_all, 0.03)
    evaluation, evaluation_audit = eligible(rat_files(args.corpus, "eval"), 0.03)
    prior = _prior(args.prior).to(target)
    model = (RATNet(args.channels) if args.version == 1 else RATNet2(args.channels, args.groups)).to(target)
    if args.init is not None:
        initial = torch.load(args.init, map_location="cpu", weights_only=True)
        if initial.get("architecture") != "normalized-control-conditioned-time-residual":
            raise ValueError("RAT initialization is not a v2 normalized temporal inverse")
        model.load_state_dict(initial["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.rate, weight_decay=1.0e-5)
    loader = DataLoader(Windows(fit, args.frames, args.clips), batch_size=args.batch, shuffle=True, num_workers=0)
    best_score, best_state, history = -float("inf"), None, []
    for epoch in range(args.epochs):
        model.train(); totals: dict[str, list[float]] = {}
        for wet, clean, controls in loader:
            wet = wet.flatten(0, 1).to(target); clean = clean.flatten(0, 1).to(target)
            controls = controls.to(target).repeat_interleave(args.clips, 0)
            with torch.no_grad():
                base = prior(wet, strength=args.strength)
            prediction = model(wet, base, controls)
            if args.loss_version == 1:
                value, parts = loss(prediction, clean)
            elif args.loss_version == 2:
                value, parts = relative_loss(prediction, clean, wet)
            else:
                value, parts = attack_loss(prediction, clean, wet)
            optimizer.zero_grad(set_to_none=True); value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            for name, item in {"loss": value, **parts}.items():
                totals.setdefault(name, []).append(float(item.detach().cpu()))
        calibration_report = audit(model, prior, calibration, target, args.strength, limit=24)
        score = calibration_report["median_transient_improvement"] + calibration_report["median_spectral_improvement"] + calibration_report["median_high_band_improvement"] + calibration_report["pass_fraction"]
        if score > best_score:
            best_score = score; best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        row = {"epoch": epoch + 1, "train": {name: float(np.mean(values)) for name, values in totals.items()}, "calibration": calibration_report, "score": score}
        history.append(row); (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps({"stage": "train-rat", **row}), flush=True)
    if best_state is None:
        raise RuntimeError("RAT training produced no checkpoint")
    model.load_state_dict(best_state)
    checkpoint = args.output / "model.pt"
    architecture = "control-conditioned-time-residual" if args.version == 1 else "normalized-control-conditioned-time-residual"
    torch.save({"schema": 1, "architecture": architecture, "sample_rate": RATE, "channels": args.channels, "groups": args.groups if args.version == 2 else args.channels, "dilations": list(DILATIONS), "state_dict": best_state, "prior_sha256": hashlib.sha256(args.prior.read_bytes()).hexdigest(), "prior_strength": args.strength, "device": "ProCo RAT", "controls": ["distortion", "tone", "volume"], "license": LICENSE}, checkpoint)
    calibration_report = audit(model, prior, calibration, target, args.strength)
    development_report = audit(model, prior, evaluation, target, args.strength)
    automatic = development_report["accepted"]
    result = {
        "schema": 1,
        "status": "audition-required" if automatic else "rejected-perceptual",
        "accepted": False,
        "automatic_accepted": automatic,
        "human": {"required": True, "approved": False},
        "model": {"checkpoint": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "parameters": sum(value.numel() for value in model.parameters()), "prior": str(args.prior), "initialization": str(args.init) if args.init is not None else None},
        "data": {"record": RECORD_URL, "license": LICENSE, "fit": len(fit), "calibration": len(calibration), "development": len(evaluation), "eligibility": {"fit": fit_audit, "calibration": calibration_audit, "development": evaluation_audit}, "source_read_only": True, "physical_audio_devices_used": False},
        "calibration": calibration_report,
        "development": development_report,
        "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest(),
        "quality": {"source_audio_modified": False, "physical_audio_devices_used": False, "automatic_normalization": False, "automatic_limiting": False, "automatic_dither": False, "lossy_reencoding": False},
        "limitations": ["oracle RAT controls during this upper-bound round", "official eval is development evidence, not an independent locked final set", "human phrase-level listening remains mandatory"],
    }
    (args.output / "valid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
