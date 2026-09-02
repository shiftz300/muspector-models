"""Joint ASRNN and EGFx fine-tuning for the compact RAT clean restorer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from .asrnn_data import LICENSE as ASRNN_LICENSE
from .asrnn_data import rat_files
from .clean import Windows as AsrnnWindows
from .clean import accepted, eligible, evaluate, loss
from .egfx import Windows as EgfxWindows
from .egfx import audit as audit_egfx
from .egfx import evaluate_rows
from .net import SpectralNet
from .train_asrnn_rat_adapter import _partition


def objective(asrnn: dict, egfx: dict, gates: dict) -> float:
    """Maximize the weakest average-quality gate while penalizing tail regressions."""

    ratios = []
    for metrics, limits in ((asrnn, gates["asrnn"]), (egfx, gates["egfx"])):
        for name in ("aligned_esr_improvement", "aligned_mae_improvement", "sidr_improvement_db"):
            ratios.append(metrics[name] / limits[name])
        ratios.append(metrics["baseline_aligned_esr_p95"] / max(metrics["restored_aligned_esr_p95"], 1e-12))
        ratios.append(metrics["baseline_aligned_peak_error_p95"] / max(metrics["restored_aligned_peak_error_p95"], 1e-12))
    return float(min(ratios) + 0.01 * np.mean(ratios))


def decision(asrnn: dict, egfx: dict, gates: dict) -> tuple[bool, dict]:
    asrnn_passed, asrnn_failures = accepted(asrnn, gates["asrnn"])
    egfx_passed, egfx_failures = accepted(egfx, gates["egfx"])
    return asrnn_passed and egfx_passed, {
        "asrnn": asrnn_failures,
        "egfx": egfx_failures,
    }


def bucket(group: str) -> int:
    return int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10


def split_egfx(rows: list[dict], cycle: dict) -> dict[str, list[dict]]:
    declared = cycle.get("egfx_split")
    if declared is None:
        return {
            name: [row for row in rows if row["eligible"] and row["split"] == name]
            for name in ("fit", "calibration", "development")
        }
    selected = {
        name: [row for row in rows if row["eligible"] and bucket(row["group"]) in declared[name]]
        for name in ("fit", "calibration", "development")
    }
    groups = {name: {row["group"] for row in values} for name, values in selected.items()}
    if any(groups[left] & groups[right] for left in groups for right in groups if left < right):
        raise RuntimeError("declared EGFx groups overlap")
    return selected


def widen(source: SpectralNet, channels: int) -> SpectralNet:
    """Expand hidden width while preserving the source function exactly."""

    if channels < source.channels:
        raise ValueError("mixed model width cannot shrink")
    if channels == source.channels:
        return source
    target = SpectralNet(channels, source.n_fft, source.hop)
    old, new = source.channels, channels
    with torch.no_grad():
        for value in target.parameters():
            value.zero_()
        target.stem.weight[:old].copy_(source.stem.weight)
        target.stem.bias[:old].copy_(source.stem.bias)
        nn.init.normal_(target.stem.weight[old:], std=0.01)
        for left, right in zip(source.blocks, target.blocks):
            right.filter.weight[:old, :old].copy_(left.filter.weight[:old])
            right.filter.weight[new : new + old, :old].copy_(left.filter.weight[old:])
            right.filter.bias[:old].copy_(left.filter.bias[:old])
            right.filter.bias[new : new + old].copy_(left.filter.bias[old:])
            right.mix.weight[:old, :old].copy_(left.mix.weight)
            right.mix.bias[:old].copy_(left.mix.bias)
        target.head.weight[:, :old].copy_(source.head.weight)
        target.head.bias.copy_(source.head.bias)
    return target


def load_model(path: Path, channels: int | None = None) -> tuple[SpectralNet, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("architecture") != "complex-stft" or payload.get("sample_rate") != 48_000:
        raise ValueError("unsupported clean initialization")
    model = SpectralNet(payload["channels"], payload["n_fft"], payload["hop"])
    model.load_state_dict(payload["state_dict"])
    return widen(model, model.channels if channels is None else channels), payload


def train(args: argparse.Namespace) -> dict:
    cycle_bytes = args.cycle.read_bytes()
    cycle = json.loads(cycle_bytes)
    if cycle.get("id") != "clean1" or cycle.get("round") not in (9, 10, 11) or cycle.get("status") != "planned":
        raise ValueError("invalid mixed clean cycle")
    if hashlib.sha256(args.initial.read_bytes()).hexdigest() != cycle["architecture"]["initialization_sha256"]:
        raise ValueError("clean initialization hash differs")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace mixed run {args.output}")
    args.output.mkdir(parents=True)
    (args.output / "lock.json").write_text(json.dumps({
        "schema": 1,
        "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest(),
        "initialization_sha256": cycle["architecture"]["initialization_sha256"],
        "development_opened": False,
    }, indent=2) + "\n")

    fit_all, calibration_all = _partition(rat_files(args.asrnn, "train"))
    asrnn_fit, asrnn_fit_audit = eligible(fit_all, cycle["alignment"]["asrnn_minimum_wet_to_dry_rms_ratio"])
    asrnn_calibration, asrnn_calibration_audit = eligible(calibration_all, cycle["alignment"]["asrnn_minimum_wet_to_dry_rms_ratio"])
    egfx_rows, egfx_audit = audit_egfx(
        args.egfx,
        cycle["alignment"]["egfx_maximum_lag_frames"],
        cycle["alignment"]["egfx_minimum_absolute_correlation"],
    )
    egfx = split_egfx(egfx_rows, cycle)
    strength = float(cycle.get("inference", {}).get("strength", 1.0))
    loader = DataLoader(
        ConcatDataset((
            AsrnnWindows(asrnn_fit, args.frames, args.clips),
            EgfxWindows(args.egfx, egfx["fit"], args.frames, args.clips),
        )),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    model, source = load_model(args.initial, args.channels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.rate, weight_decay=1e-5)
    history, best, state = [], -float("inf"), None

    initial_asrnn = evaluate(model, asrnn_calibration, torch.device("cpu"), strength)["metrics"]
    initial_asrnn["coverage"] = asrnn_calibration_audit["coverage"]
    initial_egfx = evaluate_rows(model, args.egfx, egfx["calibration"], strength)
    for epoch in range(args.epochs):
        model.train()
        totals: dict[str, list[float]] = {}
        for wet, dry in loader:
            wet, dry = wet.flatten(0, 1), dry.flatten(0, 1)
            value, parts = loss(model, wet, dry)
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name, item in {"loss": value, **parts}.items():
                totals.setdefault(name, []).append(float(item.detach()))
        asrnn_report = evaluate(model, asrnn_calibration, torch.device("cpu"), strength)["metrics"]
        asrnn_report["coverage"] = asrnn_calibration_audit["coverage"]
        egfx_report = evaluate_rows(model, args.egfx, egfx["calibration"], strength)
        score = objective(asrnn_report, egfx_report, cycle["gates"])
        if score > best:
            best = score
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        row = {
            "epoch": epoch + 1,
            "score": score,
            "train": {name: float(np.mean(values)) for name, values in totals.items()},
            "asrnn": asrnn_report,
            "egfx": egfx_report,
        }
        history.append(row)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps({"stage": "mixed-train", **row}), flush=True)
    if state is None:
        raise RuntimeError("mixed training produced no checkpoint")

    model.load_state_dict(state)
    checkpoint = args.output / "clean.pt"
    torch.save({
        "schema": 1,
        "architecture": "complex-stft",
        "sample_rate": 48_000,
        "channels": model.channels,
        "n_fft": model.n_fft,
        "hop": model.hop,
        "device": "rat-mixed",
        "kind": "clean",
        "state_dict": state,
        "initialization": cycle["architecture"]["initialization_sha256"],
        "datasets": ["https://zenodo.org/records/20406285", "https://zenodo.org/records/7044411"],
        "licenses": [ASRNN_LICENSE, "CC-BY-4.0"],
    }, checkpoint)
    lock = json.loads((args.output / "lock.json").read_text())
    lock["development_opened"] = True
    (args.output / "lock.json").write_text(json.dumps(lock, indent=2) + "\n")

    asrnn_development, asrnn_development_audit = eligible(
        rat_files(args.asrnn, "eval"),
        cycle["alignment"]["asrnn_minimum_wet_to_dry_rms_ratio"],
    )
    asrnn_report = evaluate(model, asrnn_development, torch.device("cpu"), strength)["metrics"]
    asrnn_report["coverage"] = asrnn_development_audit["coverage"]
    egfx_report = evaluate_rows(model, args.egfx, egfx["development"], strength)
    passed, failures = decision(asrnn_report, egfx_report, cycle["gates"])
    parameters, artifact_bytes = sum(value.numel() for value in model.parameters()), checkpoint.stat().st_size
    if parameters > cycle["architecture"]["parameters_maximum"]:
        passed = False; failures["runtime_parameters"] = [parameters]
    if artifact_bytes > cycle["architecture"]["artifact_bytes_maximum"]:
        passed = False; failures["runtime_artifact"] = [artifact_bytes]
    report = {
        "schema": 1,
        "status": "accepted-quality-development" if passed else "rejected",
        "accepted": passed,
        "failures": failures,
        "model": {
            "checkpoint": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "parameters": parameters,
            "artifact_bytes": artifact_bytes,
            "initialization_sha256": cycle["architecture"]["initialization_sha256"],
            "strength": strength,
        },
        "data": {
            "asrnn": {"fit": len(asrnn_fit), "calibration": len(asrnn_calibration), "development": len(asrnn_development), "fit_audit": asrnn_fit_audit, "calibration_audit": asrnn_calibration_audit, "development_audit": asrnn_development_audit, "license": ASRNN_LICENSE},
            "egfx": {"fit": len(egfx["fit"]), "calibration": len(egfx["calibration"]), "development": len(egfx["development"]), "audit": egfx_audit, "license": "CC-BY-4.0"},
            "source_read_only": True,
            "physical_audio_devices_used": False,
        },
        "initial": {"asrnn_calibration": initial_asrnn, "egfx_calibration": initial_egfx},
        "development": {"asrnn": asrnn_report, "egfx": egfx_report},
        "gates": cycle["gates"],
        "quality": {"source_audio_modified": False, "preserve_frames": True, "preserve_channels": True, "preserve_sample_rate": True, "automatic_normalization": False, "automatic_limiting": False, "automatic_dither": False, "lossy_reencoding": False},
        "limitations": ["RAT family only", "ASRNN and EGFx development evidence", "output level unresolved", "independent device seal required"],
        "cycle_sha256": hashlib.sha256(cycle_bytes).hexdigest(),
        "source_payload_kind": source.get("kind"),
    }
    (args.output / "valid.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asrnn", type=Path, default=Path("data/corpus/asrnn-physical-effects"))
    parser.add_argument("--egfx", type=Path, default=Path("data/corpus/egfxset"))
    parser.add_argument("--cycle", type=Path, default=Path("cycles/clean9.json"))
    parser.add_argument("--initial", type=Path, default=Path("runs/clean/model4/clean.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/clean/model9"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--frames", type=int, default=16_384)
    parser.add_argument("--clips", type=int, default=1)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
