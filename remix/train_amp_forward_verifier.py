#!/usr/bin/env python3
"""Train and anti-leakage audit a conditional Amp forward verifier on MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader

from .amp_forward_verifier import WetConditionedAmpForwardVerifier
from .eg_ipt_amp_data import EgIptAmpPairs
from .egdb_pg_amp_data import EgdbPgAmpPairs, RATE
from .train_egdb_pg_amp import SEED, _collate


def _normalized_esr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = (prediction - target).square().mean(1)
    energy = target.square().mean(1).clamp_min(1.0e-8)
    return error / energy


def _loss(model: torch.nn.Module, batch: dict, device: torch.device) -> tuple[torch.Tensor, dict]:
    clean = batch["clean"].to(device)
    wet = batch["wet"].to(device)
    reference = batch["tone_reference"].to(device)
    prediction = model(clean, wet, tone_reference=reference)
    start, end = batch["crop_start"], batch["crop_end"]
    prediction, wet = prediction[:, start:end], wet[:, start:end]
    rms = wet.square().mean(1).add(1.0e-8).sqrt().clamp_min(1.0e-3)
    waveform = ((prediction - wet).abs().mean(1) / rms).clamp_max(10.0).mean()
    prediction_diff, wet_diff = torch.diff(prediction, dim=1), torch.diff(wet, dim=1)
    diff_scale = wet_diff.abs().mean(1).clamp_min(1.0e-4)
    transient = ((prediction_diff - wet_diff).abs().mean(1) / diff_scale).clamp_max(10.0).mean()
    spectral = prediction.new_zeros(())
    for fft_size in (256, 1024):
        window = torch.hann_window(fft_size, device=device, dtype=prediction.dtype)
        predicted_stft = torch.stft(
            prediction, fft_size, fft_size // 4, window=window,
            center=True, pad_mode="constant", return_complex=True,
        ).abs()
        target_stft = torch.stft(
            wet, fft_size, fft_size // 4, window=window,
            center=True, pad_mode="constant", return_complex=True,
        ).abs()
        spectral = spectral + F.l1_loss(
            torch.log1p(predicted_stft / rms[:, None, None]),
            torch.log1p(target_stft / rms[:, None, None]),
        )
    spectral = spectral / 2.0
    esr = _normalized_esr(prediction, wet).clamp_max(10.0).mean()
    loss = waveform + 0.25 * transient + 0.50 * spectral + 0.25 * esr
    return loss, {
        "waveform": float(waveform.detach()),
        "transient": float(transient.detach()),
        "spectral": float(spectral.detach()),
        "esr": float(esr.detach()),
    }


def _mean_loss(model, dataset, batch_size: int, device: torch.device) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total, examples = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            loss, _ = _loss(model, batch, device)
            count = len(batch["wet"])
            total += float(loss) * count
            examples += count
    return total / examples


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "worst": float(array.max()),
    }


def _identifiability_audit(model, dataset, batch_size: int, device: torch.device) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    values = {name: [] for name in (
        "true_clean_replay", "direct_identity", "wet_as_clean_replay",
        "shuffled_clean_replay", "zero_candidate_replay",
    )}
    wins = {name: 0 for name in (
        "direct_identity", "wet_as_clean_replay",
        "shuffled_clean_replay", "zero_candidate_replay",
    )}
    examples = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            clean = batch["clean"].to(device)
            wet = batch["wet"].to(device)
            reference = batch["tone_reference"].to(device)
            count = clean.shape[0]
            if count > 1:
                shuffled = clean.roll(1, dims=0)
            else:
                shuffled = clean.flip(1)
            candidates = {
                "true_clean_replay": model(clean, wet, tone_reference=reference),
                "direct_identity": clean,
                "wet_as_clean_replay": model(wet, wet, tone_reference=reference),
                "shuffled_clean_replay": model(shuffled, wet, tone_reference=reference),
                "zero_candidate_replay": model(torch.zeros_like(clean), wet, tone_reference=reference),
            }
            start, end = batch["crop_start"], batch["crop_end"]
            target = wet[:, start:end]
            batch_esr = {
                name: _normalized_esr(value[:, start:end], target).cpu().numpy()
                for name, value in candidates.items()
            }
            truth = batch_esr["true_clean_replay"]
            for name, row in batch_esr.items():
                values[name].extend(float(item) for item in row)
            for name in wins:
                wins[name] += int(np.count_nonzero(truth < batch_esr[name]))
            examples += count
    summaries = {name: _summary(row) for name, row in values.items()}
    win_fractions = {name: count / examples for name, count in wins.items()}
    truth = summaries["true_clean_replay"]["mean"]
    gates = {
        "forward_beats_direct_identity": (
            truth <= 0.90 * summaries["direct_identity"]["mean"]
            and win_fractions["direct_identity"] >= 0.75
        ),
        "rejects_wet_as_clean": (
            truth <= 0.80 * summaries["wet_as_clean_replay"]["mean"]
            and win_fractions["wet_as_clean_replay"] >= 0.80
        ),
        "rejects_shuffled_clean": (
            truth <= 0.50 * summaries["shuffled_clean_replay"]["mean"]
            and win_fractions["shuffled_clean_replay"] >= 0.90
        ),
        "rejects_zero_candidate": (
            truth <= 0.50 * summaries["zero_candidate_replay"]["mean"]
            and win_fractions["zero_candidate_replay"] >= 0.90
        ),
    }
    return {
        "examples": examples,
        "normalized_esr": summaries,
        "true_clean_win_fraction": win_fractions,
        "gates": gates,
        "accepted": all(gates.values()),
    }


def train(args: argparse.Namespace) -> dict:
    workspace, output = args.workspace.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace Amp verifier run: {output}")
    reference_frames = 3 * RATE
    fit = EgdbPgAmpPairs(
        workspace, "fit", args.train_samples, args.target_frames,
        args.context_frames, SEED + 31, reference_frames,
    )
    calibration = EgdbPgAmpPairs(
        workspace, "calibration", args.calibration_samples, args.target_frames,
        args.context_frames, SEED + 32, reference_frames,
    )
    training_dataset = fit
    eg_ipt_fit = eg_ipt_calibration = None
    if args.include_eg_ipt_fit:
        eg_ipt_fit = EgIptAmpPairs(
            workspace, "fit", args.eg_ipt_samples, args.target_frames,
            args.context_frames, SEED + 33,
        )
        eg_ipt_calibration = EgIptAmpPairs(
            workspace, "internal_calibration", args.eg_ipt_calibration_samples,
            args.target_frames, args.context_frames, SEED + 34,
        )
        training_dataset = ConcatDataset((fit, eg_ipt_fit))
    device = torch.device(args.device)
    model = WetConditionedAmpForwardVerifier(
        args.channels, args.depth, args.condition_size
    ).to(device)
    model.load_tone_encoder(args.tone_encoder_checkpoint.resolve())
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate, weight_decay=1.0e-5,
    )
    loader = DataLoader(
        training_dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED + 30), collate_fn=_collate,
    )
    initial = _mean_loss(model, calibration, args.batch_size, device)
    best, best_epoch = initial, 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history = [{"epoch": 0, "train": None, "calibration_loss": initial}]
    print(json.dumps({"epoch": 0, "calibration_loss": initial}), flush=True)
    for epoch in range(1, args.epochs + 1):
        totals, examples = defaultdict(float), 0
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = _loss(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = len(batch["wet"])
            totals["loss"] += float(loss.detach()) * count
            for name, value in parts.items():
                totals[name] += value * count
            examples += count
        calibration_loss = _mean_loss(model, calibration, args.batch_size, device)
        history.append({
            "epoch": epoch,
            "train": {name: value / examples for name, value in sorted(totals.items())},
            "calibration_loss": calibration_loss,
        })
        if calibration_loss < best:
            best, best_epoch = calibration_loss, epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        print(json.dumps({"epoch": epoch, "calibration_loss": calibration_loss}), flush=True)
    model = model.cpu()
    model.load_state_dict(best_state)
    audit = _identifiability_audit(model, calibration, args.batch_size, torch.device("cpu"))
    eg_ipt_audit = None
    if eg_ipt_calibration is not None:
        eg_ipt_audit = _identifiability_audit(
            model, eg_ipt_calibration, args.batch_size, torch.device("cpu")
        )
    manifest = model.manifest()
    gates = {
        "full_prerequisite_schedule": (
            args.epochs >= 8
            and args.train_samples >= 2160
            and args.calibration_samples >= 360
        ),
        "calibration_improved": best_epoch > 0 and best <= 0.90 * initial,
        "egdb_identifiable": audit["accepted"],
        "eg_ipt_fit_only_identifiable": (
            True if eg_ipt_audit is None else eg_ipt_audit["accepted"]
        ),
        "no_sample_rate_wet_leakage_path": manifest["sample_rate_wet_path"] is False,
        "order_independent": (
            manifest["graph_order_input"] is False
            and manifest["neighbor_effect_input"] is False
        ),
        "locked_final_unopened": fit.audit.get("locked_final_downloaded") is False,
    }
    accepted = bool(not args.quick and all(gates.values()))
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    torch.save({
        "schema": 1, "sample_rate": RATE, "architecture": manifest,
        "state_dict": model.state_dict(),
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 1,
        "status": "accepted-forward-verifier-prerequisite" if accepted else "diagnostic-forward-verifier-not-admitted",
        "accepted": accepted,
        "usable_model": None,
        "mechanism": "amp",
        "purpose": "prerequisite verifier only; no inverse candidate is trained or promoted here",
        "model": {**manifest, "checkpoint": str(checkpoint), "sha256": digest},
        "training": {
            "accelerator": device.type,
            "epochs": args.epochs,
            "selected_epoch": best_epoch,
            "initial_calibration_loss": initial,
            "selected_calibration_loss": best,
            "fit_samples_per_epoch": len(training_dataset),
            "egdb_fit_samples_per_epoch": len(fit),
            "eg_ipt_fit_samples_per_epoch": 0 if eg_ipt_fit is None else len(eg_ipt_fit),
            "calibration_samples": len(calibration),
            "history": history,
        },
        "calibration_identifiability": audit,
        "eg_ipt_internal_calibration_identifiability": eg_ipt_audit,
        "gates": gates,
        "data": {
            "egdb_source_id": "egdb-pg-v2",
            "eg_ipt_source_id": None if eg_ipt_fit is None else "eg-ipt",
            "eg_ipt_product_gate_role": None if eg_ipt_fit is None else "fit-only",
            "development_audio_opened": False,
            "fresh_validation_audio_opened": False,
            "locked_final_audio_opened": False,
        },
        "limitations": [
            "a passing verifier is only permission to test cycle-assisted inversion, not restoration acceptance",
            "Wet is available only through a global frozen tone embedding; empirical anti-leakage gates remain mandatory",
            "development and all fresh-validation splits remain unopened by this prerequisite run",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "accepted": accepted,
        "selected_epoch": best_epoch, "gates": gates,
        "egdb_identifiability": audit,
        "eg_ipt_identifiability": eg_ipt_audit,
        "sha256": digest,
    }, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tone-encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=2160)
    parser.add_argument("--calibration-samples", type=int, default=360)
    parser.add_argument("--target-frames", type=int, default=16384)
    parser.add_argument("--context-frames", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--condition-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--include-eg-ipt-fit", action="store_true")
    parser.add_argument("--eg-ipt-samples", type=int, default=720)
    parser.add_argument("--eg-ipt-calibration-samples", type=int, default=180)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.epochs = 2
        args.train_samples = 180
        args.calibration_samples = 60
        args.target_frames = 8192
        args.context_frames = 2048
        args.channels = 24
        args.depth = 8
        args.condition_size = 48
        if args.include_eg_ipt_fit:
            args.eg_ipt_samples = 60
            args.eg_ipt_calibration_samples = 30
    train(args)


if __name__ == "__main__":
    main()
