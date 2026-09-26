#!/usr/bin/env python3
"""Train and audit a Wet-only EGDB-PG Amp+cab inverse on MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader

from .egdb_pg_amp_data import CATEGORIES, EgdbPgAmpPairs, RATE
from .eg_ipt_amp_data import EgIptAmpPairs
from .egdb_pg_amp_model import WetConditionedAmpCabInverse
from .egdb_pg_amp_model2 import WetSpectralAmpCabInverse
from .egdb_pg_amp_model3 import WetHybridAmpCabInverse
from .egdb_pg_amp_model4 import WetToneConditionedAmpCabInverse
from .egdb_pg_amp_model5 import WetToneComplexAmpCabInverse
from .egdb_pg_amp_model6 import WetToneComplexDynamicsAmpCabInverse
from .egdb_pg_amp_model7 import WetToneComplexTemporalAmpCabInverse
from .egdb_pg_amp_model8 import (
    WetToneComplexUNetAmpCabInverse,
    WetToneComplexUNetJointAmpCabInverse,
)
from .egdb_pg_amp_model9 import WetToneComplexDemucsJointAmpCabInverse
from .egdb_pg_amp_model10 import (
    WetToneGrayBoxAmpCabInverse,
    WetTonePhaseGrayBoxAmpCabInverse,
)
from .egdb_pg_amp_model11 import WetTonePhaseGrayBoxDynamicsAmpCabInverse
from .egdb_pg_amp_model12 import WetTonePhaseGrayBoxMultibandDynamicsAmpCabInverse
from .open_riff_box_stage_model import (
    OpenRiffBoxStagewiseGrayBoxInverse,
    OpenRiffBoxStagewiseTransientGrayBoxInverse,
)
from .rusty_amp_stage_model import RustyAmpStagewiseGrayBoxInverse
from .quality2 import summarize


SEED = 20260905
TONE_MODELS = (
    WetToneConditionedAmpCabInverse,
    WetToneComplexAmpCabInverse,
    WetToneComplexDynamicsAmpCabInverse,
    WetToneComplexTemporalAmpCabInverse,
    WetToneComplexUNetAmpCabInverse,
    WetToneComplexDemucsJointAmpCabInverse,
    WetToneGrayBoxAmpCabInverse,
    WetTonePhaseGrayBoxAmpCabInverse,
    WetTonePhaseGrayBoxDynamicsAmpCabInverse,
    WetTonePhaseGrayBoxMultibandDynamicsAmpCabInverse,
    OpenRiffBoxStagewiseGrayBoxInverse,
    OpenRiffBoxStagewiseTransientGrayBoxInverse,
    RustyAmpStagewiseGrayBoxInverse,
)


def _collate(rows: list[dict]) -> dict:
    starts = {row["crop_start"] for row in rows}
    ends = {row["crop_end"] for row in rows}
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("one EGDB-PG batch must share crop geometry")
    batch = {
        "wet": torch.stack([row["wet"] for row in rows]),
        "clean": torch.stack([row["clean"] for row in rows]),
        "category": [row["category"] for row in rows],
        "profile_id": [row["profile_id"] for row in rows],
        "crop_start": starts.pop(),
        "crop_end": ends.pop(),
    }
    if all("tone_reference" in row for row in rows):
        batch["tone_reference"] = torch.stack([row["tone_reference"] for row in rows])
    return batch


def _perceptual_loss(
    restored: torch.Tensor, uncertainty: torch.Tensor, clean: torch.Tensor,
    *, dynamic_emphasis: bool = False, temporal_emphasis: bool = False,
    generator_emphasis: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Energy-robust objective that cannot improve by collapsing to silence."""
    rms = clean.square().mean(1).add(1.0e-8).sqrt().clamp_min(1.0e-3)
    waveform = ((restored - clean).abs().mean(1) / rms).clamp_max(10.0).mean()
    restored_diff = restored[:, 1:] - restored[:, :-1]
    clean_diff = clean[:, 1:] - clean[:, :-1]
    diff_scale = clean_diff.abs().mean(1).clamp_min(1.0e-4)
    transient = ((restored_diff - clean_diff).abs().mean(1) / diff_scale).clamp_max(10.0).mean()
    spectral_log = restored.new_zeros(())
    spectral_convergence = restored.new_zeros(())
    spectral_count = 0
    for fft_size in (256, 512, 1024):
        if restored.shape[1] < fft_size:
            continue
        window = torch.hann_window(fft_size, device=restored.device, dtype=restored.dtype)
        predicted = torch.stft(
            restored, fft_size, fft_size // 4, window=window,
            center=True, pad_mode="constant", return_complex=True,
        ).abs()
        target = torch.stft(
            clean, fft_size, fft_size // 4, window=window,
            center=True, pad_mode="constant", return_complex=True,
        ).abs()
        normalizer = rms[:, None, None]
        spectral_log = spectral_log + F.l1_loss(
            torch.log1p(predicted / normalizer), torch.log1p(target / normalizer)
        )
        convergence = torch.linalg.vector_norm(predicted - target, dim=(-2, -1)) / torch.linalg.vector_norm(
            target, dim=(-2, -1)
        ).clamp_min(1.0e-3)
        spectral_convergence = spectral_convergence + convergence.clamp_max(10.0).mean()
        spectral_count += 1
    spectral_log = spectral_log / spectral_count
    spectral_convergence = spectral_convergence / spectral_count
    clean_envelope = clean.unfold(1, 256, 128).square().mean(-1).add(1.0e-8).sqrt()
    restored_envelope = restored.unfold(1, 256, 128).square().mean(-1).add(1.0e-8).sqrt()
    envelope = F.l1_loss(
        torch.log(clean_envelope + 1.0e-4), torch.log(restored_envelope + 1.0e-4)
    )
    clean_crest_frames = clean.unfold(1, 1024, 256)
    restored_crest_frames = restored.unfold(1, 1024, 256)
    clean_crest_rms = clean_crest_frames.square().mean(-1).add(1.0e-8).sqrt()
    restored_crest_rms = restored_crest_frames.square().mean(-1).add(1.0e-8).sqrt()
    clean_crest = clean_crest_frames.abs().amax(-1) / clean_crest_rms.clamp_min(1.0e-4)
    restored_crest = restored_crest_frames.abs().amax(-1) / restored_crest_rms.clamp_min(1.0e-4)
    crest = F.l1_loss(torch.log(clean_crest + 1.0e-4), torch.log(restored_crest + 1.0e-4))
    smooth_power = 8.0
    clean_soft_peak = clean_crest_frames.abs().pow(smooth_power).mean(-1).add(1.0e-8).pow(1.0 / smooth_power)
    restored_soft_peak = restored_crest_frames.abs().pow(smooth_power).mean(-1).add(1.0e-8).pow(1.0 / smooth_power)
    clean_soft_crest = clean_soft_peak / clean_crest_rms.clamp_min(1.0e-4)
    restored_soft_crest = restored_soft_peak / restored_crest_rms.clamp_min(1.0e-4)
    soft_crest = F.l1_loss(
        torch.log(clean_soft_crest + 1.0e-4),
        torch.log(restored_soft_crest + 1.0e-4),
    )
    clean_attack_rms = clean.unfold(1, 240, 120).square().mean(-1).add(1.0e-8).sqrt()
    restored_attack_rms = restored.unfold(1, 240, 120).square().mean(-1).add(1.0e-8).sqrt()
    clean_attack = torch.relu(torch.diff(torch.log(clean_attack_rms + 1.0e-4), dim=1))
    restored_attack = torch.relu(torch.diff(torch.log(restored_attack_rms + 1.0e-4), dim=1))
    attack = F.l1_loss(clean_attack, restored_attack)
    centered_clean = clean - clean.mean(1, keepdim=True)
    centered_restored = restored - restored.mean(1, keepdim=True)
    correlation = 1.0 - (
        (centered_clean * centered_restored).sum(1)
        / (
            torch.linalg.vector_norm(centered_clean, dim=1)
            * torch.linalg.vector_norm(centered_restored, dim=1)
        ).clamp_min(1.0e-4)
    ).clamp(-1.0, 1.0).mean()
    uncertainty_target = (restored - clean).abs().detach()
    uncertainty_loss = ((uncertainty - uncertainty_target).abs().mean(1) / rms).clamp_max(10.0).mean()
    crest_weight = 2.0 if dynamic_emphasis else 0.75
    soft_crest_weight = 2.0 if dynamic_emphasis else 0.0
    transient_weight = 0.50 if temporal_emphasis else 0.10
    attack_weight = 2.0 if temporal_emphasis else 0.75
    waveform_weight = 5.0 if generator_emphasis else 0.30
    if generator_emphasis:
        transient_weight = 1.0
        attack_weight = 1.0
        crest_weight = 1.0
        soft_crest_weight = 1.0
    loss = (
        waveform_weight * waveform + transient_weight * transient + spectral_log
        + 0.50 * spectral_convergence + 0.25 * envelope
        + 0.25 * correlation + crest_weight * crest
        + soft_crest_weight * soft_crest + attack_weight * attack
        + 0.01 * uncertainty_loss
    )
    return loss, {
        "waveform": float(waveform.detach()),
        "transient": float(transient.detach()),
        "spectral_log": float(spectral_log.detach()),
        "spectral_convergence": float(spectral_convergence.detach()),
        "envelope": float(envelope.detach()),
        "crest": float(crest.detach()),
        "soft_crest": float(soft_crest.detach()),
        "attack": float(attack.detach()),
        "correlation": float(correlation.detach()),
        "uncertainty": float(uncertainty_loss.detach()),
    }


def _loss(model: torch.nn.Module, batch: dict, device: torch.device) -> tuple[torch.Tensor, dict]:
    wet = batch["wet"].to(device)
    clean = batch["clean"].to(device)
    reference = batch.get("tone_reference")
    restored, uncertainty, _ = model(
        wet, tone_reference=reference.to(device) if reference is not None else None
    ) if isinstance(model, TONE_MODELS) else model(wet)
    start, end = batch["crop_start"], batch["crop_end"]
    return _perceptual_loss(
        restored[:, start:end], uncertainty[:, start:end], clean[:, start:end],
        dynamic_emphasis=isinstance(
            model,
            (
                WetToneComplexDynamicsAmpCabInverse,
                WetToneComplexTemporalAmpCabInverse,
                WetToneComplexUNetAmpCabInverse,
                WetToneGrayBoxAmpCabInverse,
                WetTonePhaseGrayBoxAmpCabInverse,
                WetTonePhaseGrayBoxDynamicsAmpCabInverse,
                WetTonePhaseGrayBoxMultibandDynamicsAmpCabInverse,
                OpenRiffBoxStagewiseGrayBoxInverse,
                RustyAmpStagewiseGrayBoxInverse,
            ),
        ),
        temporal_emphasis=isinstance(
            model, (
                WetToneComplexTemporalAmpCabInverse,
                WetToneComplexUNetAmpCabInverse,
                OpenRiffBoxStagewiseTransientGrayBoxInverse,
            )
        ),
        generator_emphasis=isinstance(
            model,
            (
                WetToneComplexUNetJointAmpCabInverse,
                WetToneComplexDemucsJointAmpCabInverse,
            ),
        ),
    )


def _mean_loss(model, dataset, batch_size: int, device: torch.device) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total = 0.0
    examples = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            loss, _ = _loss(model, batch, device)
            count = len(batch["wet"])
            total += float(loss) * count
            examples += count
    return total / max(examples, 1)


def _quality(model, dataset) -> dict:
    aggregate = ([], [], [])
    categories = defaultdict(lambda: ([], [], []))
    profiles = defaultdict(lambda: ([], [], []))
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            row = dataset[index]
            if isinstance(model, TONE_MODELS):
                restored, _, _ = model(
                    row["wet"].unsqueeze(0),
                    tone_reference=row["tone_reference"].unsqueeze(0),
                )
            else:
                restored, _, _ = model(row["wet"].unsqueeze(0))
            start, end = row["crop_start"], row["crop_end"]
            values = (
                row["wet"][start:end].numpy(),
                restored[0, start:end].numpy().astype(np.float32),
                row["clean"][start:end].numpy(),
            )
            for collection in (aggregate, categories[row["category"]], profiles[row["profile_id"]]):
                for target, value in zip(collection, values, strict=True):
                    target.append(value)
    report = summarize("amp", *aggregate)
    report["categories"] = {
        name: summarize("amp", *values) for name, values in sorted(categories.items())
    }
    report["profiles"] = {
        name: summarize("amp", *values) for name, values in sorted(profiles.items())
    }
    report["all_categories_accepted"] = bool(
        set(report["categories"]) == set(CATEGORIES)
        and all(row["accepted"] for row in report["categories"].values())
    )
    report["all_unseen_profiles_accepted"] = bool(
        len(report["profiles"]) == 3
        and all(row["accepted"] for row in report["profiles"].values())
    )
    return report


def _runtime(model) -> dict:
    wet = torch.zeros(1, RATE)
    reference = torch.zeros(1, 3 * RATE)
    model.eval()
    with torch.inference_mode():
        if isinstance(model, TONE_MODELS):
            model(wet, tone_reference=reference)
        else:
            model(wet)
        started = time.perf_counter()
        for _ in range(3):
            if isinstance(model, TONE_MODELS):
                model(wet, tone_reference=reference)
            else:
                model(wet)
        elapsed = (time.perf_counter() - started) / 3
    return {
        "frames": RATE, "mean_seconds": elapsed, "realtime_factor": elapsed,
        "ordinary_cpu": True, "single_expert_only": True, "audio_callback": False,
    }


def train(args: argparse.Namespace) -> dict:
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace EGDB-PG run: {output}")
    reference_frames = args.tone_reference_frames if args.model_version in {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17} else 0
    fit = EgdbPgAmpPairs(
        workspace, "fit", args.train_samples, args.target_frames,
        args.context_frames, SEED + 1, reference_frames,
    )
    consumed_fresh = None
    eg_ipt_fit = None
    training_dataset = fit
    if args.include_consumed_fresh_v1_in_fit:
        consumed_fresh = EgdbPgAmpPairs(
            workspace, "fresh_validation", args.consumed_fresh_samples,
            args.target_frames, args.context_frames, SEED + 5,
            reference_frames,
        )
        training_dataset = ConcatDataset((fit, consumed_fresh))
    if args.include_eg_ipt_fit:
        eg_ipt_fit = EgIptAmpPairs(
            workspace, "fit", args.eg_ipt_samples, args.target_frames,
            args.context_frames, SEED + 7,
        )
        training_dataset = ConcatDataset((training_dataset, eg_ipt_fit))
    calibration = EgdbPgAmpPairs(
        workspace, "calibration", args.calibration_samples, args.target_frames,
        args.context_frames, SEED + 2, reference_frames,
    )
    development = None if args.defer_development else EgdbPgAmpPairs(
        workspace, "development", args.development_samples, args.target_frames,
        args.context_frames, SEED + 3, reference_frames,
    )
    device = torch.device(args.device)
    model_classes = {
        1: WetConditionedAmpCabInverse,
        2: WetSpectralAmpCabInverse,
        3: WetHybridAmpCabInverse,
        4: WetToneConditionedAmpCabInverse,
        5: WetToneComplexAmpCabInverse,
        6: WetToneComplexDynamicsAmpCabInverse,
        7: WetToneComplexTemporalAmpCabInverse,
        8: WetToneComplexUNetAmpCabInverse,
        9: WetToneComplexUNetJointAmpCabInverse,
        10: WetToneComplexDemucsJointAmpCabInverse,
        11: WetToneGrayBoxAmpCabInverse,
        12: WetTonePhaseGrayBoxAmpCabInverse,
        13: WetTonePhaseGrayBoxDynamicsAmpCabInverse,
        14: WetTonePhaseGrayBoxMultibandDynamicsAmpCabInverse,
        15: OpenRiffBoxStagewiseGrayBoxInverse,
        16: OpenRiffBoxStagewiseTransientGrayBoxInverse,
        17: RustyAmpStagewiseGrayBoxInverse,
    }
    model_class = model_classes[args.model_version]
    model = model_class(args.channels, args.depth, args.condition_size).to(device)
    tone_encoder_checkpoint = None
    spectral_warm_start = None
    stage_pretrain = None
    if args.model_version in {4, 5, 11, 12}:
        if args.tone_encoder_checkpoint is None:
            raise ValueError("tone-conditioned model requires --tone-encoder-checkpoint")
        model.load_tone_encoder(args.tone_encoder_checkpoint.resolve())
        tone_encoder_checkpoint = str(args.tone_encoder_checkpoint.resolve())
    if args.model_version in {6, 7, 8, 9, 10, 13, 14}:
        if args.warm_start is None:
            raise ValueError("complex refiner requires --warm-start from model v5")
        model.load_base(args.warm_start.resolve())
        spectral_warm_start = str(args.warm_start.resolve())
    if args.model_version == 3:
        if args.warm_start is None:
            raise ValueError("hybrid model requires --warm-start from spectral v2")
        payload = torch.load(args.warm_start.resolve(), map_location="cpu", weights_only=True)
        model.spectral.load_state_dict(payload["state_dict"])
        spectral_warm_start = str(args.warm_start.resolve())
    if args.model_version == 17:
        if args.stage_pretrain_checkpoint is None:
            raise ValueError("rusty-amp gray box requires --stage-pretrain-checkpoint")
        stage_checkpoint = args.stage_pretrain_checkpoint.resolve()
        payload = torch.load(stage_checkpoint, map_location="cpu", weights_only=True)
        architecture = payload.get("architecture", {})
        if architecture.get("schema") != 17:
            raise ValueError("rusty-amp stage pretrain must use architecture schema 17")
        model.load_state_dict(payload["state_dict"], strict=True)
        stage_pretrain = {
            "checkpoint": str(stage_checkpoint),
            "sha256": hashlib.sha256(stage_checkpoint.read_bytes()).hexdigest(),
        }
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate, weight_decay=1.0e-5
    )
    loader = DataLoader(
        training_dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED), collate_fn=_collate,
    )
    initial = _mean_loss(model, calibration, args.batch_size, device)
    best, best_epoch = initial, 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history = [{"epoch": 0, "train": None, "calibration_loss": initial}]
    print(json.dumps({"epoch": 0, "calibration_loss": initial}), flush=True)
    for epoch in range(1, args.epochs + 1):
        if args.model_version == 3:
            train_spectral = epoch > args.freeze_spectral_epochs
            for parameter in model.spectral.parameters():
                parameter.requires_grad_(train_spectral)
        totals = defaultdict(float)
        examples = 0
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = _loss(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
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
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(json.dumps({"epoch": epoch, "calibration_loss": calibration_loss}), flush=True)
    model = model.cpu()
    model.load_state_dict(best_state)
    development_report = None if development is None else _quality(model, development)
    runtime = _runtime(model)
    manifest = model.manifest()
    base_gates = {
        "calibration_improved": best_epoch > 0 and best <= initial * 0.90,
        "cpu_budget": runtime["realtime_factor"] <= 0.35,
        "wet_only_order_independent": all(manifest[name] is False for name in (
            "profile_id_input", "gain_category_input", "graph_order_input",
            "neighbor_effect_input", "recurrent_state_input", "clean_or_oracle_input",
        )),
        "locked_final_unopened": fit.audit.get("locked_final_downloaded") is False,
    }
    if development is None:
        gates = {**base_gates, "fresh_validation_deferred": True}
    else:
        gates = {
            **base_gates,
            "aggregate_quality": bool(development_report["accepted"]),
            "individual_pass_fraction": development_report["pass_fraction"] >= 0.80,
            "all_gain_categories": bool(development_report["all_categories_accepted"]),
            "all_unseen_profiles": bool(development_report["all_unseen_profiles_accepted"]),
            "profile_disjoint_development": not (
            {profile for _, profile in fit.profile_rows}
            & {profile for _, profile in development.profile_rows}
            ),
        }
    accepted = bool(not args.quick and development is not None and all(gates.values()))
    if accepted:
        status = "accepted-development"
    elif development is None and all(base_gates.values()):
        status = "trained-awaiting-fresh-validation"
    elif development is None:
        status = "diagnostic-calibration-not-promoted"
    else:
        status = "diagnostic-not-promoted"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    torch.save({
        "schema": 1, "sample_rate": RATE, "architecture": manifest,
        "state_dict": model.state_dict(),
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": 1,
        "status": status,
        "accepted": accepted,
        "quick": args.quick,
        "mechanism": "amp",
        "device_scope": (
            "EGDB-PG software Amp+cab presets plus fit-only EG-IPT physical Amp+cab+SM57; "
            "unseen EGDB profile development"
            if eg_ipt_fit is not None else
            "EGDB-PG software Amp+cab presets; unseen-profile development"
        ),
        "model": {**manifest, "checkpoint": str(checkpoint), "sha256": digest},
        "training": {
            "accelerator": device.type, "epochs": args.epochs,
            "model_version": args.model_version,
            "spectral_warm_start": spectral_warm_start,
            "tone_encoder_checkpoint": tone_encoder_checkpoint,
            "stage_pretrain": stage_pretrain,
            "freeze_spectral_epochs": args.freeze_spectral_epochs if args.model_version == 3 else None,
            "selected_epoch": best_epoch, "initial_calibration_loss": initial,
            "selected_calibration_loss": best,
            "fit_samples_per_epoch": len(training_dataset),
            "primary_fit_samples_per_epoch": args.train_samples,
            "consumed_fresh_v1_samples_per_epoch": (
                0 if consumed_fresh is None else len(consumed_fresh)
            ),
            "eg_ipt_fit_samples_per_epoch": 0 if eg_ipt_fit is None else len(eg_ipt_fit),
            "calibration_samples": args.calibration_samples,
            "development_samples": args.development_samples,
            "target_frames": args.target_frames, "context_frames_each_side": args.context_frames,
            "history": history,
        },
        "development": development_report,
        "runtime": runtime,
        "gates": gates,
        "data": {
            "contract": str((workspace / "remix/egdb_pg_subset_v1.json").resolve()),
            "audit": str((workspace / "data/corpus/egdb-pg-subset-v1/audio_audit.json").resolve()),
            "authorized_sources": fit.authorization["sources"] + (
                [] if eg_ipt_fit is None else eg_ipt_fit.authorization["sources"]
            ),
            "required_attribution": fit.authorization["required_attribution"] + (
                [] if eg_ipt_fit is None else eg_ipt_fit.authorization["required_attribution"]
            ),
            "fit_profile_count": len(fit.profile_rows),
            "consumed_fresh_v1_profile_count": (
                0 if consumed_fresh is None else len(consumed_fresh.profile_rows)
            ),
            "consumed_fresh_v1_used_for_fit": consumed_fresh is not None,
            "development_profile_count": 0 if development is None else len(development.profile_rows),
            "development_profiles_seen_in_fit": None if development is None else False,
            "development_audio_opened": development is not None,
            "eg_ipt_fit_only": None if eg_ipt_fit is None else {
                "samples_per_epoch": len(eg_ipt_fit),
                "source_pairs": 8717,
                "source_duration_hours": 4.726641510416666,
                "short_source_pairs_excluded_for_crop_geometry": eg_ipt_fit.short_pairs_excluded,
                "physical_chain": "EVH 5150 III plus Mesa 4x12 V30 plus close SM57",
                "used_for_product_gate": False,
            },
        },
        "quality": {
            "metric_schema": 2, "generated_audio_written": False,
            "demo_generated": False, "source_audio_modified": False,
            "locked_final_audio_opened": False,
        },
        "limitations": [
            "software Amp+cab evidence, not physical amplifier evidence",
            "three unseen development profiles are necessary but not sufficient for broad hardware generalization",
            "development metrics do not replace listening acceptance",
        ],
    }
    if eg_ipt_fit is not None:
        report["limitations"][0] = (
            "physical EG-IPT fit evidence covers one player/guitar/fixed Amp-cab-mic profile; "
            "development remains EGDB software profiles"
        )
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "accepted": accepted,
        "selected_epoch": best_epoch,
        "development_pass_fraction": (
            None if development_report is None
            else development_report["pass_fraction"]
        ),
        "gates": gates, "runtime": runtime, "sha256": digest,
    }, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--model-version", type=int, choices=tuple(range(1, 18)), default=2)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--tone-encoder-checkpoint", type=Path)
    parser.add_argument("--stage-pretrain-checkpoint", type=Path)
    parser.add_argument("--tone-reference-frames", type=int, default=3 * RATE)
    parser.add_argument("--defer-development", action="store_true")
    parser.add_argument("--include-consumed-fresh-v1-in-fit", action="store_true")
    parser.add_argument("--include-eg-ipt-fit", action="store_true")
    parser.add_argument("--eg-ipt-samples", type=int, default=864)
    parser.add_argument("--consumed-fresh-samples", type=int, default=288)
    parser.add_argument("--freeze-spectral-epochs", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=864)
    parser.add_argument("--calibration-samples", type=int, default=108)
    parser.add_argument("--development-samples", type=int, default=72)
    parser.add_argument("--target-frames", type=int, default=16384)
    parser.add_argument("--context-frames", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--condition-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.epochs = 2
        args.train_samples = 108
        args.calibration_samples = 36
        args.development_samples = 24
        args.target_frames = 8192
        args.context_frames = 2048
        if args.model_version == 10:
            args.channels = 48
            args.depth = 8
            args.condition_size = 64
        elif args.model_version in {13, 14, 15, 16, 17}:
            args.channels = 24
            args.depth = 8
            args.condition_size = 64
        elif args.model_version != 3:
            args.channels = 24
            args.depth = 6
            args.condition_size = 24
        args.freeze_spectral_epochs = 1
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 6)))
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable; run outside the sandbox")
    train(args)


if __name__ == "__main__":
    main()
