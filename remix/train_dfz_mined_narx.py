"""Controlled current-prediction event-mining NARX experiment.

Model, loss, 0.5/0.25/0.25 group weights and non-core RNG draws are unchanged.
Only the core-event bucket uses fit-only current-model signed extrema refreshed
every 250 updates. Step4000 is fit audit only, never a training-event refresh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .dfz_dynamic_readout import CausalTapBank, DynamicCornerReadout, DynamicReadoutResidual, FEATURES, LAGS
from .dfz_frozen_predictions import (CACHE, SOURCE, file_sha256, load_or_create_predictions,
                                     load_training_rows, partitions, prediction_signature)
from .dfz_peak_mining import FitPeakMiner, module_sha256
from .stable_effect import load_stable_effect
from .train_asrnn_phase7 import _evaluate, _rows
from .train_dfz_core_tcn import selection_key
from .train_dfz_long_tcn import CachedCalibrationRows
from .train_dfz_narx import CachedNARXRenderer, fit_feature_rms
from .train_dfz_signed_tcn import active_window_ranges, uniform_active_start
from .train_dfz_transient_narx import (
    STEPS, CALIBRATION_INTERVAL, LEARNING_RATE, CORNERS, WIDTH, UNIFORM_FRAMES, EVENT_FRAMES, SEED,
    CornerCycle, event_window, render_windows, signed_events, smoke, transient_loss)


def expected_generation(training_step):
    if not 1 <= training_step <= STEPS:
        raise ValueError("training step must be in the fixed 1..4000 run")
    return ((training_step - 1) // CALIBRATION_INTERVAL) * CALIBRATION_INTERVAL


def choose_event(row, kind, rng, miner):
    # Preserve the old random draw even when the current-core choice is now
    # deterministic. Other windows and the per-corner shuffle stay controlled.
    old_choice = int(rng.integers(len(row[kind])))
    if kind == "wet_events":
        return row[kind][old_choice], "fixed-wet"
    if kind != "core_events":
        raise ValueError("unknown event bucket")
    return miner.next_event(row["path"])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("the fixed NARX experiment requires MPS")
    if args.smoke_only:
        if args.smoke_report and args.smoke_report.exists():
            raise ValueError("smoke report already exists")
        report = smoke()
        if args.smoke_report:
            args.smoke_report.parent.mkdir(parents=True, exist_ok=True)
            args.smoke_report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    if args.output is None or args.output.exists() or not CACHE.exists():
        raise ValueError("fresh output and existing prediction-only cache required")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    fit, cal = partitions()
    signature = prediction_signature(fit, cal)
    names = (Path(__file__).name, "dfz_peak_mining.py", "train_dfz_transient_narx.py",
             "dfz_dynamic_readout.py", "dfz_frozen_predictions.py", "train_dfz_narx.py",
             "train_dfz_signed_tcn.py", "train_dfz_long_tcn.py", "train_dfz_core_tcn.py", "train_asrnn_phase7.py")
    code_hashes = {name: file_sha256(Path(__file__).parent / name) for name in names}
    base, payload = load_stable_effect(SOURCE)
    if payload.get("schema") != 7 or payload.get("device") != "dfz":
        raise ValueError("fixed best schema7 DFZ source required")
    base.eval().requires_grad_(False)
    started = time.perf_counter()
    predictions, cache_info = load_or_create_predictions(None, fit, cal, signature)
    if cache_info["created"]:
        raise ValueError("only existing predictions may be reused")
    training = load_training_rows(fit, predictions[:len(fit)])
    calibration = load_training_rows(cal, predictions[len(fit):])
    corner_rows = [[] for _ in range(CORNERS)]
    for row in training:
        a, b = (row["controls"] * 2).long().tolist()
        corner_rows[a * 3 + b].append(row)
        body = row["wet"][1_024:]
        row["file_energy"] = body.square().mean()
        row["file_pre_energy"] = (body[1:] - .95 * body[:-1]).square().mean()
        row["uniform_ranges"] = active_window_ranges(row["dry"], UNIFORM_FRAMES)
        row["wet_events"], row["core_events"] = signed_events(row["wet"]), signed_events(row["original"])
    if any(len(rows) != 26 for rows in corner_rows):
        raise ValueError("exactly 26 fit recordings per corner required")
    rms = fit_feature_rms(training)
    cache_bytes = sum(row[key].numel() * row[key].element_size()
                      for rows in (training, calibration) for row in rows for key in ("dry", "wet", "original"))
    if cache_bytes > 550_000_000:
        raise ValueError("NARX RAM bound exceeded")

    def unchanged():
        if prediction_signature(fit, cal) != signature or file_sha256(CACHE) != cache_info["sha256"]:
            raise ValueError("source/data/shared prediction cache changed")
        if any(file_sha256(Path(__file__).parent / name) != digest for name, digest in code_hashes.items()):
            raise ValueError("transient experiment source changed")

    unchanged()
    args.output.mkdir(parents=True)
    bank, readout = CausalTapBank(), DynamicCornerReadout(rms, width=WIDTH).to("mps")
    optimizer = torch.optim.AdamW(readout.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    cycle = CornerCycle(rng)
    event_seen = {"wet_events": set(), "current_core_events": set()}
    wet_event_expected = sum(len(row["wet_events"]) for row in training)
    miner = FitPeakMiner(signature, cache_info["prediction_sha256"])
    snapshot = miner.refresh(training, readout, optimizer, 0, target=torch.device("mps"),
                             batch_size=CORNERS, block_frames=4096)
    fit_audits = [snapshot.metadata()]
    cal_dataset = CachedCalibrationRows(calibration)
    initial = _evaluate(CachedNARXRenderer(readout), cal_dataset, torch.device("mps"), CORNERS)
    history, updates = [{"step": 0, **initial}], []
    best_step, best_key = 0, selection_key(initial)
    selected_path, progress_path = args.output / "best-residual.pt", args.output / "training.json"

    def save_best():
        torch.save({"experiment_schema": 1, "architecture": "frozen-core-corner-narx64",
                    "source_sha256": signature["source_sha256"], "lags": list(LAGS), "width": WIDTH,
                    "readout_state_dict": {name: value.detach().cpu().clone() for name, value in readout.state_dict().items()}}, selected_path)

    save_best()

    def progress(*, detailed_fit=False):
        return {"schema": 1, "experiment": "same-index-current-fit-extrema-mined-narx", "seed": SEED,
                "source_sha256": signature["source_sha256"], "source_code_sha256": code_hashes,
                "prediction_signature": signature, "cache": cache_info, "cache_ram_bytes": cache_bytes,
                "history": history, "updates": updates, "best_step": best_step, "requested_steps": STEPS,
                "completed_steps": updates[-1]["step"] if updates else 0, "learning_rate": LEARNING_RATE,
                "calibration_interval": CALIBRATION_INTERVAL, "fit_examples": len(fit), "calibration_examples": len(cal),
                "corner_coverage": cycle.coverage(),
                "event_coverage": {"wet_events": {"seen": len(event_seen["wet_events"]), "available": wet_event_expected},
                                   "current_core_events": {"seen_generation_path_sign_keys": len(event_seen["current_core_events"])}},
                "fit_audits": fit_audits if detailed_fit else [
                    {name: value for name, value in audit.items() if name != "events"}
                    for audit in fit_audits],
                "fit_audit_events_included": detailed_fit,
                "current_event_update_steps": list(range(0, STEPS, CALIBRATION_INTERVAL)),
                "final_fit_audit_step": STEPS,
                "core_sign_selection": "positive/negative alternation per fit recording, retained across refresh",
                "non_core_rng_stream_preserved": True,
                "architecture": {"lags": list(LAGS), "features": FEATURES, "hidden_widths": [WIDTH, WIDTH],
                                 "independent_corners": CORNERS, "residual_scale": .1, "rms": rms.tolist()},
                "window_definition": {"uniform": UNIFORM_FRAMES, "wet_event": EVENT_FRAMES, "core_event": EVENT_FRAMES,
                                      "true_left_history": max(LAGS), "event_window_offset_menu": [96, 160]},
                "event_definition": "fixed fit Wet signed extrema; current whole-model fit extrema re-mined every250 with frozen parameter snapshot; sample1024 burn-in; never calibration",
                "loss_definition": ".5*uniform + .25*Wet-event + .25*core-event; each sum(w*(yhat-Wet)^2)/sum(w)/max(fileWetEnergy,1e-5) + .1*sum(w[1:]*(pre95(yhat)-pre95(Wet))^2)/sum(w[1:])/max(filePreWetEnergy,1e-5)",
                "weight_definition": "uniform w=1; event w=1+15*I(abs(t-fixed-event)<=8); never a predicted-max loss",
                "denominator_domain": "fit complete recording after1024; pre pairs inside that recording body",
                "selection_rule": "complete-frozen-gate-first, peak-plus-global-esr-penalty, worst-control-group",
                "source_base_frozen": True, "zero_step_retained": True, "wet_used_as_feature": False,
                "trainable_recurrent_state_cached": False, "calibration_used_for_gradient_or_scales": False,
                "physical_audio_devices_used": False, "official_eval_opened": False, "admitted": False,
                "source_audio_modified": False, "automatic_normalization": False, "automatic_gain": False,
                "automatic_limiting": False, "elapsed_seconds": time.perf_counter() - started}

    progress_path.write_text(json.dumps(progress(), indent=2) + "\n")
    print(json.dumps({"calibration": history[0], "cache_reused": True}), flush=True)
    print(json.dumps({"fit_audit_step": 0, "purpose": snapshot.purpose, "metrics": snapshot.metrics,
                      "readout_sha256": snapshot.readout_sha256}), flush=True)
    for step in range(1, STEPS + 1):
        if snapshot.completed_updates != expected_generation(step):
            raise ValueError("stale or future current-extrema snapshot used for training")
        chosen = [rows[index] for rows, index in zip(corner_rows, cycle.next())]
        starts = [uniform_active_start(rng, row["uniform_ranges"]) for row in chosen]
        readout.train()
        uniform = transient_loss(*render_windows(readout, bank, chosen, starts, UNIFORM_FRAMES)).mean()
        event_starts, offsets = [], []
        for kind in ("wet_events", "core_events"):
            for row in chosen:
                event, sign = choose_event(row, kind, rng, miner)
                start, offset = event_window(event, len(row["dry"]), rng)
                event_starts.append(start)
                offsets.append(offset)
                if kind == "wet_events":
                    event_seen[kind].add((row["path"], event))
                else:
                    event_seen["current_core_events"].add((snapshot.completed_updates, row["path"], sign))
        event_losses = transient_loss(*render_windows(readout, bank, [*chosen, *chosen], event_starts, EVENT_FRAMES),
                                      torch.tensor(offsets, device="mps"))
        loss = .5 * uniform + .25 * event_losses[:CORNERS].mean() + .25 * event_losses[CORNERS:].mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite transient loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(readout.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            updates.append({"step": step, "loss": float(loss.detach()), "elapsed_seconds": time.perf_counter() - started})
            progress_path.write_text(json.dumps(progress(), indent=2) + "\n")
            print(json.dumps(updates[-1]), flush=True)
        if step % CALIBRATION_INTERVAL == 0:
            measured = _evaluate(CachedNARXRenderer(readout), cal_dataset, torch.device("mps"), CORNERS)
            history.append({"step": step, **measured})
            if selection_key(measured) < best_key:
                best_step, best_key = step, selection_key(measured)
                save_best()
            if step < STEPS:
                snapshot = miner.refresh(training, readout, optimizer, step, target=torch.device("mps"),
                                         batch_size=CORNERS, block_frames=4096)
                audit = snapshot
            else:
                audit = miner.audit(training, readout, optimizer, step, target=torch.device("mps"),
                                    batch_size=CORNERS, block_frames=4096)
            fit_audits.append(audit.metadata())
            progress_path.write_text(json.dumps(progress(), indent=2) + "\n")
            print(json.dumps({"calibration": history[-1], "best_step": best_step}), flush=True)
            print(json.dumps({"fit_audit_step": step, "purpose": audit.purpose, "metrics": audit.metrics,
                              "readout_sha256": audit.readout_sha256}), flush=True)
    if any(int((counts > 0).sum()) != 26 for counts in cycle.counts):
        raise ValueError("fit sampling failed to cover all 234 recordings")
    result = progress(detailed_fit=True)
    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
    readout.cpu().load_state_dict(selected["readout_state_dict"])
    selected_snapshot = next(audit for audit in fit_audits if audit["completed_updates"] == best_step)
    if module_sha256(readout) != selected_snapshot["readout_sha256"]:
        raise ValueError("selected readout does not match its audited completed-update snapshot")
    result["selected_readout_snapshot_sha256"] = selected_snapshot["readout_sha256"]
    del training, calibration, predictions, corner_rows, chosen, cal_dataset
    torch.mps.empty_cache()
    unchanged()
    print(json.dumps({"stage": "actual-mined-narx-cpu54", "best_step": best_step}), flush=True)
    result["calibration"] = _evaluate(DynamicReadoutResidual(base.cpu(), readout.cpu()).eval(), _rows(cal, "dfz"), torch.device("cpu"), CORNERS)
    unchanged()
    result["source_code_audio_and_cache_reverified"] = True
    result["passes_full_calibration_gate"] = result["calibration"]["passes_selection_gate"]
    result["selected_residual_sha256"] = file_sha256(selected_path)
    result["status"] = "passed-calibration-only-needs-schema-export" if result["passes_full_calibration_gate"] else "rejected-full-calibration"
    result["elapsed_seconds"] = time.perf_counter() - started
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "calibration": result["calibration"], "best_step": best_step}), flush=True)


if __name__ == "__main__":
    main()
