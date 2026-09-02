"""One fixed transient-focused NARX experiment with same-index supervision.

Each balanced fit batch has active-uniform512, Wet-event256 and frozen-core-
event256 groups.  Event selection is immutable fit metadata, never an input or
a prediction-maximum loss. No calibration sample updates weights or scales.
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
from .stable_effect import load_stable_effect
from .train_asrnn_phase7 import _evaluate, _rows
from .train_dfz_core_tcn import selection_key
from .train_dfz_long_tcn import CachedCalibrationRows, context_windows
from .train_dfz_narx import CachedNARXRenderer, fit_feature_rms
from .train_dfz_signed_tcn import active_window_ranges, uniform_active_start


STEPS, CALIBRATION_INTERVAL, LEARNING_RATE = 4_000, 250, 1e-4
CORNERS, WIDTH, UNIFORM_FRAMES, EVENT_FRAMES = 9, 32, 512, 256
SEED = 988


class CornerCycle:
    """Every 26 updates visits every fit recording at each of nine corners."""
    def __init__(self, rng, corners=CORNERS, size=26):
        self.rng, self.corners, self.size, self.steps = rng, corners, size, 0
        self.orders = [rng.permutation(size) for _ in range(corners)]
        self.counts = np.zeros((corners, size), dtype=np.int64)

    def next(self):
        if self.steps and self.steps % self.size == 0:
            self.orders = [self.rng.permutation(self.size) for _ in range(self.corners)]
        indices = [int(order[self.steps % self.size]) for order in self.orders]
        for corner, index in enumerate(indices):
            self.counts[corner, index] += 1
        self.steps += 1
        return indices

    def coverage(self):
        return {str(corner): {"distinct_fit_recordings": int((counts > 0).sum()),
                              "available_fit_recordings": self.size,
                              "minimum_visits": int(counts.min()), "maximum_visits": int(counts.max())}
                for corner, counts in enumerate(self.counts)}


def signed_events(signal):
    body = signal[1_024:]
    if body.ndim != 1 or not len(body):
        raise ValueError("events require one complete recording after burn-in")
    return tuple(dict.fromkeys((int(body.argmax()) + 1_024, int(body.argmin()) + 1_024)))


def event_window(event, frames, rng):
    start = max(1_024, min(frames - EVENT_FRAMES, event - int(rng.integers(96, 161))))
    offset = event - start
    if not 0 <= offset < EVENT_FRAMES:
        raise ValueError("event did not land inside its complete scored window")
    return start, offset


def event_weights(reference, offsets=None):
    if offsets is None:
        return torch.ones_like(reference)
    if offsets.shape != (reference.shape[0],):
        raise ValueError("one fixed fit-event offset required per window")
    positions = torch.arange(reference.shape[1], device=reference.device)[None]
    return 1 + 15 * ((positions - offsets[:, None]).abs() <= 8).to(reference.dtype)


def transient_loss(predicted, expected, file_energy, file_pre_energy, offsets=None):
    """Positive weighted quadratic error at the exact same time and sign."""
    if predicted.shape != expected.shape or expected.ndim != 2 or expected.shape[1] < 2:
        raise ValueError("aligned complete scored windows required")
    if file_energy.shape != (expected.shape[0],) or file_pre_energy.shape != file_energy.shape:
        raise ValueError("fit-full-file energy denominators required")
    weights = event_weights(expected, offsets)
    waveform = (weights * (predicted - expected).square()).sum(1) / weights.sum(1) / file_energy.detach().clamp_min(1e-5)
    dp = predicted[:, 1:] - .95 * predicted[:, :-1]
    dt = expected[:, 1:] - .95 * expected[:, :-1]
    pre = (weights[:, 1:] * (dp - dt).square()).sum(1) / weights[:, 1:].sum(1) / file_pre_energy.detach().clamp_min(1e-5)
    return waveform + .1 * pre


def render_windows(readout, bank, rows, starts, frames):
    dry = context_windows([row["dry"] for row in rows], starts, frames, max(LAGS)).to("mps")
    original = context_windows([row["original"] for row in rows], starts, frames, max(LAGS)).to("mps")
    expected = torch.stack([row["wet"][start:start + frames] for row, start in zip(rows, starts)]).to("mps")
    controls = torch.stack([row["controls"] for row in rows]).to("mps")
    energy = torch.stack([row["file_energy"] for row in rows]).to("mps")
    pre_energy = torch.stack([row["file_pre_energy"] for row in rows]).to("mps")
    features, _ = bank(dry, original)
    predicted = original[:, max(LAGS):] + readout.at_training_knots(features[:, max(LAGS):], controls)
    return predicted, expected, energy, pre_energy


def smoke():
    torch.manual_seed(SEED)
    bank, readout = CausalTapBank(), DynamicCornerReadout(width=WIDTH).to("mps")
    optimizer = torch.optim.AdamW(readout.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    measurements = []
    for iteration in range(3):
        torch.mps.synchronize()
        started = time.perf_counter()
        losses = []
        for frames, groups in ((UNIFORM_FRAMES, 1), (EVENT_FRAMES, 2)):
            count = CORNERS * groups
            controls = torch.tensor([(a, b) for a in (0., .5, 1.) for b in (0., .5, 1.)] * groups, device="mps")
            dry = torch.randn(count, frames + max(LAGS), device="mps") * .03
            original = torch.tanh(dry * 7) * .8
            expected = original[:, max(LAGS):] + dry[:, max(LAGS):] * .1
            features, _ = bank(dry, original)
            predicted = original[:, max(LAGS):] + readout.at_training_knots(features[:, max(LAGS):], controls)
            offsets = None if groups == 1 else torch.full((count,), 128, device="mps", dtype=torch.long)
            losses.append(transient_loss(predicted, expected, expected.square().mean(1), expected.square().mean(1), offsets).mean())
        loss = .5 * losses[0] + .5 * losses[1]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.mps.synchronize()
        measurements.append({"iteration": iteration, "seconds": time.perf_counter() - started,
                             "loss": float(loss.detach()), "driver_bytes": torch.mps.driver_allocated_memory()})
    return {"schema": 1, "stage": "synthetic-transient-narx-smoke", "measurements": measurements,
            "physical_audio_devices_used": False, "audio_loaded": False, "official_eval_opened": False}


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
    names = (Path(__file__).name, "dfz_dynamic_readout.py", "dfz_frozen_predictions.py", "train_dfz_narx.py",
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
    event_seen = {"wet_events": set(), "core_events": set()}
    event_expected = {kind: sum(len(row[kind]) for row in training) for kind in event_seen}
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

    def progress():
        return {"schema": 1, "experiment": "same-index-dual-event-transient-narx", "seed": SEED,
                "source_sha256": signature["source_sha256"], "source_code_sha256": code_hashes,
                "prediction_signature": signature, "cache": cache_info, "cache_ram_bytes": cache_bytes,
                "history": history, "updates": updates, "best_step": best_step, "requested_steps": STEPS,
                "completed_steps": updates[-1]["step"] if updates else 0, "learning_rate": LEARNING_RATE,
                "calibration_interval": CALIBRATION_INTERVAL, "fit_examples": len(fit), "calibration_examples": len(cal),
                "corner_coverage": cycle.coverage(), "event_coverage": {kind: {"seen": len(event_seen[kind]), "available": event_expected[kind]}
                                                                          for kind in event_seen},
                "architecture": {"lags": list(LAGS), "features": FEATURES, "hidden_widths": [WIDTH, WIDTH],
                                 "independent_corners": CORNERS, "residual_scale": .1, "rms": rms.tolist()},
                "window_definition": {"uniform": UNIFORM_FRAMES, "wet_event": EVENT_FRAMES, "core_event": EVENT_FRAMES,
                                      "true_left_history": max(LAGS), "event_window_offset_menu": [96, 160]},
                "event_definition": "fit-only signed argmax and argmin of full Wet and immutable core output after sample1024",
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
    for step in range(1, STEPS + 1):
        chosen = [rows[index] for rows, index in zip(corner_rows, cycle.next())]
        starts = [uniform_active_start(rng, row["uniform_ranges"]) for row in chosen]
        readout.train()
        uniform = transient_loss(*render_windows(readout, bank, chosen, starts, UNIFORM_FRAMES)).mean()
        event_starts, offsets = [], []
        for kind in ("wet_events", "core_events"):
            for row in chosen:
                event = row[kind][int(rng.integers(len(row[kind])))]
                start, offset = event_window(event, len(row["dry"]), rng)
                event_starts.append(start)
                offsets.append(offset)
                event_seen[kind].add((row["path"], event))
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
            progress_path.write_text(json.dumps(progress(), indent=2) + "\n")
            print(json.dumps({"calibration": history[-1], "best_step": best_step}), flush=True)
    if any(int((counts > 0).sum()) != 26 for counts in cycle.counts):
        raise ValueError("fit sampling failed to cover all 234 recordings")
    result = progress()
    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
    readout.cpu().load_state_dict(selected["readout_state_dict"])
    del training, calibration, predictions, corner_rows, chosen, cal_dataset
    torch.mps.empty_cache()
    unchanged()
    print(json.dumps({"stage": "actual-transient-narx-cpu54", "best_step": best_step}), flush=True)
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
