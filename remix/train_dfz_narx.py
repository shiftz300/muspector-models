"""Fixed short-history, corner-independent signed-waveform DFZ experiment.

Use the existing full-prefix core cache, fit234/cal54 only, and never optimize
on calibration audio. 1500 steps, one fit recording per corner per batch, and
global gate-first checkpoint selection every 150 steps are fixed in advance.
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
from .train_dfz_signed_tcn import active_window_ranges, signed_loss_per_example, uniform_active_start


STEPS, CALIBRATION_INTERVAL = 1500, 150
BATCH_SIZE, SCORED_FRAMES, WIDTH, LEARNING_RATE = 9, 4096, 32, 3e-4


class CachedNARXRenderer(torch.nn.Module):
    def __init__(self, readout):
        super().__init__()
        self.bank, self.readout = CausalTapBank(), readout

    def forward(self, inputs, controls, state=None):
        if not torch.equal(controls * 2, (controls * 2).round()):
            raise ValueError("fast calibration renderer requires exact audited control knots")
        dry, original = inputs[..., 0], inputs[..., 1]
        features, state = self.bank(dry, original, state)
        return original + self.readout.at_training_knots(features, controls), state


def fit_feature_rms(rows):
    """Fixed uniform fit-only sample locations, never calibration scales."""
    total, count = torch.zeros(FEATURES, dtype=torch.float64), 0
    for row in rows:
        indices = torch.linspace(1024, len(row["dry"]) - 1, 512).long()
        columns = []
        for lag in LAGS:
            columns.extend((row["dry"][indices - lag] * 21.4, row["original"][indices - lag]))
        features = torch.stack(columns, -1).double()
        total += features.square().sum(0)
        count += len(indices)
    if not count:
        raise ValueError("fit rows are required for static RMS scaling")
    return (total / count).sqrt().clamp_min(1e-4).float()


def smoke():
    torch.manual_seed(988)
    bank = CausalTapBank()
    readout = DynamicCornerReadout(width=WIDTH).to("mps")
    optimizer = torch.optim.AdamW(readout.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    results = []
    controls = torch.tensor([(a, b) for a in (0., .5, 1.) for b in (0., .5, 1.)], device="mps")
    for _ in range(3):
        dry = torch.randn(BATCH_SIZE, SCORED_FRAMES + max(LAGS), device="mps") * .03
        original = torch.tanh(dry * 7) * .8
        wet = original[:, max(LAGS):] + dry[:, max(LAGS):] * .1
        torch.mps.synchronize()
        started = time.perf_counter()
        features, _ = bank(dry, original)
        predicted = original[:, max(LAGS):] + readout.at_training_knots(features[:, max(LAGS):], controls)
        loss = signed_loss_per_example(predicted, wet, wet.abs().amax(1)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.mps.synchronize()
        results.append({"seconds": time.perf_counter() - started, "loss": float(loss.detach()),
                        "driver_bytes": torch.mps.driver_allocated_memory()})
    return {"measurements": results, "audio_loaded": False, "physical_audio_devices_used": False,
            "trainable_parameters": sum(value.numel() for value in readout.parameters())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("bounded NARX training requires MPS compute, never an audio device")
    if args.smoke_only:
        print(json.dumps(smoke()), flush=True)
        return
    if args.output is None or args.output.exists() or not CACHE.exists():
        raise ValueError("new output directory and existing prediction-only cache required")
    torch.manual_seed(988)
    rng = np.random.default_rng(988)
    fit, cal = partitions()
    signature = prediction_signature(fit, cal)
    code_names = (Path(__file__).name, "dfz_dynamic_readout.py", "dfz_frozen_predictions.py", "train_dfz_signed_tcn.py",
                  "train_dfz_long_tcn.py", "train_dfz_core_tcn.py", "train_asrnn_phase7.py")
    code_hashes = {name: file_sha256(Path(__file__).parent / name) for name in code_names}
    base, payload = load_stable_effect(SOURCE)
    if payload.get("schema") != 7 or payload.get("device") != "dfz":
        raise ValueError("unchanged best schema7 DFZ source required")
    base.eval().requires_grad_(False)
    started = time.perf_counter()
    predictions, cache_info = load_or_create_predictions(None, fit, cal, signature)
    if cache_info["created"]:
        raise ValueError("NARX must reuse the prediction cache")
    training = load_training_rows(fit, predictions[:len(fit)])
    calibration = load_training_rows(cal, predictions[len(fit):])
    corner_rows = [[] for _ in range(9)]
    for row in training:
        ids = (row["controls"] * 2).long().tolist()
        corner_rows[ids[0] * 3 + ids[1]].append(row)
        row["file_wet_peak"] = row["wet"][1024:].abs().amax()
        row["active_window_ranges"] = active_window_ranges(row["dry"])
    if any(len(rows) != 26 for rows in corner_rows):
        raise ValueError("each independent corner must have exactly 26 fit recordings")
    rms = fit_feature_rms(training)
    cache_bytes = sum(row[name].numel() * row[name].element_size()
                      for rows in (training, calibration) for row in rows for name in ("dry", "wet", "original"))
    if cache_bytes > 550_000_000:
        raise ValueError("NARX data exceeds the fixed 550 MB RAM budget")

    def unchanged():
        if prediction_signature(fit, cal) != signature or file_sha256(CACHE) != cache_info["sha256"]:
            raise ValueError("frozen source, audio or prediction cache changed")
        if any(file_sha256(Path(__file__).parent / name) != digest for name, digest in code_hashes.items()):
            raise ValueError("NARX experiment code changed")

    unchanged()
    args.output.mkdir(parents=True)
    bank = CausalTapBank()
    readout = DynamicCornerReadout(rms, width=WIDTH).to("mps")
    optimizer = torch.optim.AdamW(readout.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    cal_dataset = CachedCalibrationRows(calibration)
    initial = _evaluate(CachedNARXRenderer(readout), cal_dataset, torch.device("mps"), BATCH_SIZE)
    history, updates = [{"step": 0, **initial}], []
    best_step, best_key = 0, selection_key(initial)
    selected_path = args.output / "best-residual.pt"

    def save_selected():
        torch.save({"experiment_schema": 1, "architecture": "frozen-core-corner-narx64",
                    "source_sha256": signature["source_sha256"], "lags": list(LAGS), "width": WIDTH,
                    "readout_state_dict": {key: value.detach().cpu().clone() for key, value in readout.state_dict().items()}}, selected_path)

    def progress():
        return {"schema": 1, "experiment": "frozen-core-independent-corner-narx64", "source_sha256": signature["source_sha256"],
                "source_code_sha256": code_hashes, "prediction_signature": signature, "cache": cache_info,
                "cache_ram_bytes": cache_bytes, "history": history, "updates": updates, "best_step": best_step,
                "requested_steps": STEPS, "completed_steps": updates[-1]["step"] if updates else 0,
                "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "scored_frames": SCORED_FRAMES,
                "calibration_interval": CALIBRATION_INTERVAL, "fit_examples": len(fit), "calibration_examples": len(cal),
                "architecture": {"lags": list(LAGS), "features": FEATURES, "hidden_widths": [WIDTH, WIDTH],
                                 "independent_corners": 9, "residual_scale": .1,
                                 "trainable_parameters": sum(value.numel() for value in readout.parameters()),
                                 "origin": "f(features)-f(zeros)", "rms": rms.tolist()},
                "sampling": "one random recording per corner per batch, 50%active-uniform/50%Wet-peak-local",
                "loss": "same-time signed weighted MSE/energy + 0.1 preemphasis MSE/energy; same fixed signed-TCN function",
                "selection_rule": "complete-frozen-gate-first, peak-plus-global-esr-penalty, worst-control-group",
                "source_base_frozen": True, "wet_used_as_feature": False, "automatic_normalization": False,
                "automatic_limiting": False, "automatic_gain": False, "source_audio_modified": False,
                "physical_audio_devices_used": False, "official_eval_opened": False, "admitted": False,
                "elapsed_seconds": time.perf_counter() - started}

    save_selected()
    print(json.dumps({"calibration": history[0], "cache_reused": not cache_info["created"]}), flush=True)
    for step in range(1, STEPS + 1):
        chosen = [rows[int(rng.integers(len(rows)))] for rows in corner_rows]
        starts = []
        for row in chosen:
            if rng.random() < .5:
                start = uniform_active_start(rng, row["active_window_ranges"])
            else:
                start = row["peak_index"] - int(rng.integers(512, SCORED_FRAMES - 512))
                start = max(1024, min(len(row["dry"]) - SCORED_FRAMES, start))
            starts.append(start)
        dry = context_windows([row["dry"] for row in chosen], starts, SCORED_FRAMES, max(LAGS)).to("mps")
        original = context_windows([row["original"] for row in chosen], starts, SCORED_FRAMES, max(LAGS)).to("mps")
        wet = torch.stack([row["wet"][start:start + SCORED_FRAMES] for row, start in zip(chosen, starts)]).to("mps")
        controls = torch.stack([row["controls"] for row in chosen]).to("mps")
        peaks = torch.stack([row["file_wet_peak"] for row in chosen]).to("mps")
        features, _ = bank(dry, original)
        readout.train()
        predicted = original[:, max(LAGS):] + readout.at_training_knots(features[:, max(LAGS):], controls)
        loss = signed_loss_per_example(predicted, wet, peaks).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite NARX loss; source remains untouched")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(readout.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            updates.append({"step": step, "loss": float(loss.detach()), "elapsed_seconds": time.perf_counter() - started})
            print(json.dumps(updates[-1]), flush=True)
        if step % CALIBRATION_INTERVAL == 0:
            measured = _evaluate(CachedNARXRenderer(readout), cal_dataset, torch.device("mps"), BATCH_SIZE)
            history.append({"step": step, **measured})
            if selection_key(measured) < best_key:
                best_step, best_key = step, selection_key(measured)
                save_selected()
            (args.output / "training.json").write_text(json.dumps(progress(), indent=2) + "\n")
            print(json.dumps({"calibration": history[-1], "best_step": best_step}), flush=True)
    result = progress()
    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
    readout.cpu().load_state_dict(selected["readout_state_dict"])
    del training, calibration, predictions, corner_rows, chosen, cal_dataset
    torch.mps.empty_cache()
    unchanged()
    print(json.dumps({"stage": "actual-complete-model-cpu54", "best_step": best_step}), flush=True)
    result["calibration"] = _evaluate(DynamicReadoutResidual(base.cpu(), readout.cpu()).eval(), _rows(cal, "dfz"), torch.device("cpu"), BATCH_SIZE)
    unchanged()
    result["source_code_audio_and_cache_reverified"] = True
    result["passes_full_calibration_gate"] = result["calibration"]["passes_selection_gate"]
    result["selected_residual_sha256"] = file_sha256(selected_path)
    result["status"] = "passed-calibration-only-needs-schema-export" if result["passes_full_calibration_gate"] else "rejected-full-calibration"
    result["elapsed_seconds"] = time.perf_counter() - started
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "calibration": result["calibration"]}), flush=True)


if __name__ == "__main__":
    main()
