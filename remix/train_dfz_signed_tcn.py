"""Fixed same-time signed-MSE experiment around the immutable DFZ core.

The target supplies bounded fit-only frame weights, never a model feature.
There is no predicted maximum, extremum matching, gain fit or worst-file loss.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .core_conditioned_tcn import CoreConditionedResidual, CoreConditionedTCN
from .dfz_frozen_predictions import (CACHE, SOURCE, file_sha256, load_or_create_predictions,
                                     load_training_rows, partitions, prediction_signature)
from .stable_effect import load_stable_effect
from .train_asrnn_phase7 import _evaluate, _rows
from .train_dfz_core_tcn import CachedCoreTCNRenderer, selection_key
from .train_dfz_long_tcn import CachedCalibrationRows, context_windows


WIDTH, BLOCKS, BATCH_SIZE = 16, 12, 8
LEFT_CONTEXT, SCORED_FRAMES = 8_190, 4_096
STEPS, CALIBRATION_INTERVAL, LEARNING_RATE = 4_000, 250, 1e-4
ENERGY_FLOOR = 1e-5


def signed_frame_weights(expected, file_wet_peak):
    """Fit-only target weights; complete-file peaks bound every weight by 5."""
    return 1 + 4 * (expected.detach().abs() / file_wet_peak.detach().clamp_min(1e-5)[:, None]).pow(4)


def signed_loss_per_example(prediction, expected, file_wet_peak):
    if prediction.shape != expected.shape or expected.ndim != 2 or expected.shape[1] < 2:
        raise ValueError("signed loss needs matching [batch,scored-time] arrays with at least two frames")
    if file_wet_peak.shape != (expected.shape[0],):
        raise ValueError("one complete-file target peak is required per fit example")
    weights = signed_frame_weights(expected, file_wet_peak)
    energy = expected.square().mean(1).clamp_min(ENERGY_FLOOR)
    waveform = (weights * (prediction - expected).square()).mean(1) / energy
    predicted_pre = prediction[:, 1:] - .95 * prediction[:, :-1]
    expected_pre = expected[:, 1:] - .95 * expected[:, :-1]
    emphasized = (predicted_pre - expected_pre).square().mean(1) / expected_pre.square().mean(1).clamp_min(ENERGY_FLOOR)
    return waveform + .1 * emphasized


def active_window_ranges(dry, scored_frames=SCORED_FRAMES):
    """Intervals of starts whose scored window intersects a Dry activity sample.

Uniform selection over their combined integer length is genuinely uniform over
eligible starts, including disjoint activity regions, without big index caches.
"""
    last_start = len(dry) - scored_frames
    if last_start < 1_024:
        raise ValueError("recording is too short for a complete scored window")
    values = dry.numpy() if isinstance(dry, torch.Tensor) else np.asarray(dry)
    active = np.flatnonzero(np.abs(values) > max(1e-4, float(np.abs(values).max()) * .01))
    if not len(active):
        return [(1_024, last_start)]
    first_start = min(last_start, max(1_024, int(active[0]) - 256))
    boundaries = np.flatnonzero(np.diff(active) > scored_frames) + 1
    groups = np.split(active, boundaries)
    ranges = []
    for group in groups:
        first = max(first_start, int(group[0]) - scored_frames + 1)
        last = min(last_start, int(group[-1]))
        if first <= last:
            ranges.append((first, last))
    if not ranges:
        raise ValueError("activity has no legal scored window after the fixed burn-in")
    return ranges


def uniform_active_start(rng, ranges):
    offset = int(rng.integers(sum(last - first + 1 for first, last in ranges)))
    for first, last in ranges:
        count = last - first + 1
        if offset < count:
            return first + offset
        offset -= count
    raise AssertionError("uniform start fell outside its exact interval union")


def smoke():
    torch.manual_seed(987)
    tcn = CoreConditionedTCN(WIDTH, BLOCKS).to("mps")
    optimizer = torch.optim.AdamW(tcn.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    measurements = []
    for count in (1, BATCH_SIZE, BATCH_SIZE):
        dry = torch.randn(count, LEFT_CONTEXT + SCORED_FRAMES, device="mps") * .03
        original = torch.tanh(dry * 7) * .8
        controls = torch.full((count, 2), .5, device="mps")
        expected = original[:, LEFT_CONTEXT:] + dry[:, LEFT_CONTEXT:] * .1
        file_peak = expected.abs().amax(1)
        torch.mps.synchronize()
        started = time.perf_counter()
        predicted = original[:, LEFT_CONTEXT:] + tcn(dry, original, controls)[0][:, LEFT_CONTEXT:]
        loss = signed_loss_per_example(predicted, expected, file_peak).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.mps.synchronize()
        measurements.append({"batch": count, "seconds": time.perf_counter() - started, "loss": float(loss.detach()),
                             "allocated_bytes": torch.mps.current_allocated_memory(),
                             "driver_allocated_bytes": torch.mps.driver_allocated_memory()})
    return {"schema": 1, "stage": "synthetic-same-time-signed-loss-smoke", "measurements": measurements,
            "physical_audio_devices_used": False, "official_eval_opened": False, "audio_loaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("the fixed signed-loss experiment requires MPS")
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
        raise ValueError("a new output directory and the existing signed prediction-only cache are required")
    torch.manual_seed(987)
    rng = np.random.default_rng(987)
    fit, cal = partitions()
    signature = prediction_signature(fit, cal)
    code_hashes = {name: file_sha256(Path(__file__).parent / name) for name in
                   (Path(__file__).name, "core_conditioned_tcn.py", "dfz_frozen_predictions.py",
                    "train_dfz_core_tcn.py", "train_dfz_long_tcn.py", "train_asrnn_phase7.py")}
    base, payload = load_stable_effect(SOURCE)
    if payload.get("schema") != 7 or payload.get("device") != "dfz":
        raise ValueError("fixed best schema7 DFZ core required")
    base.eval().requires_grad_(False)
    started = time.perf_counter()
    args.output.mkdir(parents=True)
    predictions, cache_info = load_or_create_predictions(None, fit, cal, signature)
    if cache_info["created"]:
        raise ValueError("this experiment must reuse, never duplicate, the existing predictions")
    training = load_training_rows(fit, predictions[:len(fit)])
    calibration = load_training_rows(cal, predictions[len(fit):])
    for row in training:
        row["file_wet_peak"] = row["wet"][1_024:].abs().amax()
        row["active_window_ranges"] = active_window_ranges(row["dry"])
    cache_bytes = sum(row[key].numel() * row[key].element_size()
                      for rows in (training, calibration) for row in rows for key in ("dry", "wet", "original"))
    if cache_bytes > 550_000_000:
        raise ValueError("signed-loss data cache exceeds the fixed RAM bound")

    def assert_inputs_unchanged():
        if prediction_signature(fit, cal) != signature:
            raise ValueError("frozen source/data/extractor signature changed")
        if any(file_sha256(Path(__file__).parent / name) != digest for name, digest in code_hashes.items()):
            raise ValueError("signed-loss experiment code changed")
        if file_sha256(CACHE) != cache_info["sha256"]:
            raise ValueError("shared prediction-only cache changed")

    assert_inputs_unchanged()
    tcn = CoreConditionedTCN(WIDTH, BLOCKS).to("mps")
    if tcn.receptive_field != LEFT_CONTEXT + 1:
        raise ValueError("signed-loss context does not cover the exact receptive field")
    optimizer = torch.optim.AdamW(tcn.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    cal_dataset = CachedCalibrationRows(calibration)
    initial = _evaluate(CachedCoreTCNRenderer(tcn), cal_dataset, torch.device("mps"), BATCH_SIZE)
    history, updates = [{"step": 0, **initial}], []
    best_step, best_key = 0, selection_key(initial)
    selected_path, training_path = args.output / "best-residual.pt", args.output / "training.json"

    def save_best():
        torch.save({"experiment_schema": 1, "architecture": "core-conditioned-long-tcn-residual",
                    "source_sha256": signature["source_sha256"], "width": WIDTH, "blocks": BLOCKS,
                    "tcn_state_dict": {name: value.detach().cpu().clone() for name, value in tcn.state_dict().items()}}, selected_path)

    save_best()

    def progress():
        return {"schema": 1, "experiment": "same-time-signed-mse-core-conditioned-tcn", "source": str(SOURCE),
                "source_sha256": signature["source_sha256"], "source_code_sha256": code_hashes,
                "prediction_signature": signature, "cache": cache_info, "cache_ram_bytes": cache_bytes,
                "history": history, "updates": updates, "best_step": best_step, "requested_steps": STEPS,
                "completed_steps": updates[-1]["step"] if updates else 0,
                "width": WIDTH, "blocks": BLOCKS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                "calibration_interval": CALIBRATION_INTERVAL, "left_context_frames": LEFT_CONTEXT,
                "scored_frames": SCORED_FRAMES, "fit_examples": len(fit), "calibration_examples": len(cal),
                "loss_definition": "mean((1+4*(abs(wet)/max(fileWetPeak,1e-5))^4)*(yhat-wet)^2)/max(mean(wet^2),1e-5) + 0.1*mean((pre95(yhat)-pre95(wet))^2)/max(mean(pre95(wet)^2),1e-5); yhat=frozen-core+TCN-residual",
                "loss_reduction": "mean across fit batch; no predicted-max or extrema or worst-file term",
                "file_wet_peak_domain": "fit only, full recording after fixed 1024-frame burn-in",
                "preemphasis_domain": "4095 adjacent pairs inside the 4096 scored frames",
                "window_menu": "50% uniform over starts intersecting Dry activity; 50% Wet-peak-local fit supervision",
                "dry_activity_threshold": "max(1e-4,0.01*DryPeak); minimum start max(1024,onset-256)",
                "selection_rule": "complete-frozen-gate-first, peak-plus-global-esr-penalty, worst-control-group",
                "source_base_frozen": True, "wet_used_as_feature": False, "trainable_state_cached_between_steps": False,
                "zero_output_step0_retained_in_history": True,
                "selection_compute": "mps-float32-full-recordings", "final_verification_compute": "cpu-float32-full-recordings",
                "admitted": False, "official_eval_opened": False, "physical_audio_devices_used": False,
                "source_audio_modified": False, "automatic_normalization": False, "automatic_limiting": False,
                "automatic_gain": False, "elapsed_seconds": time.perf_counter() - started}

    training_path.write_text(json.dumps(progress(), indent=2) + "\n")
    print(json.dumps({"calibration": history[0], "cache": cache_info, "cache_ram_bytes": cache_bytes}), flush=True)
    for step in range(1, STEPS + 1):
        chosen = [training[int(index)] for index in rng.integers(0, len(training), BATCH_SIZE)]
        starts = []
        for row in chosen:
            if rng.random() < .5:
                start = uniform_active_start(rng, row["active_window_ranges"])
            else:
                start = row["peak_index"] - int(rng.integers(512, SCORED_FRAMES - 512))
                start = max(1_024, min(len(row["dry"]) - SCORED_FRAMES, start))
            starts.append(start)
        dry = context_windows([row["dry"] for row in chosen], starts).to("mps")
        original = context_windows([row["original"] for row in chosen], starts).to("mps")
        wet = torch.stack([row["wet"][start:start + SCORED_FRAMES] for row, start in zip(chosen, starts)]).to("mps")
        peak = torch.stack([row["file_wet_peak"] for row in chosen]).to("mps")
        controls = torch.stack([row["controls"] for row in chosen]).to("mps")
        step_started = time.perf_counter()
        tcn.train()
        predicted = original[:, LEFT_CONTEXT:] + tcn(dry, original, controls)[0][:, LEFT_CONTEXT:]
        loss = signed_loss_per_example(predicted, wet, peak).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite signed loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tcn.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            updates.append({"step": step, "loss": float(loss.detach()), "step_seconds": time.perf_counter() - step_started,
                            "elapsed_seconds": time.perf_counter() - started})
            training_path.write_text(json.dumps(progress(), indent=2) + "\n")
            print(json.dumps(updates[-1]), flush=True)
        if step % CALIBRATION_INTERVAL == 0:
            measured = _evaluate(CachedCoreTCNRenderer(tcn), cal_dataset, torch.device("mps"), BATCH_SIZE)
            history.append({"step": step, **measured})
            key = selection_key(measured)
            if key < best_key:
                best_step, best_key = step, key
                save_best()
            training_path.write_text(json.dumps(progress(), indent=2) + "\n")
            print(json.dumps({"calibration": history[-1], "best_step": best_step}), flush=True)
    result = progress()
    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
    tcn.cpu().load_state_dict(selected["tcn_state_dict"])
    del training, calibration, cal_dataset, predictions, chosen
    torch.mps.empty_cache()
    assert_inputs_unchanged()
    print(json.dumps({"stage": "actual-signed-loss-model-cpu-calibration", "examples": 54, "best_step": best_step}), flush=True)
    result["calibration"] = _evaluate(CoreConditionedResidual(base, tcn).eval(), _rows(cal, "dfz"), torch.device("cpu"), BATCH_SIZE)
    assert_inputs_unchanged()
    result["source_code_audio_and_prediction_sha256_reverified_after_cache_and_final_cpu_calibration"] = True
    result["passes_full_calibration_gate"] = result["calibration"]["passes_selection_gate"]
    result["selected_residual_sha256"] = file_sha256(selected_path)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["status"] = "passed-calibration-only-needs-schema-export" if result["passes_full_calibration_gate"] else "rejected-full-calibration"
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "calibration": result["calibration"], "best_step": best_step}), flush=True)


if __name__ == "__main__":
    main()
