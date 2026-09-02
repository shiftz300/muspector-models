"""Bounded width16 long TCN conditioned on Dry and immutable core output.

4000 updates, fixed 250-step global calibration, unchanged signed waveform
loss, exact 8190-frame history for each input, no learned recurrent cache.
Only the small new residual is trained and saved; no runtime schema is added.
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
from .train_dfz_long_tcn import CachedCalibrationRows, context_windows, loss_per_example


WIDTH, BLOCKS, BATCH_SIZE = 16, 12, 8
LEFT_CONTEXT, SCORED_FRAMES = 8_190, 4_096
STEPS, CALIBRATION_INTERVAL, LEARNING_RATE = 4_000, 250, 3e-4


def selection_key(metrics):
    """A complete frozen gate pass always outranks an incomplete candidate."""
    return (not metrics["passes_selection_gate"],
            metrics["absolute_peak_error_p95"] + max(0., metrics["global_esr"] - .03),
            metrics["worst_attack_absolute_peak_error_p95"])


class CachedCoreTCNRenderer(torch.nn.Module):
    def __init__(self, tcn):
        super().__init__()
        self.tcn = tcn

    def forward(self, dry_and_original, controls, state=None):
        dry, original = dry_and_original[..., 0], dry_and_original[..., 1]
        residual, state = self.tcn(dry, original, controls, state)
        return original + residual, state


def smoke():
    torch.manual_seed(985)
    model = CoreConditionedTCN(WIDTH, BLOCKS).to("mps")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    measurements = []
    for count in (1, BATCH_SIZE, BATCH_SIZE):
        dry = torch.randn(count, LEFT_CONTEXT + SCORED_FRAMES, device="mps") * .03
        original = torch.tanh(dry * 7) * .8
        controls = torch.full((count, 2), .5, device="mps")
        expected = original[:, LEFT_CONTEXT:] + dry[:, LEFT_CONTEXT:] * .1
        torch.mps.synchronize()
        started = time.perf_counter()
        prediction = original[:, LEFT_CONTEXT:] + model(dry, original, controls)[0][:, LEFT_CONTEXT:]
        loss = loss_per_example(prediction, expected).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.mps.synchronize()
        measurements.append({"batch": count, "seconds": time.perf_counter() - started,
                             "loss": float(loss.detach()), "allocated_bytes": torch.mps.current_allocated_memory(),
                             "driver_allocated_bytes": torch.mps.driver_allocated_memory()})
    return {"schema": 1, "stage": "synthetic-core-conditioned-tcn-smoke", "width": WIDTH, "blocks": BLOCKS,
            "receptive_field_frames": model.receptive_field, "parameters": sum(value.numel() for value in model.parameters()),
            "measurements": measurements, "cache_ram_bytes": 288 * 144_000 * 3 * 4,
            "prediction_disk_tensor_bytes": 288 * 144_000 * 4, "physical_audio_devices_used": False,
            "official_eval_opened": False, "audio_loaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("this bounded experiment requires MPS")
    if args.smoke_only:
        if args.smoke_report and args.smoke_report.exists():
            raise ValueError("smoke report already exists")
        report = smoke()
        if args.smoke_report:
            args.smoke_report.parent.mkdir(parents=True, exist_ok=True)
            args.smoke_report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    if args.output is None or args.output.exists():
        raise ValueError("a new experiment output directory is required")
    torch.manual_seed(985)
    rng = np.random.default_rng(985)
    fit, cal = partitions()
    signature = prediction_signature(fit, cal)
    source_hash = signature["source_sha256"]
    code_hashes = {name: file_sha256(Path(__file__).parent / name) for name in
                   (Path(__file__).name, "core_conditioned_tcn.py", "dfz_frozen_predictions.py",
                    "train_dfz_long_tcn.py", "train_asrnn_phase7.py")}
    base, payload = load_stable_effect(SOURCE)
    if payload.get("schema") != 7 or payload.get("device") != "dfz":
        raise ValueError("the fixed best schema7 DFZ core is required")
    base.eval().requires_grad_(False).to("mps")
    started = time.perf_counter()
    args.output.mkdir(parents=True)
    predictions, cache_info = load_or_create_predictions(base, fit, cal, signature)
    training = load_training_rows(fit, predictions[:len(fit)])
    calibration = load_training_rows(cal, predictions[len(fit):])
    cache_bytes = sum(row[name].numel() * row[name].element_size()
                      for rows in (training, calibration) for row in rows for name in ("dry", "wet", "original"))
    if cache_bytes > 550_000_000:
        raise ValueError("Dry/Wet/p RAM cache exceeds its 550 MB budget")
    base.cpu()
    torch.mps.empty_cache()

    def assert_inputs_unchanged():
        if prediction_signature(fit, cal) != signature:
            raise ValueError("frozen source/audio/extractor signature changed")
        for name, digest in code_hashes.items():
            if file_sha256(Path(__file__).parent / name) != digest:
                raise ValueError(f"training source changed: {name}")
        if file_sha256(CACHE) != cache_info["sha256"]:
            raise ValueError("frozen prediction cache changed during training")

    assert_inputs_unchanged()
    tcn = CoreConditionedTCN(WIDTH, BLOCKS).to("mps")
    if tcn.receptive_field != LEFT_CONTEXT + 1:
        raise ValueError("training windows do not cover the exact receptive field")
    optimizer = torch.optim.AdamW(tcn.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    selected_path = args.output / "best-residual.pt"
    cal_dataset = CachedCalibrationRows(calibration)
    initial = _evaluate(CachedCoreTCNRenderer(tcn), cal_dataset, torch.device("mps"), BATCH_SIZE)
    history = [{"step": 0, **initial}]
    best_step, best_key = 0, selection_key(initial)

    def save_best():
        torch.save({"experiment_schema": 1, "architecture": "core-conditioned-long-tcn-residual",
                    "source_sha256": source_hash, "width": WIDTH, "blocks": BLOCKS,
                    "tcn_state_dict": {name: value.detach().cpu().clone() for name, value in tcn.state_dict().items()}}, selected_path)

    save_best()
    updates = []

    def progress():
        return {"schema": 1, "experiment": "frozen-schema7-output-conditioned-long-tcn", "source": str(SOURCE),
                "source_sha256": source_hash, "source_code_sha256": code_hashes, "prediction_signature": signature,
                "cache": cache_info, "cache_ram_bytes": cache_bytes, "history": history, "updates": updates,
                "best_step": best_step, "requested_steps": STEPS, "completed_steps": updates[-1]["step"] if updates else 0,
                "selection_rule": "complete-frozen-gate-first, peak-plus-global-esr-penalty, worst-control-group",
                "width": WIDTH, "blocks": BLOCKS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                "calibration_interval": CALIBRATION_INTERVAL, "left_context_frames": LEFT_CONTEXT,
                "scored_frames": SCORED_FRAMES, "inputs": ["Dry*21.4", "frozen-core-full-prefix-output"],
                "fit_examples": len(fit), "calibration_examples": len(cal), "source_base_frozen": True,
                "wet_used_as_feature": False, "trainable_state_cached_between_steps": False,
                "selection_compute": "mps-float32-full-recordings", "final_verification_compute": "cpu-float32-full-recordings",
                "zero_output_step0_retained_in_history": True, "admitted": False, "official_eval_opened": False,
                "physical_audio_devices_used": False, "source_audio_modified": False,
                "automatic_normalization": False, "automatic_limiting": False, "automatic_gain": False,
                "elapsed_seconds": time.perf_counter() - started}

    training_path = args.output / "training.json"
    training_path.write_text(json.dumps(progress(), indent=2) + "\n")
    print(json.dumps({"calibration": history[0], "cache_ram_bytes": cache_bytes, "cache": cache_info}), flush=True)
    for step in range(1, STEPS + 1):
        chosen = [training[int(index)] for index in rng.integers(0, len(training), BATCH_SIZE)]
        starts = []
        for row in chosen:
            last_start = len(row["dry"]) - SCORED_FRAMES
            if rng.random() < .7:
                start = row["peak_index"] - int(rng.integers(512, SCORED_FRAMES - 512))
            else:
                start = int(rng.integers(min(last_start, max(1_024, row["onset"] - 256)), last_start + 1))
            starts.append(max(1_024, min(last_start, start)))
        dry = context_windows([row["dry"] for row in chosen], starts).to("mps")
        original = context_windows([row["original"] for row in chosen], starts).to("mps")
        wet = torch.stack([row["wet"][start:start + SCORED_FRAMES] for row, start in zip(chosen, starts)]).to("mps")
        controls = torch.stack([row["controls"] for row in chosen]).to("mps")
        step_started = time.perf_counter()
        tcn.train()
        prediction = original[:, LEFT_CONTEXT:] + tcn(dry, original, controls)[0][:, LEFT_CONTEXT:]
        per_file = loss_per_example(prediction, wet)
        loss = .5 * per_file.mean() + .5 * per_file.topk(BATCH_SIZE // 3).values.mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss; original model remains untouched")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tcn.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            updates.append({"step": step, "loss": float(loss.detach()),
                            "step_seconds": time.perf_counter() - step_started,
                            "elapsed_seconds": time.perf_counter() - started})
            print(json.dumps(updates[-1]), flush=True)
            training_path.write_text(json.dumps(progress(), indent=2) + "\n")
        if step % CALIBRATION_INTERVAL == 0:
            measured = _evaluate(CachedCoreTCNRenderer(tcn), cal_dataset, torch.device("mps"), BATCH_SIZE)
            history.append({"step": step, **measured})
            key = selection_key(measured)
            if key < best_key:
                best_key, best_step = key, step
                save_best()
            training_path.write_text(json.dumps(progress(), indent=2) + "\n")
            print(json.dumps({"calibration": history[-1], "best_step": best_step}), flush=True)
    result = progress()
    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
    tcn.cpu().load_state_dict(selected["tcn_state_dict"])
    del training, calibration, cal_dataset, predictions, chosen
    torch.mps.empty_cache()
    assert_inputs_unchanged()
    model = CoreConditionedResidual(base.cpu(), tcn.cpu()).eval()
    print(json.dumps({"stage": "actual-core-conditioned-model-cpu-calibration", "examples": 54, "best_step": best_step}), flush=True)
    result["calibration"] = _evaluate(model, _rows(cal, "dfz"), torch.device("cpu"), BATCH_SIZE)
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
