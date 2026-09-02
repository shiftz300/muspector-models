"""Bounded long-memory TCN residual on immutable full-prefix schema-7 outputs.

Only the new residual is trained. Dry/Wet/frozen predictions occupy a small
CPU-RAM cache; no recurrent state from trainable weights is cached. Every
gradient window contains the entire 8190-sample finite receptive history.
This standalone experiment does not register or modify any runtime schema.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from .asrnn_effects import effect_controls, effect_files, read_effect_pair
from .fit_asrnn_effect_output import _partition
from .stable_effect import load_stable_effect
from .stable_tcn_residual import ZeroGatedTCN
from .train_asrnn_phase7 import _evaluate, _rows


SOURCE = Path("remix/runs/dfz-nonlinear-readout-phase9/candidate.pt")
TCN_WIDTH, TCN_BLOCKS = 8, 12
LEFT_CONTEXT, SCORED_FRAMES = 8_190, 4_096
STEPS, CALIBRATION_INTERVAL = 1_200, 300
LEARNING_RATE = 3e-4


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receptive_field(tcn):
    return 1 + 2 * sum(tcn.dilations)


def context_windows(signals, starts, scored_frames=SCORED_FRAMES, left_context=LEFT_CONTEXT):
    """Real left history, with zeros only before the recording starts."""
    if len(signals) != len(starts) or not signals:
        raise ValueError("matching nonempty windows required")
    windows = torch.zeros(len(signals), left_context + scored_frames, dtype=torch.float32)
    for index, (signal, start) in enumerate(zip(signals, starts)):
        if signal.ndim != 1 or start < 0 or start + scored_frames > len(signal):
            raise ValueError("window lies outside a complete recording")
        first = max(0, start - left_context)
        padding = max(0, left_context - start)
        windows[index, padding:].copy_(signal[first:start + scored_frames])
    return windows


class GenericLongTCNResidual(torch.nn.Module):
    """Independent wrapper compatible with schema-7's nested frozen base."""
    def __init__(self, base, tcn):
        super().__init__()
        self.base, self.tcn = base.eval().requires_grad_(False), tcn
        self.control_count = base.control_count
        base_widths = getattr(base, "export_state_widths", [layer.hidden_size for layer in base.modules()
                                                           if isinstance(layer, torch.nn.LSTM)])
        self.base_states = len(base_widths)
        self.export_state_widths = [*base_widths, *[tcn.width * d for d in tcn.dilations]]

    def forward(self, dry, controls, state=None):
        if state is not None and len(state) != len(self.export_state_widths):
            raise ValueError("wrong generic long-TCN state count")
        original, base_state = self.base(dry, controls, None if state is None else state[:self.base_states])
        residual, tcn_state = self.tcn(dry, controls, None if state is None else state[self.base_states:])
        return original + residual, (*base_state, *tcn_state)


@torch.inference_mode()
def cache_full_prefix(base, paths, target, label, batch_size=4, block_frames=8_192):
    rows = []
    for offset in range(0, len(paths), batch_size):
        selected = paths[offset:offset + batch_size]
        pairs = [read_effect_pair(path, "dfz") for path in selected]
        if any(len(pair[0]) != 144_000 for pair in pairs):
            raise ValueError("the bounded DFZ full-recording geometry differs")
        dry_cpu = torch.from_numpy(np.stack([pair[0] for pair in pairs]))
        controls = torch.from_numpy(np.stack([pair[2] for pair in pairs])).to(target)
        prediction = torch.empty_like(dry_cpu)
        state = None
        for first in range(0, dry_cpu.shape[1], block_frames):
            dry = dry_cpu[:, first:first + block_frames].to(target)
            hidden, state = base.base.encode(dry, controls, state)
            value = base.base.output_layer(hidden).squeeze(-1) + base.readout.at_training_knots(hidden, controls)
            prediction[:, first:first + value.shape[1]].copy_(value.cpu())
        for local, (path, (dry, wet, control)) in enumerate(zip(selected, pairs)):
            active = np.flatnonzero(np.abs(dry) > max(1e-4, float(np.abs(dry).max()) * .01))
            onset = int(active[0]) if len(active) else 1_024
            rows.append({"dry": torch.from_numpy(dry), "wet": torch.from_numpy(wet),
                         "original": prediction[local].clone(), "controls": torch.from_numpy(control),
                         "attack": int(path.stem.split(",")[0]), "path": str(path), "onset": onset,
                         "peak_index": int(np.abs(wet[1_024:]).argmax()) + 1_024})
        if offset % 32 == 0 or offset + len(selected) == len(paths):
            print(json.dumps({"stage": "immutable-full-prefix-cache", "partition": label,
                              "completed": len(rows), "total": len(paths)}), flush=True)
    return rows


class CachedCalibrationRows(Dataset):
    """Materialize only the current batch's Dry/base pair, never a second cache."""
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {"dry": torch.stack((row["dry"], row["original"]), -1), "wet": row["wet"],
                "controls": row["controls"], "attack": row["attack"]}


class CachedTCNRenderer(torch.nn.Module):
    def __init__(self, tcn):
        super().__init__()
        self.tcn = tcn

    def forward(self, dry_and_original, controls, state=None):
        residual, state = self.tcn(dry_and_original[..., 0], controls, state)
        return dry_and_original[..., 1] + residual, state


def loss_per_example(prediction, expected):
    error = prediction - expected
    waveform = error.square().mean(1) / expected.square().mean(1).clamp_min(1e-6)
    dp, dt = prediction[:, 1:] - .95 * prediction[:, :-1], expected[:, 1:] - .95 * expected[:, :-1]
    emphasized = (dp - dt).square().mean(1) / dt.square().mean(1).clamp_min(1e-6)
    envelope = (functional.max_pool1d(prediction.abs()[:, None], 128, 64)
                - functional.max_pool1d(expected.abs()[:, None], 128, 64)).abs().mean((1, 2))
    signed_extrema = .5 * ((prediction.amax(1) - expected.amax(1)).abs()
                          + (prediction.amin(1) - expected.amin(1)).abs())
    peak = (prediction.abs().amax(1) - expected.abs().amax(1)).abs()
    return waveform + .25 * emphasized + 4 * envelope + 12 * signed_extrema + 4 * peak


def smoke(batch_size=6):
    """Two tiny synthetic updates only; no corpus loading or source inference."""
    torch.set_num_threads(2)
    torch.manual_seed(983)
    model = ZeroGatedTCN(TCN_WIDTH, TCN_BLOCKS).to("mps")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    before = torch.mps.driver_allocated_memory()
    durations = []
    for count in (1, batch_size):
        dry = torch.randn(count, LEFT_CONTEXT + SCORED_FRAMES, device="mps") * .03
        controls = torch.full((count, 2), .5, device="mps")
        target = dry[:, LEFT_CONTEXT:] * .8
        torch.mps.synchronize()
        started = time.perf_counter()
        prediction = model(dry, controls)[0][:, LEFT_CONTEXT:]
        loss = loss_per_example(prediction, target).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.mps.synchronize()
        durations.append({"batch": count, "update_seconds_including_first_shape_compile": time.perf_counter() - started,
                          "loss": float(loss.detach()), "allocated_bytes": torch.mps.current_allocated_memory(),
                          "driver_allocated_bytes": torch.mps.driver_allocated_memory()})
    return {"schema": 1, "stage": "synthetic-resource-smoke", "width": TCN_WIDTH, "blocks": TCN_BLOCKS,
            "receptive_field_frames": receptive_field(model), "receptive_field_ms": receptive_field(model) / 48,
            "left_context_frames": LEFT_CONTEXT, "scored_frames": SCORED_FRAMES,
            "parameters": sum(value.numel() for value in model.parameters()), "measurements": durations,
            "driver_allocated_before_inputs": before, "estimated_fit_plus_calibration_cpu_cache_bytes": 288 * 144_000 * 3 * 4,
            "other_process_memory_not_measured": True, "physical_audio_devices_used": False,
            "official_eval_opened": False, "source_audio_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--batch-size", type=int, choices=(6, 8), default=6)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("the bounded training experiment requires MPS")
    if args.smoke_only:
        if args.smoke_report and args.smoke_report.exists():
            raise ValueError("smoke evidence already exists")
        result = smoke(args.batch_size)
        if args.smoke_report:
            args.smoke_report.parent.mkdir(parents=True, exist_ok=True)
            args.smoke_report.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return
    if args.output is None or args.output.exists():
        raise ValueError("a fresh experiment output directory is required")
    torch.manual_seed(983)
    rng = np.random.default_rng(983)
    target = torch.device("mps")
    source_hash = sha256(SOURCE)
    source_code_hashes = {name: sha256(Path(__file__).parent / name) for name in
                          (Path(__file__).name, "stable_tcn_residual.py", "stable_effect.py",
                           "stable_nonlinear_readout.py", "train_asrnn_phase7.py",
                           "asrnn_effects.py", "fit_asrnn_effect_output.py")}
    base, payload = load_stable_effect(SOURCE)
    if payload.get("schema") != 7 or payload.get("device") != "dfz":
        raise ValueError("the fixed best schema-7 DFZ source is required")
    base.eval().requires_grad_(False).to(target)
    fit, cal = _partition(effect_files(Path("data/corpus/asrnn-physical-effects"), "dfz", "train"))
    if len(fit) != 234 or len(cal) != 54:
        raise ValueError("frozen fit/calibration partition differs")
    control_groups = {}
    expected_corners = {(first, second) for first in (0., .5, 1.) for second in (0., .5, 1.)}
    for name, paths, per_corner in (("fit", fit, 26), ("calibration", cal, 6)):
        groups = Counter(tuple(float(value) for value in effect_controls(path, "dfz")) for path in paths)
        if set(groups) != expected_corners or set(groups.values()) != {per_corner}:
            raise ValueError(f"{name} must contain the exact 3x3 control grid with {per_corner} files per corner")
        control_groups[name] = {f"{corner[0]:g},{corner[1]:g}": count for corner, count in sorted(groups.items())}
    provenance = {name: [{"path": str(path), "sha256": sha256(path)} for path in paths]
                  for name, paths in (("fit", fit), ("calibration", cal))}

    def assert_inputs_unchanged():
        if sha256(SOURCE) != source_hash:
            raise ValueError("frozen schema-7 source changed during the experiment")
        for name, expected in source_code_hashes.items():
            if sha256(Path(__file__).parent / name) != expected:
                raise ValueError(f"experiment source code changed: {name}")
        for rows in provenance.values():
            for row in rows:
                if sha256(row["path"]) != row["sha256"]:
                    raise ValueError(f"fit/calibration audio changed: {row['path']}")

    args.output.mkdir(parents=True)
    started = time.perf_counter()
    training = cache_full_prefix(base, fit, target, "fit")
    calibration = cache_full_prefix(base, cal, target, "calibration")
    assert_inputs_unchanged()
    base.cpu()
    torch.mps.empty_cache()
    cache_bytes = sum(row[key].numel() * row[key].element_size()
                      for rows in (training, calibration) for row in rows for key in ("dry", "wet", "original"))
    if cache_bytes > 550_000_000:
        raise ValueError("full-prefix cache exceeds the bounded RAM budget")
    tcn = ZeroGatedTCN(TCN_WIDTH, TCN_BLOCKS).to(target)
    if receptive_field(tcn) != LEFT_CONTEXT + 1:
        raise ValueError("training context does not cover the exact receptive field")
    optimizer = torch.optim.AdamW(tcn.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    selected_path = args.output / "best-residual.pt"
    cal_dataset = CachedCalibrationRows(calibration)
    initial = _evaluate(CachedTCNRenderer(tcn), cal_dataset, target, 6)
    history = [{"step": 0, **initial}]
    best_step, best_score = 0, initial["absolute_peak_error_p95"] + max(0., initial["global_esr"] - .03)

    def save_best():
        torch.save({"experiment_schema": 1, "source_sha256": source_hash, "width": TCN_WIDTH, "blocks": TCN_BLOCKS,
                    "tcn_state_dict": {key: value.detach().cpu().clone() for key, value in tcn.state_dict().items()}}, selected_path)

    save_best()
    updates = []

    def progress():
        return {"schema": 1, "experiment": "frozen-schema7-long-tcn-residual", "source": str(SOURCE), "source_sha256": source_hash,
                "source_code_sha256": source_code_hashes,
                "history": history, "updates": updates, "best_step": best_step, "requested_steps": STEPS,
                "completed_steps": updates[-1]["step"] if updates else 0, "learning_rate": LEARNING_RATE,
                "width": TCN_WIDTH, "blocks": TCN_BLOCKS, "receptive_field_frames": LEFT_CONTEXT + 1,
                "left_context_frames": LEFT_CONTEXT, "scored_frames": SCORED_FRAMES, "batch_size": args.batch_size,
                "fit_examples": len(fit), "calibration_examples": len(cal), "cache_bytes": cache_bytes,
                "control_groups": control_groups, "selection_compute": "mps-float32-full-recordings",
                "final_verification_compute": "cpu-float32-full-recordings",
                "cache_storage": "RAM only: full-causal-prefix immutable Dry/Wet/base prediction",
                "trainable_state_cached_between_steps": False, "source_base_frozen": True,
                "zero_output_step0_retained_in_history": True, "provenance": provenance,
                "admitted": False, "official_eval_opened": False, "physical_audio_devices_used": False,
                "source_audio_modified": False, "automatic_normalization": False, "automatic_limiting": False,
                "elapsed_seconds": time.perf_counter() - started}

    training_path = args.output / "training.json"
    training_path.write_text(json.dumps(progress(), indent=2) + "\n")
    print(json.dumps({"calibration": history[0], "cache_bytes": cache_bytes}), flush=True)
    for step in range(1, STEPS + 1):
        indices = rng.integers(0, len(training), args.batch_size)
        chosen = [training[int(index)] for index in indices]
        starts = []
        for row in chosen:
            if rng.random() < .7:
                start = row["peak_index"] - int(rng.integers(512, SCORED_FRAMES - 512))
            else:
                last_start = len(row["dry"]) - SCORED_FRAMES
                first_start = min(last_start, max(1_024, row["onset"] - 256))
                start = int(rng.integers(first_start, last_start + 1))
            starts.append(max(1_024, min(len(row["dry"]) - SCORED_FRAMES, start)))
        dry = context_windows([row["dry"] for row in chosen], starts).to(target)
        wet = torch.stack([row["wet"][start:start + SCORED_FRAMES] for row, start in zip(chosen, starts)]).to(target)
        original = torch.stack([row["original"][start:start + SCORED_FRAMES] for row, start in zip(chosen, starts)]).to(target)
        controls = torch.stack([row["controls"] for row in chosen]).to(target)
        step_started = time.perf_counter()
        tcn.train()
        prediction = tcn(dry, controls)[0][:, LEFT_CONTEXT:] + original
        per_file = loss_per_example(prediction, wet)
        loss = .5 * per_file.mean() + .5 * per_file.topk(max(1, args.batch_size // 3)).values.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tcn.parameters(), 2.)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            row = {"step": step, "loss": float(loss.detach()), "step_seconds": time.perf_counter() - step_started,
                   "elapsed_seconds": time.perf_counter() - started}
            updates.append(row)
            print(json.dumps(row), flush=True)
            training_path.write_text(json.dumps(progress(), indent=2) + "\n")
        if step % CALIBRATION_INTERVAL == 0:
            measured = _evaluate(CachedTCNRenderer(tcn), cal_dataset, target, 6)
            history.append({"step": step, **measured})
            score = measured["absolute_peak_error_p95"] + max(0., measured["global_esr"] - .03)
            if score < best_score:
                best_score, best_step = score, step
                save_best()
            training_path.write_text(json.dumps(progress(), indent=2) + "\n")
            print(json.dumps({"calibration": history[-1], "best_step": best_step}), flush=True)
    result = progress()
    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
    tcn.cpu().load_state_dict(selected["tcn_state_dict"])
    del training, calibration, cal_dataset, chosen
    torch.mps.empty_cache()
    assert_inputs_unchanged()
    model = GenericLongTCNResidual(base.cpu(), tcn.cpu()).eval()
    print(json.dumps({"stage": "full-actual-model-cpu-calibration", "examples": 54, "best_step": best_step}), flush=True)
    result["calibration"] = _evaluate(model, _rows(cal, "dfz"), torch.device("cpu"), 8)
    assert_inputs_unchanged()
    result["source_code_and_audio_sha256_reverified_after_cache_and_final_cpu_calibration"] = True
    result["passes_full_calibration_gate"] = result["calibration"]["passes_selection_gate"]
    result["selected_residual_sha256"] = sha256(selected_path)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["status"] = "passed-calibration-only-needs-schema-export" if result["passes_full_calibration_gate"] else "rejected-full-calibration"
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "calibration": result["calibration"], "best_step": best_step}), flush=True)


if __name__ == "__main__":
    main()
