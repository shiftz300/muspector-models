"""Fixed first training phase for an independent causal two-rate fuzz model.

1500 updates and full cal54 every150; fit234 only; no source LSTM, original-p
features, resampling, target shifting, output gain or audio-device access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .asrnn_effects import effect_files, read_effect_pair
from .fit_asrnn_effect_output import _partition
from .multirate_fuzz import BLOCK, MultirateFuzz
from .train_asrnn_phase7 import _evaluate
from .train_dfz_core_tcn import selection_key
from .train_dfz_long_tcn import context_windows
from .train_dfz_signed_tcn import active_window_ranges, signed_loss_per_example, uniform_active_start
from .train_dfz_transient_narx import CornerCycle


STEPS, CALIBRATION_INTERVAL, SCORED_FRAMES, LEARNING_RATE = 1500, 150, 4096, 3e-4
PRECHECK = Path("remix/runs/dfz-multirate-precheck-phase11/metrics.json")
ARCHITECTURE = {"audio_width": 16, "audio_blocks": 10, "controller_width": 8, "controller_blocks": 10}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_training(model, rows, starts):
    """Current weights, full causal low-rate history and exact fast history."""
    complete = torch.stack([row["dry"] for row in rows]).to("mps")
    controls = torch.stack([row["controls"] for row in rows]).to("mps")
    held = model.training_held(complete, controls)
    left = model.audio.receptive_field - 1
    dry = context_windows([row["dry"] for row in rows], starts, SCORED_FRAMES, left).to("mps")
    times = torch.tensor(starts, device="mps")[:, None] - left + torch.arange(left + SCORED_FRAMES, device="mps")[None]
    indices = times.clamp_min(0) // BLOCK
    prediction, _ = model.audio(dry, controls[:, None].expand(-1, dry.shape[1], -1), held, indices)
    expected = torch.stack([row["wet"][start:start + SCORED_FRAMES] for row, start in zip(rows, starts)]).to("mps")
    peaks = torch.stack([row["file_peak"] for row in rows]).to("mps")
    return prediction[:, left:], expected, peaks


def smoke():
    torch.manual_seed(989)
    model = MultirateFuzz(**ARCHITECTURE).to("mps")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    rows = []
    for a in (0., .5, 1.):
        for b in (0., .5, 1.):
            dry = torch.randn(144_000) * .03
            wet = torch.tanh(dry * 7) * .8
            rows.append({"dry": dry, "wet": wet, "controls": torch.tensor((a, b)), "file_peak": wet.abs().amax()})
    results = []
    for step in range(3):
        torch.mps.synchronize()
        started = time.perf_counter()
        prediction, wet, peaks = render_training(model, rows, [50_000 + 64 * index for index in range(9)])
        loss = signed_loss_per_example(prediction, wet, peaks).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        torch.mps.synchronize()
        results.append({"step": step, "seconds": time.perf_counter() - started, "loss": float(loss.detach()),
                        "driver_bytes": torch.mps.driver_allocated_memory(), "allocated_bytes": torch.mps.current_allocated_memory()})
    with torch.inference_mode():
        # The exact streaming implementation, including int clock/dynamic shapes,
        # must also work on MPS before full calibration uses it.
        dry = torch.stack([row["dry"][:2051] for row in rows[:2]]).to("mps")
        controls = torch.rand(2, 2051, 2, device="mps")
        whole, _ = model(dry, controls)
        a, state = model(dry[:, :17], controls[:, :17])
        b, _ = model(dry[:, 17:], controls[:, 17:], state)
        stream = float((whole - torch.cat((a, b), 1)).abs().max())
        zero = float(model(torch.zeros_like(dry), controls)[0].abs().max())
    if stream > 2e-6 or zero != 0 or max(row["driver_bytes"] for row in results) > 8_000_000_000:
        raise ValueError("MPS smoke failed stream/zero/resource precondition")
    return {"measurements": results, "mps_stream_max_error": stream, "mps_zero_peak": zero,
            "source_audio_loaded": False, "physical_audio_devices_used": False, "teacher_lstm_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("this offline training requires Apple GPU compute")
    if args.smoke_only:
        print(json.dumps(smoke()), flush=True)
        return
    if args.output is None or args.output.exists():
        raise ValueError("new training output directory required")
    precheck = json.loads(PRECHECK.read_text())
    precheck_hash = sha256(PRECHECK)
    if not precheck["passed"] or not precheck["single_onnx_graph"] or precheck["trained"]:
        raise ValueError("completed standalone architecture precheck required")
    if any(sha256(Path(__file__).parent / name) != digest for name, digest in precheck["source_sha256"].items()):
        raise ValueError("prechecked architecture changed")
    if sha256(PRECHECK.parent / "model.onnx") != precheck["graph_sha256"]:
        raise ValueError("precheck graph changed")
    names = (Path(__file__).name, "multirate_fuzz.py", "train_dfz_core_tcn.py", "train_dfz_long_tcn.py",
             "train_dfz_signed_tcn.py", "train_dfz_transient_narx.py", "train_asrnn_phase7.py",
             "asrnn_effects.py", "asrnn_data.py", "fit_asrnn_effect_output.py")
    code_hashes = {name: sha256(Path(__file__).parent / name) for name in names}
    fit, cal = _partition(effect_files(Path("data/corpus/asrnn-physical-effects"), "dfz", "train"))
    if len(fit) != 234 or len(cal) != 54:
        raise ValueError("frozen fit234/cal54 required")
    provenance = {name: [{"path": str(path), "sha256": sha256(path)} for path in paths]
                  for name, paths in (("fit", fit), ("calibration", cal))}

    def unchanged():
        if sha256(PRECHECK) != precheck_hash:
            raise ValueError("architecture precheck changed")
        if any(sha256(Path(__file__).parent / name) != digest for name, digest in code_hashes.items()):
            raise ValueError("training source changed")
        if any(sha256(row["path"]) != row["sha256"] for rows in provenance.values() for row in rows):
            raise ValueError("original fit/calibration audio changed")

    torch.manual_seed(989)
    rng = np.random.default_rng(989)
    rows_by_split = []
    corners = [[] for _ in range(9)]
    for paths in (fit, cal):
        rows = []
        for path in paths:
            dry, wet, controls = read_effect_pair(path, "dfz")
            if len(dry) != 144_000:
                raise ValueError("complete three-second DFZ recordings required")
            row = {"dry": torch.from_numpy(dry), "wet": torch.from_numpy(wet),
                   "controls": torch.from_numpy(controls), "attack": int(path.stem.split(",")[0])}
            if paths is fit:
                row.update(path=str(path), file_peak=row["wet"][1024:].abs().amax(),
                           peak_index=int(row["wet"][1024:].abs().argmax()) + 1024,
                           active_ranges=active_window_ranges(row["dry"]))
                a, b = (row["controls"] * 2).long().tolist()
                corners[a * 3 + b].append(row)
            rows.append(row)
        rows_by_split.append(rows)
    if any(len(rows) != 26 for rows in corners):
        raise ValueError("nine corners with26 fit recordings required")
    training, calibration = rows_by_split
    unchanged()
    args.output.mkdir(parents=True)
    model = MultirateFuzz(**ARCHITECTURE).to("mps")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    cycle = CornerCycle(rng)
    started = time.perf_counter()
    initial = _evaluate(model, calibration, torch.device("mps"), 9)
    history, updates = [{"step": 0, **initial}], []
    best_step, best_key = 0, selection_key(initial)
    checkpoint = args.output / "candidate.pt"

    def save():
        torch.save({"experimental_schema": 1, "architecture": "causal-multirate-gcn", "device": "dfz",
                    "sample_rate": 48_000, "geometry": ARCHITECTURE,
                    "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}}, checkpoint)

    def progress():
        return {"schema": 1, "experiment": "standalone-causal-two-rate-fuzz", "architecture": ARCHITECTURE,
                "steps": STEPS, "completed_steps": updates[-1]["step"] if updates else 0,
                "calibration_interval": CALIBRATION_INTERVAL, "learning_rate": LEARNING_RATE,
                "scored_frames": SCORED_FRAMES, "batch_size": 9, "history": history, "updates": updates,
                "best_step": best_step, "corner_coverage": cycle.coverage(),
                "fit_examples": len(fit), "calibration_examples": len(cal), "source_sha256": code_hashes,
                "audio_provenance": provenance, "precheck_sha256": precheck_hash,
                "full_lowrate_causal_prefix_recomputed_each_update": True,
                "real_fast_context_frames": model.audio.receptive_field - 1,
                "trainable_recurrent_state_cached": False, "loss": "fixed same-time signed weighted MSE + 0.1preemphasis",
                "sampling": "balanced corner-cycle, half active-uniform and half Wet-peak-local",
                "selection": "complete-frozen-gate-first, peak-plus-ESR-penalty, worst-Blend",
                "teacher_lstm_used": False, "frozen_p_cache_used": False,
                "physical_audio_devices_used": False, "official_eval_opened": False,
                "source_audio_modified": False, "automatic_normalization": False, "automatic_gain": False,
                "automatic_limiting": False, "admitted": False, "elapsed_seconds": time.perf_counter() - started}

    save()
    print(json.dumps({"calibration": history[0]}), flush=True)
    for step in range(1, STEPS + 1):
        chosen = [rows[index] for rows, index in zip(corners, cycle.next())]
        starts = []
        for row in chosen:
            if rng.random() < .5:
                start = uniform_active_start(rng, row["active_ranges"])
            else:
                start = max(1024, min(len(row["dry"]) - SCORED_FRAMES,
                                     row["peak_index"] - int(rng.integers(512, SCORED_FRAMES - 512))))
            starts.append(start)
        model.train()
        prediction, wet, peaks = render_training(model, chosen, starts)
        loss = signed_loss_per_example(prediction, wet, peaks).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite standalone training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            updates.append({"step": step, "loss": float(loss.detach()), "elapsed_seconds": time.perf_counter() - started,
                            "driver_bytes": torch.mps.driver_allocated_memory()})
            if updates[-1]["driver_bytes"] > 8_000_000_000:
                raise ValueError("training exceeds the fixed8GB GPU budget")
            print(json.dumps(updates[-1]), flush=True)
        if step % CALIBRATION_INTERVAL == 0:
            measured = _evaluate(model, calibration, torch.device("mps"), 9)
            history.append({"step": step, **measured})
            if selection_key(measured) < best_key:
                best_step, best_key = step, selection_key(measured)
                save()
            (args.output / "training.json").write_text(json.dumps(progress(), indent=2) + "\n")
            print(json.dumps({"calibration": history[-1], "best_step": best_step}), flush=True)
    result = progress()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.cpu().load_state_dict(payload["state_dict"])
    del training, corners, chosen, rows_by_split
    torch.mps.empty_cache()
    unchanged()
    print(json.dumps({"stage": "actual-standalone-cpu54", "best_step": best_step}), flush=True)
    result["calibration"] = _evaluate(model.eval(), calibration, torch.device("cpu"), 9)
    unchanged()
    result["source_and_audio_reverified"] = True
    result["checkpoint_sha256"] = sha256(checkpoint)
    result["passes_full_calibration_gate"] = result["calibration"]["passes_selection_gate"]
    result["status"] = "passed-calibration-only" if result["passes_full_calibration_gate"] else "rejected-full-calibration"
    result["elapsed_seconds"] = time.perf_counter() - started
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "calibration": result["calibration"]}), flush=True)


if __name__ == "__main__":
    main()
