"""Conditional-threshold candidate after the completed fixed20000 core phase.

Fixed12000 updates: threshold heads alone for1000, then all trainable weights.
Zero-origin activation and ordinary materialized convolution weights preserve
the offline/stateful contract. No audio normalization, limiter, hardware or eval.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .asrnn_effects import read_effect_pair
from .centered_multirate_fuzz import CenteredMultirateFuzz
from .dfz_temporal_basis import attach_temporal_basis, materialized_state_dict
from .multirate_fuzz import MultirateFuzz
from .train_asrnn_phase7 import _evaluate
from .train_dfz_core_tcn import selection_key
from .train_dfz_signed_tcn import active_window_ranges, signed_loss_per_example, uniform_active_start
from .train_dfz_transient_narx import CornerCycle
from .train_multirate_fuzz import SCORED_FRAMES, render_training, sha256
from .widen_multirate_fuzz import WIDE_GEOMETRY


STEPS, HEAD_ONLY_STEPS, CALIBRATION_INTERVAL = 12000, 1000, 1000
SOURCE = Path("remix/runs/dfz-wide-basis-20000-phase11/candidate.pt")
PREFLIGHT = Path("remix/runs/dfz-centered-multirate-precheck-phase11/metrics.json")
ARCHITECTURE = "zero-origin-centered-causal-multirate"


def learning_rate(step):
    if not 1 <= step <= STEPS:
        raise ValueError("step outside fixed centered-core phase")
    if step <= HEAD_ONLY_STEPS:
        return 3e-4
    if step <= 6000:
        return 2e-4
    return 1e-5 + .5 * (2e-4 - 1e-5) * (1 + math.cos(math.pi * (step - 6000) / 6000))


def set_head_only(model, enabled):
    model.requires_grad_(not enabled)
    if enabled:
        for layer in (*model.audio.activation_current, *model.audio.activation_slow):
            layer.requires_grad_(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new experiment directory required")
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("offline MPS compute required")
    parent_path = SOURCE.with_name("metrics.json")
    parent = json.loads(parent_path.read_text())
    source_hash, parent_hash = sha256(SOURCE), sha256(parent_path)
    if (parent.get("completed_steps") != 20000 or parent.get("checkpoint_sha256") != source_hash
            or not parent.get("source_and_audio_reverified") or parent.get("geometry") != WIDE_GEOMETRY):
        raise ValueError("completed source-bound20000-update phase required")
    preflight = json.loads(PREFLIGHT.read_text())
    preflight_hash = sha256(PREFLIGHT)
    if (not preflight.get("passed") or preflight.get("trained") is not False or preflight.get("geometry") != WIDE_GEOMETRY
            or not preflight.get("nonzero_threshold_heads_tested") or preflight.get("architecture") != ARCHITECTURE):
        raise ValueError("nonzero-threshold single-graph synthetic preflight required")
    for report in (parent, preflight):
        for name, digest in report["source_sha256"].items():
            if sha256(Path(__file__).parent / name) != digest:
                raise ValueError("source/preflight implementation changed")
    sources = {**parent["source_sha256"], **preflight["source_sha256"], Path(__file__).name: sha256(Path(__file__))}
    provenance = parent["audio_provenance"]
    if set(provenance) != {"fit", "calibration"} or len(provenance["fit"]) != 234 or len(provenance["calibration"]) != 54:
        raise ValueError("same original234/54 partition required")

    def unchanged():
        if sha256(SOURCE) != source_hash or sha256(parent_path) != parent_hash or sha256(PREFLIGHT) != preflight_hash:
            raise ValueError("source weights/training/preflight evidence changed")
        if sha256(PREFLIGHT.with_name("model.onnx")) != preflight["graph_sha256"]:
            raise ValueError("preflight graph changed")
        if any(sha256(Path(__file__).parent / name) != digest for name, digest in sources.items()):
            raise ValueError("experiment source changed")
        if any(sha256(row["path"]) != row["sha256"] for values in provenance.values() for row in values):
            raise ValueError("original audio changed")

    unchanged()
    torch.manual_seed(1002)
    rng = np.random.default_rng(1002)
    payload = torch.load(SOURCE, map_location="cpu", weights_only=True)
    if payload["geometry"] != WIDE_GEOMETRY or payload["architecture"] != "causal-multirate-gcn":
        raise ValueError("plain wide-core warmstart required")
    base = MultirateFuzz(**WIDE_GEOMETRY).to("mps")
    base.load_state_dict(payload["state_dict"])
    model = attach_temporal_basis(CenteredMultirateFuzz.from_base(base))
    probe, knobs = torch.randn(2, 6173, device="mps") * .03, torch.rand(2, 6173, 2, device="mps")
    with torch.inference_mode():
        initial_error = float((base(probe, knobs)[0] - model(probe, knobs)[0]).abs().max())
    initial_weights = materialized_state_dict(model)
    if initial_error != 0 or not all(torch.equal(value, initial_weights[name]) for name, value in payload["state_dict"].items()):
        raise ValueError("zero-head initial audio/source-weight identity failed")
    del base, payload, initial_weights, probe, knobs
    corners, calibration = [[] for _ in range(9)], []
    for split in ("fit", "calibration"):
        for record in provenance[split]:
            path = Path(record["path"])
            if "train" not in path.parts or (int(path.stem.split(",")[-1]) % 5 == 0) != (split == "calibration"):
                raise ValueError("original train partition changed")
            dry, wet, controls = read_effect_pair(path, "dfz")
            row = {"dry": torch.from_numpy(dry), "wet": torch.from_numpy(wet), "controls": torch.from_numpy(controls),
                   "attack": int(path.stem.split(",")[0])}
            if split == "fit":
                body = row["wet"][1024:]
                row.update(file_peak=body.abs().amax(), peak_index=int(body.abs().argmax()) + 1024,
                           active_ranges=active_window_ranges(row["dry"]))
                a, b = (row["controls"] * 2).long().tolist()
                corners[a * 3 + b].append(row)
            else:
                calibration.append(row)
    if any(len(rows) != 26 for rows in corners):
        raise ValueError("balanced nine fit corners required")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate(1), weight_decay=1e-5)
    set_head_only(model, True)
    cycle = CornerCycle(rng)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    initial = _evaluate(model, calibration, torch.device("mps"), 9)
    history, updates = [{"step": 0, **initial}], []
    best_step, best_key = 0, selection_key(initial)
    checkpoint = args.output / "candidate.pt"

    def save():
        torch.save({"experimental_schema": 1, "architecture": ARCHITECTURE, "device": "dfz", "sample_rate": 48000,
                    "geometry": WIDE_GEOMETRY, "state_dict": materialized_state_dict(model)}, checkpoint)

    def progress():
        return {"schema": 1, "experiment": "zero-origin-threshold-core-fixed12000", "seed": 1002,
                "requested_steps": STEPS, "completed_steps": updates[-1]["step"] if updates else 0, "head_only_steps": HEAD_ONLY_STEPS,
                "calibration_interval": CALIBRATION_INTERVAL, "history": history, "updates": updates, "best_step": best_step,
                "architecture": ARCHITECTURE, "geometry": WIDE_GEOMETRY, "parent_checkpoint_sha256": source_hash,
                "parent_report_sha256": parent_hash, "parent_selected_step": parent["best_step"], "preflight_sha256": preflight_hash,
                "source_sha256": sources, "audio_provenance": provenance, "optimizer_resume": False,
                "initialization_max_audio_error": initial_error, "source_parameters_initially_exact": True,
                "new_parameters": 6080, "activation": "tanh(activation+threshold)-tanh(threshold), zero-initialized conditional threshold",
                "optimization_coordinates": "first4 fast kernels retain full-rank delta temporal basis scale8",
                "loss": "unchanged signed weighted waveform+.1preemphasis",
                "learning_rate_schedule": "head-only1000 at3e-4; joint2e-4 through6000; cosine to1e-5 at12000",
                "sampling": "unchanged50/50active-uniform/Wet-peak-local4096, balanced26-file corner cycle",
                "corner_coverage": cycle.coverage(), "full_current_lowrate_prefix": True, "real_fast_context_frames": 2046,
                "fit_examples": 234, "calibration_examples": 54, "trainable_state_cached": False,
                "physical_audio_devices_used": False, "official_eval_opened": False, "source_audio_modified": False,
                "automatic_gain_or_normalization": False, "automatic_limiting": False, "admitted": False,
                "elapsed_seconds": time.perf_counter() - started}

    save()
    print(json.dumps({"calibration": history[0], "initialization_error": initial_error}), flush=True)
    for step in range(1, STEPS + 1):
        if step == HEAD_ONLY_STEPS + 1:
            set_head_only(model, False)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        chosen = [rows[index] for rows, index in zip(corners, cycle.next())]
        starts = [uniform_active_start(rng, row["active_ranges"]) if rng.random() < .5
                  else max(1024, min(len(row["dry"]) - SCORED_FRAMES,
                                    row["peak_index"] - int(rng.integers(512, SCORED_FRAMES - 512)))) for row in chosen]
        model.train()
        predicted, expected, peaks = render_training(model, chosen, starts)
        loss = signed_loss_per_example(predicted, expected, peaks).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite centered-core loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 200 == 0:
            updates.append({"step": step, "loss": float(loss.detach()), "head_only": step <= HEAD_ONLY_STEPS,
                            "learning_rate": learning_rate(step), "elapsed_seconds": time.perf_counter() - started,
                            "driver_bytes": torch.mps.driver_allocated_memory()})
            if updates[-1]["driver_bytes"] > 8_000_000_000:
                raise ValueError("fixed8GB driver budget exceeded")
            (args.output / "training.json").write_text(json.dumps(progress(), indent=2) + "\n")
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
    selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
    del model, optimizer, corners, chosen
    torch.mps.empty_cache()
    model = CenteredMultirateFuzz(**WIDE_GEOMETRY).eval()
    model.load_state_dict(selected["state_dict"])
    unchanged()
    print(json.dumps({"stage": "complete-materialized-selected-cpu54", "best_step": best_step}), flush=True)
    result["calibration"] = _evaluate(model, calibration, torch.device("cpu"), 9)
    unchanged()
    result.update(checkpoint_sha256=sha256(checkpoint), source_and_audio_reverified=True,
                  passes_full_calibration_gate=result["calibration"]["passes_selection_gate"],
                  status="passed-calibration-only" if result["calibration"]["passes_selection_gate"] else "rejected-full-calibration",
                  elapsed_seconds=time.perf_counter() - started)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "calibration": result["calibration"]}), flush=True)


if __name__ == "__main__":
    main()
