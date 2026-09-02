"""Fixed8000 complete-core absolute-waveform refinement, no architecture change.

Uses the same234/54 split, full causal context and bounded target-only frame
weights. Retains a quarter of the signed squared objective; adds normalized
absolute waveform/pre-emphasis supervision. No output correction or new audio.
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
from .dfz_temporal_basis import attach_temporal_basis, materialized_state_dict
from .evaluate_multirate_fuzz import load_candidate, model_from_payload
from .train_asrnn_phase7 import _evaluate
from .train_dfz_core_tcn import selection_key
from .train_dfz_signed_tcn import active_window_ranges, signed_frame_weights, signed_loss_per_example, uniform_active_start
from .train_dfz_transient_narx import CornerCycle
from .train_multirate_fuzz import SCORED_FRAMES, render_training, sha256


STEPS, INTERVAL = 8000, 1000


def learning_rate(step):
    if not 1 <= step <= STEPS:
        raise ValueError("step outside fixed absolute-waveform phase")
    if step <= 500:
        return 1e-5 + 9e-5 * step / 500
    return 1e-5 + .5 * 9e-5 * (1 + math.cos(math.pi * (step - 500) / (STEPS - 500)))


def absolute_loss(prediction, expected, peaks):
    squared = signed_loss_per_example(prediction, expected, peaks)
    waveform = (signed_frame_weights(expected, peaks) * (prediction - expected).abs()).mean(1)
    waveform = waveform / expected.abs().mean(1).clamp_min(1e-4)
    a, b = prediction[:, 1:] - .95*prediction[:, :-1], expected[:, 1:] - .95*expected[:, :-1]
    pre = (a-b).abs().mean(1) / b.abs().mean(1).clamp_min(1e-4)
    return waveform + .1*pre + .25*squared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh complete-core experiment directory required")
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("offline MPS compute required")
    model, payload, parent = load_candidate(args.checkpoint)
    source_hash, parent_hash = sha256(args.checkpoint), sha256(args.checkpoint.with_name("metrics.json"))
    names = (Path(__file__).name, "evaluate_multirate_fuzz.py", "dfz_temporal_basis.py", "train_multirate_fuzz.py",
             "train_dfz_signed_tcn.py", "train_dfz_transient_narx.py", "train_dfz_core_tcn.py", "train_asrnn_phase7.py")
    sources = {**parent["source_sha256"], **{name: sha256(Path(__file__).parent / name) for name in names}}
    provenance = parent["audio_provenance"]
    if set(provenance) != {"fit", "calibration"} or len(provenance["fit"]) != 234 or len(provenance["calibration"]) != 54:
        raise ValueError("original234/54 partition required")

    def unchanged():
        if sha256(args.checkpoint) != source_hash or sha256(args.checkpoint.with_name("metrics.json")) != parent_hash:
            raise ValueError("parent changed")
        if any(sha256(Path(__file__).parent / name) != digest for name, digest in sources.items()):
            raise ValueError("training source changed")
        if any(sha256(row["path"]) != row["sha256"] for rows in provenance.values() for row in rows):
            raise ValueError("original audio changed")

    unchanged()
    torch.manual_seed(1006)
    rng = np.random.default_rng(1006)
    model = attach_temporal_basis(model.requires_grad_(True).to("mps"))
    initial_weights = materialized_state_dict(model)
    if any(not torch.equal(value, initial_weights[name]) for name, value in payload["state_dict"].items()):
        raise ValueError("source weights changed at initialization")
    del initial_weights
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
                row.update(file_peak=body.abs().amax(), peak_index=int(body.abs().argmax())+1024,
                           active_ranges=active_window_ranges(row["dry"]))
                a, b = (row["controls"]*2).long().tolist()
                corners[a*3+b].append(row)
            else:
                calibration.append(row)
    if any(len(rows) != 26 for rows in corners):
        raise ValueError("balanced nine-corner coverage required")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate(1), weight_decay=1e-5)
    cycle = CornerCycle(rng)
    started = time.perf_counter()
    initial = _evaluate(model, calibration, torch.device("mps"), 9)
    history, updates = [{"step": 0, **initial}], []
    best_step, best_key = 0, selection_key(initial)
    args.output.mkdir(parents=True)
    checkpoint = args.output / "candidate.pt"

    def save():
        torch.save({**{key: payload[key] for key in ("experimental_schema", "architecture", "device", "sample_rate", "geometry")},
                    "state_dict": materialized_state_dict(model)}, checkpoint)

    def progress():
        return {"schema": 1, "experiment": "absolute-waveform-complete-core-fixed8000", "seed": 1006,
                "requested_steps": STEPS, "completed_steps": updates[-1]["step"] if updates else 0,
                "history": history, "updates": updates, "best_step": best_step, "calibration_interval": INTERVAL,
                "architecture": payload["architecture"], "geometry": payload["geometry"],
                "parent_checkpoint_sha256": source_hash, "parent_report_sha256": parent_hash,
                "source_sha256": sources, "audio_provenance": provenance, "optimizer_resume": False,
                "loss": "normalized signed-weighted MAE +.1normalized pre95 MAE +.25previous signed squared objective",
                "learning_rate_schedule": "500-step1e-5 to1e-4 ramp, cosine to1e-5 at8000",
                "sampling": "unchanged50/50active-uniform/Wet-peak-local4096, balanced26-file corner cycle",
                "optimization_coordinates": "same first4 fast kernels full-rank delta temporal basis scale8",
                "corner_coverage": cycle.coverage(), "full_current_lowrate_prefix": True, "real_fast_context_frames": 2046,
                "fit_examples": 234, "calibration_examples": 54, "trainable_state_cached": False,
                "physical_audio_devices_used": False, "official_eval_opened": False, "source_audio_modified": False,
                "automatic_gain_or_normalization": False, "automatic_limiting": False, "admitted": False,
                "elapsed_seconds": time.perf_counter()-started}

    save()
    print(json.dumps({"calibration": history[0]}), flush=True)
    for step in range(1, STEPS+1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        chosen = [rows[index] for rows, index in zip(corners, cycle.next())]
        starts = [uniform_active_start(rng, row["active_ranges"]) if rng.random() < .5 else
                  max(1024, min(len(row["dry"])-SCORED_FRAMES, row["peak_index"]-int(rng.integers(512, SCORED_FRAMES-512))))
                  for row in chosen]
        model.train()
        predicted, expected, peaks = render_training(model, chosen, starts)
        loss = absolute_loss(predicted, expected, peaks).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite absolute-waveform loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 200 == 0:
            updates.append({"step": step, "loss": float(loss.detach()), "learning_rate": learning_rate(step),
                            "elapsed_seconds": time.perf_counter()-started, "driver_bytes": torch.mps.driver_allocated_memory()})
            if updates[-1]["driver_bytes"] > 8_000_000_000:
                raise ValueError("fixed8GB driver memory budget exceeded")
            (args.output / "training.json").write_text(json.dumps(progress(), indent=2)+"\n")
            print(json.dumps(updates[-1]), flush=True)
        if step % INTERVAL == 0:
            measured = _evaluate(model, calibration, torch.device("mps"), 9)
            history.append({"step": step, **measured})
            if selection_key(measured) < best_key:
                best_step, best_key = step, selection_key(measured)
                save()
            (args.output / "training.json").write_text(json.dumps(progress(), indent=2)+"\n")
            print(json.dumps({"calibration": history[-1], "best_step": best_step}), flush=True)
    result = progress()
    selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
    del model, optimizer, corners, chosen, payload
    torch.mps.empty_cache()
    model = model_from_payload(selected)
    unchanged()
    print(json.dumps({"stage": "complete-selected-cpu54", "best_step": best_step}), flush=True)
    result["calibration"] = _evaluate(model, calibration, torch.device("cpu"), 9)
    unchanged()
    result.update(checkpoint_sha256=sha256(checkpoint), source_and_audio_reverified=True,
                  passes_full_calibration_gate=result["calibration"]["passes_selection_gate"],
                  status="passed-calibration-only" if result["calibration"]["passes_selection_gate"] else "rejected-full-calibration")
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "calibration": result["calibration"]}), flush=True)


if __name__ == "__main__":
    main()
