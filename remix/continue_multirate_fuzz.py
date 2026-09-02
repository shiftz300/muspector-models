"""A fixed 13500-update independent-core training phase after its1500 warmup.

Weights-only continuation with a fresh AdamW optimizer, not exact optimizer
resume. Keep one best complete checkpoint, no snapshots/audio caches per epoch.
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
from .multirate_fuzz import MultirateFuzz
from .train_asrnn_phase7 import _evaluate
from .train_dfz_core_tcn import selection_key
from .train_dfz_signed_tcn import active_window_ranges, signed_loss_per_example, uniform_active_start
from .train_dfz_transient_narx import CornerCycle
from .train_multirate_fuzz import ARCHITECTURE, SCORED_FRAMES, render_training, sha256


STEPS, CALIBRATION_INTERVAL = 13_500, 1_500
SOURCE = Path("remix/runs/dfz-multirate-phase11/candidate.pt")
SOURCE_REPORT = SOURCE.with_name("metrics.json")


def learning_rate(step):
    """Fixed warmup, plateau, cosine finish; never inspect calibration to set LR."""
    if not 1 <= step <= STEPS:
        raise ValueError("step outside fixed continuation phase")
    if step <= 150:
        return 3e-4 + (1e-3 - 3e-4) * step / 150
    if step <= 7_500:
        return 1e-3
    ratio = (step - 7_500) / (STEPS - 7_500)
    return 3e-5 + .5 * (1e-3 - 3e-5) * (1 + math.cos(math.pi * ratio))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh continuation output directory required")
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("offline Apple GPU training required")
    parent = json.loads(SOURCE_REPORT.read_text())
    parent_hash, source_hash = sha256(SOURCE_REPORT), sha256(SOURCE)
    if parent["completed_steps"] != 1500 or parent["checkpoint_sha256"] != source_hash or not parent["source_and_audio_reverified"]:
        raise ValueError("complete verified1500-update warmup checkpoint required")
    if any(sha256(Path(__file__).parent / name) != digest for name, digest in parent["source_sha256"].items()):
        raise ValueError("warmup model/training sources changed")
    provenance = parent["audio_provenance"]
    if len(provenance["fit"]) != 234 or len(provenance["calibration"]) != 54:
        raise ValueError("same fixed234/54 corpus required")
    code_hashes = {**parent["source_sha256"], Path(__file__).name: sha256(Path(__file__))}

    def unchanged():
        if sha256(SOURCE) != source_hash or sha256(SOURCE_REPORT) != parent_hash:
            raise ValueError("parent checkpoint/evidence changed")
        if any(sha256(Path(__file__).parent / name) != digest for name, digest in code_hashes.items()):
            raise ValueError("continuation source changed")
        if any(sha256(row["path"]) != row["sha256"] for rows in provenance.values() for row in rows):
            raise ValueError("fit/calibration source audio changed")

    unchanged()
    torch.manual_seed(990)
    rng = np.random.default_rng(990)
    model = MultirateFuzz(**ARCHITECTURE)
    payload = torch.load(SOURCE, map_location="cpu", weights_only=True)
    if payload["geometry"] != ARCHITECTURE or payload["architecture"] != "causal-multirate-gcn":
        raise ValueError("continuation architecture differs from warmup")
    model.load_state_dict(payload["state_dict"])
    model.to("mps")
    training, calibration, corners = [], [], [[] for _ in range(9)]
    for split in ("fit", "calibration"):
        for record in provenance[split]:
            path = Path(record["path"])
            dry, wet, controls = read_effect_pair(path, "dfz")
            row = {"dry": torch.from_numpy(dry), "wet": torch.from_numpy(wet), "controls": torch.from_numpy(controls),
                   "attack": int(path.stem.split(",")[0])}
            if split == "fit":
                row.update(path=str(path), file_peak=row["wet"][1024:].abs().amax(),
                           peak_index=int(row["wet"][1024:].abs().argmax()) + 1024,
                           active_ranges=active_window_ranges(row["dry"]))
                a, b = (row["controls"] * 2).long().tolist()
                corners[a * 3 + b].append(row)
                training.append(row)
            else:
                calibration.append(row)
    if any(len(rows) != 26 for rows in corners):
        raise ValueError("balanced original fit corners required")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate(1), weight_decay=1e-5)
    cycle = CornerCycle(rng)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    initial = _evaluate(model, calibration, torch.device("mps"), 9)
    history, updates = [{"step": 0, **initial}], []
    best_step, best_key = 0, selection_key(initial)
    checkpoint = args.output / "candidate.pt"

    def save():
        torch.save({"experimental_schema": 1, "architecture": "causal-multirate-gcn", "device": "dfz", "sample_rate": 48_000,
                    "geometry": ARCHITECTURE, "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}}, checkpoint)

    def progress():
        return {"schema": 1, "experiment": "standalone-multirate-15000-total-update-phase", "architecture": ARCHITECTURE,
                "parent_checkpoint_sha256": source_hash, "parent_report_sha256": parent_hash, "parent_trained_updates": 1500,
                "optimizer_resume": False, "optimizer": "fresh-AdamW", "requested_additional_updates": STEPS,
                "completed_additional_updates": updates[-1]["step"] if updates else 0,
                "learning_rate_schedule": "150-step3e-4->1e-3 ramp;1e-3 through7500;cosine to3e-5 at13500",
                "calibration_interval": CALIBRATION_INTERVAL, "history": history, "updates": updates,
                "best_step": best_step, "corner_coverage": cycle.coverage(), "source_sha256": code_hashes,
                "audio_provenance": provenance, "fit_examples": 234, "calibration_examples": 54,
                "loss": "unchanged same-index signed weighted MSE+.1preemphasis, same50/50active/peak windows",
                "selection": "complete-frozen-gate-first, peak-plus-ESR-penalty, worst-Blend",
                "lowrate_prefix": "complete recording recomputed with current weights every update",
                "fast_context_frames": 2046, "trainable_state_cached": False, "teacher_lstm_used": False,
                "frozen_p_cache_used": False, "source_audio_modified": False, "physical_audio_devices_used": False,
                "official_eval_opened": False, "automatic_normalization": False, "automatic_gain": False,
                "automatic_limiting": False, "admitted": False, "elapsed_seconds": time.perf_counter() - started}

    save()
    print(json.dumps({"calibration": history[0]}), flush=True)
    for step in range(1, STEPS + 1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
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
        predicted, wet, peaks = render_training(model, chosen, starts)
        loss = signed_loss_per_example(predicted, wet, peaks).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite continuation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 150 == 0:
            updates.append({"step": step, "loss": float(loss.detach()), "learning_rate": learning_rate(step),
                            "elapsed_seconds": time.perf_counter() - started, "driver_bytes": torch.mps.driver_allocated_memory()})
            if updates[-1]["driver_bytes"] > 8_000_000_000:
                raise ValueError("fixed8GB GPU memory budget exceeded")
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
    model.cpu().load_state_dict(selected["state_dict"])
    del training, corners, chosen
    torch.mps.empty_cache()
    unchanged()
    print(json.dumps({"stage": "complete-selected-cpu54", "best_step": best_step}), flush=True)
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
