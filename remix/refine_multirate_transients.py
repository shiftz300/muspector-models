"""Fixed same-time worst-error refinement of the independent trained core.

The 16 largest squared sample errors in each real scored window supplement
the existing signed waveform loss. This never compares independent output
and target extrema, and does not add gains, clipping or target alignment.
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


STEPS, CALIBRATION_INTERVAL, TAIL_SAMPLES, TAIL_WEIGHT = 6000, 1000, 16, .1
SOURCE = Path("remix/runs/dfz-multirate-15000-phase11/candidate.pt")


def tail_loss(predicted, expected, energy):
    """Worst aligned errors, not amplitude-only extrema or shifted targets."""
    if predicted.shape != expected.shape or expected.ndim != 2 or expected.shape[1] < TAIL_SAMPLES:
        raise ValueError("aligned scored windows with at least16 samples required")
    if energy.shape != (expected.shape[0],) or not torch.isfinite(energy).all() or (energy < 0).any():
        raise ValueError("one finite nonnegative full-file fit energy per window required")
    return (predicted - expected).square().topk(TAIL_SAMPLES, dim=1).values.mean(1) / energy.detach().clamp_min(1e-5)


def learning_rate(step):
    if not 1 <= step <= STEPS:
        raise ValueError("step outside fixed refinement")
    return 1e-5 + .5 * (3e-4 - 1e-5) * (1 + math.cos(math.pi * (step - 1) / (STEPS - 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new refinement directory required")
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("offline MPS compute required")
    parent_path = SOURCE.with_name("metrics.json")
    parent = json.loads(parent_path.read_text())
    source_hash, parent_hash = sha256(SOURCE), sha256(parent_path)
    if (parent.get("completed_additional_updates") != 13500 or not parent.get("source_and_audio_reverified")
            or parent.get("checkpoint_sha256") != source_hash):
        raise ValueError("completed fixed15000-update phase required")
    provenance = parent["audio_provenance"]
    if set(provenance) != {"fit", "calibration"} or len(provenance["fit"]) != 234 or len(provenance["calibration"]) != 54:
        raise ValueError("same original234/54 partition required")
    sources = {**parent["source_sha256"], Path(__file__).name: sha256(Path(__file__))}

    def unchanged():
        if sha256(SOURCE) != source_hash or sha256(parent_path) != parent_hash:
            raise ValueError("parent weights/report changed")
        if any(sha256(Path(__file__).parent / name) != digest for name, digest in sources.items()):
            raise ValueError("refinement source changed")
        if any(sha256(row["path"]) != row["sha256"] for rows in provenance.values() for row in rows):
            raise ValueError("original fit/calibration audio changed")

    unchanged()
    torch.manual_seed(992)
    rng = np.random.default_rng(992)
    payload = torch.load(SOURCE, map_location="cpu", weights_only=True)
    if payload["geometry"] != ARCHITECTURE or payload["architecture"] != "causal-multirate-gcn":
        raise ValueError("same independent architecture required")
    model = MultirateFuzz(**ARCHITECTURE)
    model.load_state_dict(payload["state_dict"])
    model.to("mps")
    corners, calibration = [[] for _ in range(9)], []
    for split in ("fit", "calibration"):
        for record in provenance[split]:
            path = Path(record["path"])
            if "train" not in path.parts or (int(path.stem.split(",")[-1]) % 5 == 0) != (split == "calibration"):
                raise ValueError("frozen original train partition changed")
            dry, wet, controls = read_effect_pair(path, "dfz")
            row = {"dry": torch.from_numpy(dry), "wet": torch.from_numpy(wet), "controls": torch.from_numpy(controls),
                   "attack": int(path.stem.split(",")[0])}
            if split == "fit":
                body = row["wet"][1024:]
                row.update(file_peak=body.abs().amax(), peak_index=int(body.abs().argmax()) + 1024,
                           file_energy=body.square().mean(), active_ranges=active_window_ranges(row["dry"]))
                a, b = (row["controls"] * 2).long().tolist()
                corners[a * 3 + b].append(row)
            else:
                calibration.append(row)
    if any(len(rows) != 26 for rows in corners):
        raise ValueError("balanced nine fit corners required")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate(1), weight_decay=1e-5)
    cycle = CornerCycle(rng)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    initial = _evaluate(model, calibration, torch.device("mps"), 9)
    history, updates = [{"step": 0, **initial}], []
    best_step, best_key = 0, selection_key(initial)
    checkpoint = args.output / "candidate.pt"

    def save():
        torch.save({"experimental_schema": 1, "architecture": "causal-multirate-gcn", "device": "dfz", "sample_rate": 48000,
                    "geometry": ARCHITECTURE, "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}}, checkpoint)

    def progress():
        return {"schema": 1, "experiment": "independent-core-same-time-worst16-refinement", "seed": 992,
                "requested_steps": STEPS, "completed_steps": updates[-1]["step"] if updates else 0,
                "calibration_interval": CALIBRATION_INTERVAL, "history": history, "updates": updates, "best_step": best_step,
                "optimizer_resume": False, "parent_checkpoint_sha256": source_hash, "parent_report_sha256": parent_hash,
                "architecture": ARCHITECTURE, "source_sha256": sources, "audio_provenance": provenance,
                "loss": "existing signed weighted waveform+.1preemphasis + .1*mean(worst16 same-time squared errors)/max(full-fit-file Wet energy,1e-5)",
                "sampling": "unchanged50/50active-uniform/Wet-peak-local4096, balanced26-file corner cycle",
                "learning_rate_schedule": "fixed cosine3e-4->1e-5 over6000 updates",
                "corner_coverage": cycle.coverage(), "full_current_lowrate_prefix": True, "real_fast_context_frames": 2046,
                "fit_examples": 234, "calibration_examples": 54, "trainable_state_cached": False,
                "teacher_lstm_used": False, "physical_audio_devices_used": False, "official_eval_opened": False,
                "automatic_gain": False, "automatic_limiting": False, "automatic_normalization": False,
                "source_audio_modified": False, "admitted": False, "elapsed_seconds": time.perf_counter() - started}

    save()
    print(json.dumps({"calibration": history[0]}), flush=True)
    for step in range(1, STEPS + 1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        chosen = [rows[index] for rows, index in zip(corners, cycle.next())]
        starts = [uniform_active_start(rng, row["active_ranges"]) if rng.random() < .5
                  else max(1024, min(len(row["dry"]) - SCORED_FRAMES,
                                    row["peak_index"] - int(rng.integers(512, SCORED_FRAMES - 512)))) for row in chosen]
        model.train()
        predicted, expected, peaks = render_training(model, chosen, starts)
        energy = torch.stack([row["file_energy"] for row in chosen]).to("mps")
        base = signed_loss_per_example(predicted, expected, peaks).mean()
        tail = tail_loss(predicted, expected, energy).mean()
        loss = base + TAIL_WEIGHT * tail
        if not torch.isfinite(loss):
            raise ValueError("nonfinite refinement loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            updates.append({"step": step, "loss": float(loss.detach()), "base_loss": float(base.detach()),
                            "tail_loss": float(tail.detach()), "learning_rate": learning_rate(step),
                            "elapsed_seconds": time.perf_counter() - started, "driver_bytes": torch.mps.driver_allocated_memory()})
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
    model.cpu().load_state_dict(selected["state_dict"])
    del corners, chosen
    torch.mps.empty_cache()
    unchanged()
    print(json.dumps({"stage": "complete-selected-cpu54", "best_step": best_step}), flush=True)
    result["calibration"] = _evaluate(model.eval(), calibration, torch.device("cpu"), 9)
    unchanged()
    result.update(checkpoint_sha256=sha256(checkpoint), source_and_audio_reverified=True,
                  passes_full_calibration_gate=result["calibration"]["passes_selection_gate"],
                  status="passed-calibration-only" if result["calibration"]["passes_selection_gate"] else "rejected-full-calibration",
                  elapsed_seconds=time.perf_counter() - started)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "calibration": result["calibration"]}), flush=True)


if __name__ == "__main__":
    main()
