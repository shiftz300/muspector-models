"""Fixed20000-step fast-tap coordinate refinement; unchanged runtime geometry.

Only optimization coordinates change: first four fast convolutions use an
exact-zero-delta full-rank temporal basis, with derivative step scale8. Saved
checkpoints contain ordinary float32 convolution weights, not custom runtime
operators, gain stages, or audio preprocessing.
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
from .multirate_fuzz import MultirateFuzz
from .train_asrnn_phase7 import _evaluate
from .train_dfz_core_tcn import selection_key
from .train_dfz_signed_tcn import active_window_ranges, signed_loss_per_example, uniform_active_start
from .train_dfz_transient_narx import CornerCycle
from .train_multirate_fuzz import SCORED_FRAMES, render_training, sha256
from .widen_multirate_fuzz import WIDE_GEOMETRY


STEPS, CALIBRATION_INTERVAL = 20000, 2000
SOURCE = Path("remix/runs/dfz-wide-8000-phase11/candidate.pt")


def learning_rate(step):
    if not 1 <= step <= STEPS:
        raise ValueError("step outside fixed20000-update phase")
    if step <= 500:
        return 3e-5 + (5e-4 - 3e-5) * step / 500
    if step <= 10000:
        return 5e-4
    return 1e-5 + .5 * (5e-4 - 1e-5) * (1 + math.cos(math.pi * (step - 10000) / 10000))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh experiment directory required")
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("offline Apple GPU required")
    parent_path = SOURCE.with_name("metrics.json")
    parent = json.loads(parent_path.read_text())
    source_hash, parent_hash = sha256(SOURCE), sha256(parent_path)
    if (parent.get("completed_steps") != 8000 or not parent.get("source_and_audio_reverified")
            or parent.get("checkpoint_sha256") != source_hash or parent.get("geometry") != WIDE_GEOMETRY):
        raise ValueError("completed hash-bound8000-update wide phase required")
    for name, digest in parent["source_sha256"].items():
        if sha256(Path(__file__).parent / name) != digest:
            raise ValueError("parent training/source code changed")
    sources = {**parent["source_sha256"], Path(__file__).name: sha256(Path(__file__)),
               "dfz_temporal_basis.py": sha256(Path(__file__).with_name("dfz_temporal_basis.py"))}
    provenance = parent["audio_provenance"]
    if set(provenance) != {"fit", "calibration"} or len(provenance["fit"]) != 234 or len(provenance["calibration"]) != 54:
        raise ValueError("same original234/54 partition required")

    def unchanged():
        if sha256(SOURCE) != source_hash or sha256(parent_path) != parent_hash:
            raise ValueError("parent weights/report changed")
        if any(sha256(Path(__file__).parent / name) != digest for name, digest in sources.items()):
            raise ValueError("experiment source changed")
        if any(sha256(row["path"]) != row["sha256"] for rows in provenance.values() for row in rows):
            raise ValueError("original source audio changed")

    unchanged()
    torch.manual_seed(997)
    rng = np.random.default_rng(997)
    payload = torch.load(SOURCE, map_location="cpu", weights_only=True)
    if (payload["geometry"] != WIDE_GEOMETRY or payload["architecture"] != "causal-multirate-gcn"
            or payload["device"] != "dfz" or payload["sample_rate"] != 48000):
        raise ValueError("same wide float32 model required")
    model = MultirateFuzz(**WIDE_GEOMETRY)
    model.load_state_dict(payload["state_dict"])
    model.to("mps")
    probe, controls = torch.randn(2, 6173, device="mps") * .03, torch.rand(2, 6173, 2, device="mps")
    with torch.inference_mode():
        before = model(probe, controls)[0]
    attach_temporal_basis(model)
    with torch.inference_mode():
        initialization_error = float((before - model(probe, controls)[0]).abs().max())
    initial_weights = materialized_state_dict(model)
    if initialization_error != 0 or not all(torch.equal(value, payload["state_dict"][name]) for name, value in initial_weights.items()):
        raise ValueError("exact initialization/materialized-weight identity failed")
    del initial_weights, before, probe, controls, payload
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
    cycle = CornerCycle(rng)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    initial = _evaluate(model, calibration, torch.device("mps"), 9)
    history, updates = [{"step": 0, **initial}], []
    best_step, best_key = 0, selection_key(initial)
    checkpoint = args.output / "candidate.pt"

    def save():
        torch.save({"experimental_schema": 1, "architecture": "causal-multirate-gcn", "device": "dfz", "sample_rate": 48000,
                    "geometry": WIDE_GEOMETRY, "state_dict": materialized_state_dict(model)}, checkpoint)

    def progress():
        return {"schema": 1, "experiment": "wide-core-temporal-coordinate-fixed20000", "seed": 997,
                "requested_steps": STEPS, "completed_steps": updates[-1]["step"] if updates else 0,
                "calibration_interval": CALIBRATION_INTERVAL, "history": history, "updates": updates, "best_step": best_step,
                "parent_checkpoint_sha256": source_hash, "parent_report_sha256": parent_hash, "optimizer_resume": False,
                "geometry": WIDE_GEOMETRY, "source_sha256": sources, "audio_provenance": provenance,
                "initialization_max_audio_error": initialization_error, "initial_materialized_weights_exact": True,
                "coordinate_transform": "first4 fast3-tap kernels: base+theta@[[1,1,1],[-8,0,8],[8,-16,8]]",
                "weight_decay_domain": "delta coordinates and other original trainable weights; fixed base buffers not decayed",
                "loss": "unchanged same-time signed weighted waveform+.1preemphasis",
                "learning_rate_schedule": "500-step3e-5->5e-4 ramp; constant through10000; cosine to1e-5 at20000",
                "sampling": "unchanged50/50active-uniform/Wet-peak-local4096, balanced26-file corner cycle",
                "corner_coverage": cycle.coverage(), "full_current_lowrate_prefix": True, "real_fast_context_frames": 2046,
                "fit_examples": 234, "calibration_examples": 54, "trainable_state_cached": False,
                "physical_audio_devices_used": False, "official_eval_opened": False, "source_audio_modified": False,
                "audio_preprocessing_added": False, "automatic_gain_or_normalization": False, "automatic_limiting": False,
                "custom_inference_operators_added": False, "admitted": False, "elapsed_seconds": time.perf_counter() - started}

    save()
    print(json.dumps({"calibration": history[0], "initialization_max_audio_error": initialization_error}), flush=True)
    for step in range(1, STEPS + 1):
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
            raise ValueError("nonfinite basis training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 200 == 0:
            updates.append({"step": step, "loss": float(loss.detach()), "learning_rate": learning_rate(step),
                            "elapsed_seconds": time.perf_counter() - started, "driver_bytes": torch.mps.driver_allocated_memory()})
            if updates[-1]["driver_bytes"] > 8_000_000_000:
                raise ValueError("fixed8GB driver memory budget exceeded")
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
    model = MultirateFuzz(**WIDE_GEOMETRY).eval()
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
