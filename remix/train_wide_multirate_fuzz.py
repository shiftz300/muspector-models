"""Fixed8000-update larger-core candidate, never an equivalent replacement.

The widening audit failed its extra cross-model2e-6 equivalence check while
the candidate's own backend/stream/safety/performance checks passed. We retain
that failure and use the weights only as an approximate training warm start.
No fidelity, export or promotion threshold is changed by this experiment.
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
from .train_multirate_fuzz import SCORED_FRAMES, render_training, sha256
from .widen_multirate_fuzz import WIDE_GEOMETRY


STEPS, CALIBRATION_INTERVAL = 8000, 1000
SOURCE = Path("remix/runs/dfz-wide-preflight-phase11/candidate.pt")


def verify_candidate_runtime(evidence):
    """No source-equivalence claim: compare each backend to THIS candidate."""
    cases, probes = evidence["real_audio_cases"], evidence["dynamic_probes"]
    if len(cases) != 54 or len(probes) != 3:
        raise ValueError("complete real54 and3 dynamic probes required")
    if not all(max(row["parity_max"], row["onnx_stream_max"]) <= 2e-6 for row in cases):
        raise ValueError("candidate backend/stream parity failed")
    if not all(max(row["parity_max"], row["onnx_stream_max"], row["cpu_stream_max"]) <= 2e-6
               and row["phase_exact"] and row["inputs_unchanged"] for row in probes):
        raise ValueError("candidate dynamic runtime failed")
    if not all(row["zero_peak"] == row["expired_tail_peak"] == row["future_prefix_error"] == 0.
               and row["quiet_peak"] <= 1e-3 for row in evidence["safety"].values()):
        raise ValueError("candidate safety failed")
    if not all(max(values) < 1 for backend in evidence["single_thread_rtf_including_validation"].values() for values in backend.values()):
        raise ValueError("candidate runtime cost failed")


def learning_rate(step):
    if not 1 <= step <= STEPS:
        raise ValueError("step outside fixed8000-update phase")
    if step <= 250:
        return 1e-4 + (1e-3 - 1e-4) * step / 250
    if step <= 4000:
        return 1e-3
    return 3e-5 + .5 * (1e-3 - 3e-5) * (1 + math.cos(math.pi * (step - 4000) / 4000))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new experiment directory required")
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("offline Apple GPU required")
    parent_path = SOURCE.with_name("metrics.json")
    parent = json.loads(parent_path.read_text())
    source_hash, parent_hash = sha256(SOURCE), sha256(parent_path)
    if (not parent.get("source_and_audio_reverified") or parent.get("checkpoint_sha256") != source_hash
            or parent.get("purpose") != "function-preserving-widen-preflight" or parent.get("geometry") != WIDE_GEOMETRY):
        raise ValueError("completed trained-source widening audit required")
    verify_candidate_runtime(parent)
    # Preserve and disclose the failed cross-model initialization test; it is
    # not a permit to admit an unfaithful model or relax within-model parity.
    equivalence_max = max(row["narrow_wide_max"] for row in parent["real_audio_cases"])
    names = (Path(__file__).name, "train_multirate_fuzz.py", "train_dfz_signed_tcn.py",
             "train_dfz_transient_narx.py", "train_dfz_long_tcn.py", "train_dfz_core_tcn.py")
    for name, digest in parent["source_sha256"].items():
        if sha256(Path(__file__).parent / name) != digest:
            raise ValueError("initialization/audit source changed")
    sources = {**parent["source_sha256"], **{name: sha256(Path(__file__).parent / name) for name in names}}
    provenance = parent["audio_provenance"]
    if set(provenance) != {"fit", "calibration"} or len(provenance["fit"]) != 234 or len(provenance["calibration"]) != 54:
        raise ValueError("same frozen234/54 partition required")

    def unchanged():
        if sha256(SOURCE) != source_hash or sha256(parent_path) != parent_hash:
            raise ValueError("initialization source/report changed")
        if sha256(SOURCE.with_name("model.onnx")) != parent["graph_sha256"]:
            raise ValueError("audited initialization graph changed")
        if any(sha256(Path(__file__).parent / name) != digest for name, digest in sources.items()):
            raise ValueError("experiment source changed")
        if any(sha256(row["path"]) != row["sha256"] for rows in provenance.values() for row in rows):
            raise ValueError("original audio changed")

    unchanged()
    torch.manual_seed(995)
    rng = np.random.default_rng(995)
    payload = torch.load(SOURCE, map_location="cpu", weights_only=True)
    if payload["geometry"] != WIDE_GEOMETRY or payload["architecture"] != "causal-multirate-gcn":
        raise ValueError("same preflighted wide architecture required")
    model = MultirateFuzz(**WIDE_GEOMETRY)
    model.load_state_dict(payload["state_dict"])
    model.to("mps")
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
        raise ValueError("balanced nine corners required")
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
                    "geometry": WIDE_GEOMETRY, "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}}, checkpoint)

    def progress():
        return {"schema": 1, "experiment": "wide-independent-core-fixed8000", "seed": 995,
                "requested_steps": STEPS, "completed_steps": updates[-1]["step"] if updates else 0,
                "calibration_interval": CALIBRATION_INTERVAL, "history": history, "updates": updates, "best_step": best_step,
                "parent_checkpoint_sha256": source_hash, "parent_report_sha256": parent_hash,
                "initialization": "approximate float32 channel-widening warmstart, not an equivalent replacement",
                "source_equivalence_max": equivalence_max, "source_equivalence_gate_passed": equivalence_max <= 2e-6,
                "preflight_overall_passed": parent["runtime_passed"], "candidate_own_runtime_checks_passed": True,
                "geometry": WIDE_GEOMETRY, "parameters": sum(value.numel() for value in model.parameters()),
                "source_sha256": sources, "audio_provenance": provenance, "optimizer_resume": False,
                "loss": "original signed weighted waveform+.1preemphasis; no worst16 loss",
                "learning_rate_schedule": "250-step1e-4->1e-3 ramp, constant through4000, cosine to3e-5 at8000",
                "sampling": "unchanged50/50active-uniform/Wet-peak-local4096, balanced26-file corner cycle",
                "corner_coverage": cycle.coverage(), "full_current_lowrate_prefix": True, "real_fast_context_frames": 2046,
                "fit_examples": 234, "calibration_examples": 54, "trainable_state_cached": False,
                "physical_audio_devices_used": False, "official_eval_opened": False, "source_audio_modified": False,
                "automatic_gain_or_normalization": False, "automatic_limiting": False, "admitted": False,
                "elapsed_seconds": time.perf_counter() - started}

    save()
    print(json.dumps({"calibration": history[0], "parameters": progress()["parameters"]}), flush=True)
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
            raise ValueError("nonfinite wide training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            updates.append({"step": step, "loss": float(loss.detach()), "learning_rate": learning_rate(step),
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
