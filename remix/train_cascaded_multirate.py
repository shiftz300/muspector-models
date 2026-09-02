"""Fresh, fixed20000 offline updates for a separate two-array causal model.

No teacher, old hidden states, eval-set listing, audio gain or device access.
Loss denominators are pooled complete-fit Wet energies, not crop energies.
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
from .cascaded_candidate import model_from_payload
from .cascaded_multirate_fuzz import ARCHITECTURE, GEOMETRY, CascadedMultirateFuzz
from .precheck_multirate_fuzz import sha256
from .train_asrnn_phase7 import _evaluate
from .train_dfz_core_tcn import selection_key
from .train_dfz_signed_tcn import active_window_ranges, uniform_active_start
from .train_dfz_transient_narx import CornerCycle
from .train_multirate_fixed_energy import fixed_energy_loss
from .train_multirate_fuzz import SCORED_FRAMES, render_training


STEPS, INTERVAL, SEED = 20000, 1000, 1012
PRECHECK = Path("remix/runs/dfz-cascaded-precheck-phase11/metrics.json")
PARTITION = Path("remix/runs/dfz-centered-core-phase11/metrics.json")


def learning_rate(step):
    if not 1 <= step <= STEPS:
        raise ValueError("step outside fixed20000 phase")
    if step <= 500:
        return 1e-4+9e-4*step/500
    if step <= 10000:
        return 1e-3
    return 3e-5+.5*(1e-3-3e-5)*(1+math.cos(math.pi*(step-10000)/10000))


def pooled_loss(prediction, expected, peaks, energy, pre_energy):
    if not math.isfinite(energy) or not math.isfinite(pre_energy) or min(energy, pre_energy) <= 0:
        raise ValueError("positive finite pooled fit-only energy required")
    return fixed_energy_loss(prediction, expected, peaks,
                             prediction.new_full((prediction.shape[0],), energy),
                             prediction.new_full((prediction.shape[0],), pre_energy))


def smoke():
    torch.manual_seed(SEED)
    model = CascadedMultirateFuzz(**GEOMETRY).to("mps")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    rows = []
    for a in (0., .5, 1.):
        for b in (0., .5, 1.):
            dry = torch.randn(144000)*.03
            wet = torch.tanh(dry*7)*.8
            rows.append({"dry": dry, "wet": wet, "controls": torch.tensor((a, b)), "file_peak": wet.abs().amax()})
    measurements = []
    for step in range(8):
        torch.mps.synchronize()
        tick = time.perf_counter()
        predicted, expected, peaks = render_training(model, rows, [50000+64*i for i in range(9)])
        loss = pooled_loss(predicted, expected, peaks, .03, .04).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite smoke loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        torch.mps.synchronize()
        measurements.append({"step": step, "seconds": time.perf_counter()-tick, "loss": float(loss.detach()),
                             "gradient_norm": float(norm), "driver_bytes": torch.mps.driver_allocated_memory()})
    with torch.inference_mode():
        dry, controls = torch.randn(2, 6173, device="mps")*.03, torch.rand(2, 6173, 2, device="mps")
        whole, _ = model(dry, controls)
        a, state = model(dry[:, :17], controls[:, :17])
        b, _ = model(dry[:, 17:], controls[:, 17:], state)
        stream = float((whole-torch.cat((a, b), 1)).abs().max())
        zero = float(model(torch.zeros_like(dry), controls)[0].abs().max())
    passed = stream <= 2e-6 and zero == 0 and max(row["driver_bytes"] for row in measurements) < 8000000000
    if not passed:
        raise ValueError("MPS stream/zero/resource smoke failed")
    return {"passed": passed, "measurements": measurements, "mps_stream_max_error": stream, "zero_peak": zero,
            "architecture": ARCHITECTURE, "source_audio_loaded": False, "physical_audio_devices_used": False,
            "admitted": False, "source_sha256": {Path(__file__).name: sha256(__file__)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("offline MPS compute required")
    if args.smoke_only:
        result = smoke()
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output/"metrics.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)
        return
    if args.output is None or args.output.exists():
        raise ValueError("new experiment output directory required")
    precheck = json.loads(PRECHECK.read_text())
    if (not precheck["passed"] or precheck["trained"] or precheck["architecture"] != ARCHITECTURE
            or precheck["geometry"] != GEOMETRY or sha256(PRECHECK.parent/"model.onnx") != precheck["graph_sha256"]):
        raise ValueError("same-architecture synthetic precheck required")
    names = (Path(__file__).name, "cascaded_candidate.py", "train_multirate_fuzz.py", "train_multirate_fixed_energy.py",
             "train_dfz_signed_tcn.py", "train_dfz_long_tcn.py", "train_dfz_core_tcn.py", "train_dfz_transient_narx.py",
             "train_asrnn_phase7.py", "asrnn_effects.py", "asrnn_data.py")
    sources = {**precheck["source_sha256"], **{name: sha256(Path(__file__).parent/name) for name in names}}
    precheck_hash, partition_hash = sha256(PRECHECK), sha256(PARTITION)
    provenance = json.loads(PARTITION.read_text())["audio_provenance"]
    if set(provenance) != {"fit", "calibration"} or len(provenance["fit"]) != 234 or len(provenance["calibration"]) != 54:
        raise ValueError("same original fit234/cal54 manifest required")

    def unchanged():
        if sha256(PRECHECK) != precheck_hash or sha256(PARTITION) != partition_hash:
            raise ValueError("precheck or fixed partition evidence changed")
        if any(sha256(Path(__file__).parent/name) != digest for name, digest in sources.items()):
            raise ValueError("training source changed")
        if any(sha256(row["path"]) != row["sha256"] for rows in provenance.values() for row in rows):
            raise ValueError("original training/calibration audio changed")

    unchanged()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    corners, calibration, energies, pre_energies = [[] for _ in range(9)], [], [], []
    for split in ("fit", "calibration"):
        for record in provenance[split]:
            path = Path(record["path"])
            if "train" not in path.parts or (int(path.stem.split(",")[-1])%5 == 0) != (split == "calibration"):
                raise ValueError("frozen training take partition differs")
            dry, wet, controls = read_effect_pair(path, "dfz")
            if len(dry) != 144000:
                raise ValueError("full three-second original recording required")
            row = {"dry": torch.from_numpy(dry), "wet": torch.from_numpy(wet), "controls": torch.from_numpy(controls),
                   "attack": int(path.stem.split(",")[0])}
            if split == "fit":
                body = row["wet"][1024:]
                energies.append(float(body.square().mean()))
                pre_energies.append(float((body[1:]-.95*body[:-1]).square().mean()))
                row.update(file_peak=body.abs().amax(), peak_index=int(body.abs().argmax())+1024,
                           active_ranges=active_window_ranges(row["dry"]))
                a, b = (row["controls"]*2).long().tolist()
                corners[a*3+b].append(row)
            else:
                calibration.append(row)
    if any(len(rows) != 26 for rows in corners):
        raise ValueError("balanced nine-corner fit partition required")
    energy, pre_energy = float(np.mean(energies)), float(np.mean(pre_energies))
    model = CascadedMultirateFuzz(**GEOMETRY).to("mps")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate(1), weight_decay=1e-5)
    cycle = CornerCycle(rng)
    started = time.perf_counter()
    history, updates = [{"step": 0, **_evaluate(model, calibration, torch.device("mps"), 9)}], []
    best_step, best_key = 0, selection_key(history[0])
    args.output.mkdir(parents=True)
    checkpoint = args.output/"candidate.pt"

    def save():
        torch.save({"experimental_schema": 1, "architecture": ARCHITECTURE, "device": "dfz", "sample_rate": 48000,
                    "geometry": GEOMETRY, "state_dict": {name: value.detach().cpu().clone()
                                                        for name, value in model.state_dict().items()}}, checkpoint)

    def progress():
        return {"schema": 1, "experiment": "fresh-cascaded-raw-mixin-skip20000", "seed": SEED,
                "architecture": ARCHITECTURE, "geometry": GEOMETRY, "requested_steps": STEPS,
                "completed_steps": updates[-1]["step"] if updates else 0, "calibration_interval": INTERVAL,
                "history": history, "updates": updates, "best_step": best_step, "source_sha256": sources,
                "precheck_sha256": precheck_hash, "partition_manifest_sha256": partition_hash,
                "audio_provenance": provenance, "fit_examples": 234, "calibration_examples": 54,
                "fit_only_pooled_energy": energy, "fit_only_pooled_pre95_energy": pre_energy,
                "loss": "signed weighted waveform MSE +.1pre95 MSE; fixed complete-fit pooled energy denominators",
                "learning_rate_schedule": "500-step1e-4 to1e-3 warmup; constant through10000; cosine to3e-5 at20000",
                "sampling": "50/50active-uniform/Wet-peak-local4096; 9corners; balanced26-file cycle",
                "corner_coverage": cycle.coverage(), "full_current_lowrate_prefix": True, "real_fast_context_frames": 4092,
                "trainable_state_cached": False, "teacher_lstm_used": False, "pretrained_weights_used": False,
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
        prediction, expected, peaks = render_training(model, chosen, starts)
        loss = pooled_loss(prediction, expected, peaks, energy, pre_energy).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite cascaded loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step%200 == 0:
            updates.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm),
                            "learning_rate": learning_rate(step), "elapsed_seconds": time.perf_counter()-started,
                            "driver_bytes": torch.mps.driver_allocated_memory()})
            if updates[-1]["driver_bytes"] > 8000000000:
                raise ValueError("fixed8GB GPU budget exceeded")
            (args.output/"training.json").write_text(json.dumps(progress(), indent=2, allow_nan=False)+"\n")
            print(json.dumps(updates[-1]), flush=True)
        if step%INTERVAL == 0:
            history.append({"step": step, **_evaluate(model, calibration, torch.device("mps"), 9)})
            if selection_key(history[-1]) < best_key:
                best_step, best_key = step, selection_key(history[-1])
                save()
            (args.output/"training.json").write_text(json.dumps(progress(), indent=2, allow_nan=False)+"\n")
            print(json.dumps({"calibration": history[-1], "best_step": best_step}), flush=True)
    result = progress()
    selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
    del model, optimizer, corners, chosen
    torch.mps.empty_cache()
    model = model_from_payload(selected)
    unchanged()
    print(json.dumps({"stage": "complete-selected-cpu54", "best_step": best_step}), flush=True)
    result["calibration"] = _evaluate(model, calibration, torch.device("cpu"), 9)
    unchanged()
    result.update(checkpoint_sha256=sha256(checkpoint), source_and_audio_reverified=True,
                  passes_full_calibration_gate=result["calibration"]["passes_selection_gate"],
                  status="passed-calibration-only" if result["calibration"]["passes_selection_gate"] else "rejected-full-calibration",
                  elapsed_seconds=time.perf_counter()-started)
    (args.output/"metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"], "calibration": result["calibration"]}), flush=True)


if __name__ == "__main__":
    main()
