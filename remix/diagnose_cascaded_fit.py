"""Fixed1000 paired coordinate diagnostic on ONE original fit file only.

Identical seeded fresh weights/windows/LR; native and derivative-coordinate
optimization compared in sample. This cannot admit a model or show generalization.
No candidate weights/audio/feature caches are saved. Main training is untouched.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .asrnn_effects import read_effect_pair
from .cascaded_candidate import model_from_payload
from .cascaded_multirate_fuzz import ARCHITECTURE, GEOMETRY, CascadedMultirateFuzz
from .cascaded_temporal_basis import attach, materialize
from .precheck_multirate_fuzz import sha256
from .train_asrnn_phase7 import _evaluate
from .train_cascaded_multirate import pooled_loss
from .train_dfz_signed_tcn import active_window_ranges, uniform_active_start
from .train_multirate_fuzz import render_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new fit-only diagnostic report required")
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("offline MPS compute required")
    manifest = Path("remix/runs/dfz-centered-core-phase11/metrics.json")
    provenance = json.loads(manifest.read_text())["audio_provenance"]
    records = [row for row in provenance["fit"] if Path(row["path"]).name == "0,0,1.wav"]
    if len(records) != 1:
        raise ValueError("fixed training take required")
    record = records[0]
    path = Path(record["path"])
    if "train" not in path.parts or sha256(path) != record["sha256"]:
        raise ValueError("original fit audio differs")
    x, y, controls = read_effect_pair(path, "dfz")
    row = {"dry": torch.from_numpy(x), "wet": torch.from_numpy(y), "controls": torch.from_numpy(controls), "attack": 0}
    body = row["wet"][1024:]
    row["file_peak"] = body.abs().amax()
    peak = int(body.abs().argmax())+1024
    ranges = active_window_ranges(row["dry"])
    energy, pre_energy = float(body.square().mean()), float((body[1:]-.95*body[:-1]).square().mean())
    rng = np.random.default_rng(1016)
    starts = [uniform_active_start(rng, ranges) if rng.random() < .5 else
              max(1024, min(len(x)-4096, peak-int(rng.integers(512, 3584)))) for _ in range(1000)]
    sources = {name: sha256(Path(__file__).parent/name) for name in
               (Path(__file__).name, "cascaded_temporal_basis.py", "dfz_temporal_basis.py", "cascaded_multirate_fuzz.py",
                "multirate_fuzz.py", "train_multirate_fuzz.py", "train_cascaded_multirate.py", "train_asrnn_phase7.py",
                "train_multirate_fixed_energy.py", "train_dfz_signed_tcn.py", "asrnn_effects.py", "cascaded_candidate.py")}
    results = []
    for coordinates in ("native", "derivative-basis"):
        torch.manual_seed(1016)
        model = CascadedMultirateFuzz(**GEOMETRY).to("mps")
        if coordinates != "native":
            before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            attach(model)
            initial = materialize(model)
            if any(not torch.equal(value, initial[name]) for name, value in before.items()):
                raise ValueError("initial coordinates change weights")
            del before, initial
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        history, maximum_driver = [], 0
        started = time.perf_counter()
        for step, start in enumerate(starts, 1):
            model.train()
            a, b, p = render_training(model, [row], [start])
            loss = pooled_loss(a, b, p, energy, pre_energy).mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite fit-only diagnostic loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            if step%250 == 0:
                maximum_driver = max(maximum_driver, torch.mps.driver_allocated_memory())
                if maximum_driver > 2000000000:
                    raise ValueError("diagnostic exceeds separate2GB driver budget")
                history.append({"step": step, **_evaluate(model, [row], torch.device("mps"), 1)})
                print(json.dumps({"coordinates": coordinates, "fit_only": history[-1]}), flush=True)
        state = materialize(model) if coordinates != "native" else {name: value.detach().cpu().clone()
                                                                   for name, value in model.state_dict().items()}
        del model, optimizer, a, b, p, loss
        torch.mps.empty_cache()
        model = model_from_payload({"experimental_schema": 1, "architecture": ARCHITECTURE, "geometry": GEOMETRY,
                                    "device": "dfz", "sample_rate": 48000, "state_dict": state})
        results.append({"coordinates": coordinates, "history": history,
                        "final_cpu_in_sample": _evaluate(model, [row], torch.device("cpu"), 1),
                        "maximum_training_driver_bytes": maximum_driver, "elapsed_seconds": time.perf_counter()-started})
        del model, state
    if sha256(path) != record["sha256"] or any(sha256(Path(__file__).parent/name) != digest for name, digest in sources.items()):
        raise ValueError("source audio or diagnostic implementation changed")
    result = {"purpose": "in-sample single-fit-file coordinate comparison; NOT quality admission or generalization",
              "seed": 1016, "source_record": record, "source_sha256": sources, "steps_per_variant": 1000,
              "results": results, "source_audio_reverified": True, "calibration_opened": False, "official_eval_opened": False,
              "physical_audio_devices_used": False, "audio_or_weights_saved": False, "admitted": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
