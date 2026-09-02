"""Bounded synthetic larger-batch training benchmark; no corpus/device I/O.

Wall times are measurements on the current machine and may include contention
from a concurrent trainer. This chooses a resource budget, not model quality.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from .cascaded_multirate_fuzz import CascadedMultirateFuzz, GEOMETRY
from .precheck_multirate_fuzz import sha256
from .train_cascaded_multirate import pooled_loss
from .train_multirate_fuzz import render_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", choices=(9, 18, 36), default=(9, 18))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new benchmark report required")
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("offline MPS compute required")
    torch.manual_seed(1015)
    rows, results = [], []
    for index in range(36):
        dry = torch.randn(144000)*.03
        wet = torch.tanh(dry*7)*.8
        rows.append({"dry": dry, "wet": wet, "controls": torch.tensor([(index%9//3)/2, (index%3)/2]),
                     "file_peak": wet.abs().amax()})
    for batch in args.batches:
        model = CascadedMultirateFuzz(**GEOMETRY).to("mps")
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
        measurements = []
        for step in range(5):
            torch.mps.synchronize()
            started = time.perf_counter()
            a, b, p = render_training(model, rows[:batch], [50000+64*i for i in range(batch)])
            loss = pooled_loss(a, b, p, .03, .04).mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite synthetic loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            torch.mps.synchronize()
            measurements.append({"seconds": time.perf_counter()-started, "loss": float(loss.detach()),
                                 "gradient_norm": float(norm), "driver_bytes": torch.mps.driver_allocated_memory()})
            if measurements[-1]["driver_bytes"] > 4000000000:
                break
        steady = measurements[1:]
        result = {"batch": batch, "measurements": measurements,
                  "passed_memory_budget": max(row["driver_bytes"] for row in measurements) <= 4000000000,
                  "steady_seconds_per_update": sum(row["seconds"] for row in steady)/len(steady) if steady else None}
        result["scored_frames_per_second"] = (batch*4096/result["steady_seconds_per_update"] if steady else None)
        results.append(result)
        print(json.dumps(result), flush=True)
        del model, optimizer, a, b, p, loss
        torch.mps.empty_cache()
    result = {"purpose": "synthetic-training-throughput-only", "results": results,
              "concurrent_training_may_affect_wall_time": True, "source_audio_loaded": False,
              "physical_audio_devices_used": False, "trained_model_saved": False, "source_sha256": sha256(__file__)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
