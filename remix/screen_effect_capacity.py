#!/usr/bin/env python3
"""Train-side screening of a declared six-checkpoint DFZ architecture pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import torch

from .asrnn_effects import effect_files
from .fit_asrnn_effect_output import _partition
from .import_asrnn_effect import import_checkpoint
from .screen_asrnn_capacity import selection_key
from .stable_effect import load_stable_effect
from .train_asrnn_phase7 import _evaluate, _rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=Path("data/downloads/asrnn-results-20406285.zip"))
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/asrnn-physical-effects"))
    parser.add_argument("--architecture", choices=("4-64", "1-64", "4-8"), default="4-64")
    parser.add_argument("--compute", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--regularization", choices=("inf", "2"), default="inf")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.output.exists():
        raise ValueError("capacity output already exists")
    fit, calibration = _partition(effect_files(args.corpus, "dfz", "train"))
    rows = _rows(calibration, "dfz")
    report = {"schema": 1, "phase": "dfz-capacity", "device": "dfz", "screen_complete": False,
              "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
              "fit_files": [p.name for p in fit], "calibration_files": [p.name for p in calibration],
              "calibration_caveat": "public pretrained official-train selection, not an independent holdout; worst_attack metric here means first control Blend",
              "physical_audio_devices_used": False, "source_audio_modified": False,
              "architecture_pool": args.architecture, "compute": args.compute,
              "regularization": args.regularization,
              "formal_infinity_norm_stability": args.regularization == "inf", "results": []}
    args.output.mkdir(parents=True)
    with zipfile.ZipFile(args.archive) as archive:
        if archive.testzip() is not None:
            raise ValueError("archive CRC failed")
        names = sorted(n for n in archive.namelist() if f"results/long/dfz/checkpoints/StableLSTM_{args.regularization}-{args.architecture}-" in n and n.endswith(".pt"))
        if len(names) != 6:
            raise ValueError("expected the six pretrained DFZ candidates")
        for index, name in enumerate(names):
            print(json.dumps({"screening": name, "candidate": index+1, "total": len(names)}), flush=True)
            with tempfile.TemporaryDirectory(prefix="muspector-dfz-capacity-") as temporary:
                path = Path(temporary)
                source = Path(archive.extract(name, path))
                converted = path / "converted.pt"
                provenance = import_checkpoint(source, converted, "dfz", allow_spectral_regularization=args.regularization == "2")
                model, _ = load_stable_effect(converted)
                model = model.to(args.compute)
                metrics = _evaluate(model, rows, torch.device(args.compute), 16)
                row = {"member": name, "checkpoint_sha256": provenance["output_sha256"],
                       "source_sha256": provenance["source_sha256"], "calibration": metrics,
                       "import_runtime": provenance, "parameters": sum(p.numel() for p in model.parameters())}
                report["results"].append(row)
                if "selected" not in report or selection_key(metrics) < selection_key(report["selected"]["calibration"]):
                    report["selected"] = row
                    shutil.copyfile(converted, args.output / "selected.pt")
                (args.output / "screen.json").write_text(json.dumps(report, indent=2)+"\n")
                print(json.dumps({"candidate": index+1, "calibration": metrics}), flush=True)
    report["screen_complete"] = True
    (args.output / "screen.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
