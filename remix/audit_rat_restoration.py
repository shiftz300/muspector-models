"""Audit an existing real ProCo RAT inverse with perceptual anti-blur gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .asrnn_data import LICENSE, rat_files, read_rat_pair
from .net import SpectralNet
from .restoration_quality import measure, summarize


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> SpectralNet:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("architecture") != "complex-stft" or payload.get("device") not in {"rat", "rat-mixed"}:
        raise ValueError("perceptual RAT audit requires a complex-STFT RAT checkpoint")
    model = SpectralNet(payload["channels"], payload["n_fft"], payload["hop"])
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/clean/model11/clean.pt")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/corpus/asrnn-physical-effects")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/clean/model11/listen.json")
    parser.add_argument("--strength", type=float, default=0.875)
    parser.add_argument("--minimum-ratio", type=float, default=0.03)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace RAT audit {args.output}")
    torch.set_num_threads(2)
    model = _load(args.checkpoint)
    wet_rows, restored_rows, clean_rows, details = [], [], [], []
    with torch.inference_mode():
        for index, path in enumerate(rat_files(args.corpus, "eval")):
            clean, wet, controls = read_rat_pair(path)
            clean_rms = float(np.sqrt(np.mean(np.square(clean[1024:], dtype=np.float64))))
            wet_rms = float(np.sqrt(np.mean(np.square(wet[1024:], dtype=np.float64))))
            if wet_rms / max(clean_rms, 1.0e-12) < args.minimum_ratio:
                continue
            restored = model(torch.from_numpy(wet).unsqueeze(0), strength=args.strength)[0].numpy()
            row = measure(wet, restored, clean)
            details.append({"file": path.name, "controls": controls.tolist(), **row})
            wet_rows.append(wet); restored_rows.append(restored); clean_rows.append(clean)
            if len(details) % 16 == 0:
                print(json.dumps({"stage": "rat-perceptual-audit", "completed": len(details)}), flush=True)
    automatic = summarize(wet_rows, restored_rows, clean_rows)
    result = {
        "schema": 1,
        "status": "audition-required" if automatic["accepted"] else "rejected-perceptual",
        "accepted": False,
        "automatic": automatic,
        "human": {"required": True, "approved": False, "veto": False},
        "model": {
            "device": "ProCo RAT",
            "checkpoint": str(args.checkpoint),
            "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "strength": args.strength,
        },
        "data": {
            "source": "ASRNN real ProCo RAT official eval",
            "license": LICENSE,
            "scope": "internal non-commercial development; no independent locked final split",
            "eligible": len(details),
            "minimum_wet_to_clean_rms_ratio": args.minimum_ratio,
            "source_read_only": True,
        },
        "metrics": {
            "purpose": ["reject blur", "reject residual drive", "reject lost pick transients", "reject new clipping"],
            "waveform_esr_is_sufficient": False,
            "threshold_state": "provisional rejection-only; promotion still requires listening approval",
        },
        "rows": details,
        "physical_audio_devices_used": False,
        "source_audio_modified": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "accepted", "automatic")}, sort_keys=True))


if __name__ == "__main__":
    main()
