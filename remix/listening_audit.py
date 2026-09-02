"""Superseding temporal and human-listening audit for the rejected 1.0 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile

from .real_pairs import chain
from .real_runtime import Runtime
from .reverb_quality import measure, summarize
from .stages import CORPUS


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "runs/stage/model20"
DEMO = MODEL / "demo"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--demo", type=Path, default=DEMO)
    parser.add_argument("--output", type=Path, default=MODEL / "listen.json")
    parser.add_argument("--frames", type=int, default=131072)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace listening audit {args.output}")
    runtime = Runtime(args.model / "drive.npz", args.model / "reverb.npz")
    rows = chain(args.corpus, "test", args.frames, 20261270)
    predictions = [runtime.run(row["source"], row["order"]) for row in rows]
    temporal = summarize([row["source"] for row in rows], predictions, [row["target"] for row in rows])
    demo = {}
    clean, _ = soundfile.read(args.demo / "clean.wav", dtype="float32")
    for name in ("dr", "rd"):
        wet, _ = soundfile.read(args.demo / f"{name}-wet.wav", dtype="float32")
        restored, _ = soundfile.read(args.demo / f"{name}-restored.wav", dtype="float32")
        demo[name] = measure(wet, restored, clean)
    result = {"schema": 1, "status": "rejected-listening", "accepted": False, "supersedes": "seal.json", "reason": "user listening veto: restoration is incomplete and can increase perceived Reverb", "temporal": temporal, "demo": demo, "human": {"required": True, "approved": False, "veto": True}, "metric_correction": {"old_tail_name": "restored_aligned_esr_p95 non-regression", "old_tail_was_reverb_specific": False, "new_gates": ["tail excess reduction", "tail envelope ESR improvement", "added Reverb fraction", "human listening approval"]}, "physical_audio_devices_used": False, "source_audio_modified": False}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
