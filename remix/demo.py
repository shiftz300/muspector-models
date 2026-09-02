"""Create a lossless phrase demo from the accepted locked-final runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile
from scipy.signal import resample_poly

from .data import _waveform, discover_order_records
from .real_runtime import Runtime
from .show import phrase
from .stages import metric


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "runs/stage/model20"
CORPUS = ROOT / "data/corpus/guitar-effects-chains"
RUN = ROOT / "runs/stage/model20/demo"
RATE = 48000
ORDERS = (("drive", "reverb"), ("reverb", "drive"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(value: np.ndarray) -> np.ndarray:
    return resample_poly(value, 160, 147).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--group", default="tele_neck_pick11")
    parser.add_argument("--seconds", type=float, default=2.5)
    args = parser.parse_args()
    if args.output.exists() and (args.output / "demo.json").exists():
        raise FileExistsError(f"refusing to replace demo {args.output}")
    records = [row for row in discover_order_records(args.corpus) if row.split == "test" and row.group == args.group and row.order in ORDERS]
    if {row.order for row in records} != set(ORDERS):
        raise RuntimeError(f"group {args.group} does not contain both supported orders")
    clean = convert(_waveform(records[0].dry))
    wet = {id(row): convert(_waveform(row.wet)) for row in records}
    length = min([len(clean), *[len(value) for value in wet.values()]])
    clean = clean[:length]
    wet = {key: value[:length] for key, value in wet.items()}
    maximum = max(float(np.max(np.abs(clean))), *[float(np.max(np.abs(value))) for value in wet.values()], 1.0e-6)
    gain = 0.22 / maximum
    clean = np.asarray(clean * gain, dtype=np.float32)
    wet = {key: np.asarray(value * gain, dtype=np.float32) for key, value in wet.items()}
    runtime = Runtime(args.model / "drive.npz", args.model / "reverb.npz")
    selected = {}
    for row in records:
        source = wet[id(row)]
        restored = runtime.run(source, row.order)
        report = metric([source], [restored], [clean])
        score = report["aligned_esr_improvement"] + 0.10 * report["sidr_improvement_db"] + 0.25 * report["closure"]
        candidate = {"record": row, "wet": source, "restored": restored, "report": report, "score": score}
        if row.order not in selected or score > selected[row.order]["score"]:
            selected[row.order] = candidate
    start, end = phrase(clean, args.seconds)
    args.output.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(round(0.6 * RATE), dtype=np.float32)
    clean_path = args.output / "clean.wav"
    soundfile.write(clean_path, clean[start:end], RATE, subtype="FLOAT")
    sequence = [clean[start:end], silence]
    documents = {}
    for order in ORDERS:
        item = selected[order]
        name = "dr" if order == ("drive", "reverb") else "rd"
        affected = item["wet"][start:end]
        restored = item["restored"][start:end]
        wet_path = args.output / f"{name}-wet.wav"
        restored_path = args.output / f"{name}-restored.wav"
        ab_path = args.output / f"{name}-ab.wav"
        soundfile.write(wet_path, affected, RATE, subtype="FLOAT")
        soundfile.write(restored_path, restored, RATE, subtype="FLOAT")
        soundfile.write(ab_path, np.concatenate((affected, silence, restored)), RATE, subtype="FLOAT")
        sequence.extend((affected, silence, restored, silence))
        documents[name] = {"forward": list(order), "restore": list(reversed(order)), "source": str(item["record"].wet), "metrics_full_record": item["report"], "files": {"wet": wet_path.name, "restored": restored_path.name, "ab": ab_path.name}}
    show_path = args.output / "show.wav"
    soundfile.write(show_path, np.concatenate(sequence[:-1]), RATE, subtype="FLOAT")
    result = {"schema": 1, "model": "../model.json", "group": args.group, "description": "Clean, Drive-Reverb Wet, Restored, Reverb-Drive Wet, Restored", "sample_rate": RATE, "subtype": "FLOAT", "crop_seconds": [start / RATE, end / RATE], "level_profile": {"id": "dafx-demo", "shared_fixed_gain": gain, "target_global_peak": 0.22, "applied_before_runtime": True, "runtime_automatic_normalization": False}, "automatic_limiting": False, "automatic_dither": False, "lossy_reencoding": False, "physical_audio_devices_used": False, "files": {"clean": clean_path.name, "show": show_path.name}, "orders": documents}
    result["sha256"] = {path.name: sha256(path) for path in sorted(args.output.glob("*.wav"))}
    (args.output / "demo.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
