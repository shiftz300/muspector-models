"""Create lossless phrase demos for both supported forward orders."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile
from scipy.signal import resample_poly

from .data import discover_order_records
from .portable import PortableRuntime
from .restore import plan
from .stages import metric


RATE = 48_000
ORDERS = (("reverb", "drive"), ("drive", "reverb"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> tuple[np.ndarray, int]:
    value, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return value.mean(1), int(rate)


def convert(value: np.ndarray, rate: int) -> np.ndarray:
    if rate == RATE:
        return np.asarray(value, dtype=np.float32)
    common = __import__("math").gcd(rate, RATE)
    return resample_poly(value, RATE // common, rate // common).astype(np.float32)


def phrase(audio: np.ndarray, seconds: float) -> tuple[int, int]:
    length = min(len(audio), round(seconds * RATE))
    if length == len(audio):
        return 0, length
    hop = RATE // 4
    best = (-float("inf"), 0)
    for start in range(0, len(audio) - length + 1, hop):
        value = audio[start : start + length]
        frame = max(1, RATE // 50)
        envelope = np.sqrt(np.convolve(value * value, np.ones(frame) / frame, mode="valid")[::frame] + 1.0e-12)
        score = float(np.mean(envelope) + 0.25 * np.std(envelope) + 0.05 * np.maximum(np.diff(envelope), 0.0).sum())
        if score > best[0]:
            best = score, start
    return best[1], best[1] + length


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/guitar-effects-chains"))
    parser.add_argument("--model", type=Path, default=Path("runs/stage/model2"))
    parser.add_argument("--output", type=Path, default=Path("runs/stage/model2/demo"))
    parser.add_argument("--group", default="tele_neck_pick11")
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()
    records = [row for row in discover_order_records(args.corpus) if row.split == "test" and row.group == args.group and row.order in ORDERS]
    if not records or {row.order for row in records} != set(ORDERS):
        raise RuntimeError(f"group {args.group} does not contain both supported orders")
    drive = PortableRuntime(args.model / "drive.pt", args.model / "drive.onnx", core=16_384, halo=4_096, threads=2)
    reverb = PortableRuntime(args.model / "reverb.pt", args.model / "reverb.onnx", core=16_384, halo=12_288, threads=2)
    bank = {"drive": drive, "reverb": reverb}
    candidates = {}
    clean_full = None
    for row in records:
        clean, clean_rate = read(row.dry); wet, wet_rate = read(row.wet)
        clean, wet = convert(clean, clean_rate), convert(wet, wet_rate)
        length = min(len(clean), len(wet)); clean, wet = clean[:length], wet[:length]
        value = wet.copy()
        for kind in plan(row.order):
            value = bank[kind].render(value, 1.0)
        report = metric([wet], [value], [clean])
        score = report["aligned_esr_improvement"] + 0.05 * report["sidr_improvement_db"]
        item = {"record": row, "clean": clean, "wet": wet, "restored": value, "metrics": report, "score": score}
        if row.order not in candidates or score > candidates[row.order]["score"]:
            candidates[row.order] = item
        clean_full = clean
    start, end = phrase(clean_full, args.seconds)
    args.output.mkdir(parents=True, exist_ok=True)
    clean_path = args.output / "clean.wav"
    soundfile.write(clean_path, clean_full[start:end], RATE, subtype="FLOAT")
    documents = {}
    silence = np.zeros(round(0.75 * RATE), dtype=np.float32)
    sequence = [clean_full[start:end], silence]
    for order in ORDERS:
        item = candidates[order]; name = "rd" if order == ("reverb", "drive") else "dr"
        wet, restored = item["wet"][start:end], item["restored"][start:end]
        wet_path, restored_path, ab_path = args.output / f"{name}-wet.wav", args.output / f"{name}-restored.wav", args.output / f"{name}-ab.wav"
        soundfile.write(wet_path, wet, RATE, subtype="FLOAT"); soundfile.write(restored_path, restored, RATE, subtype="FLOAT")
        soundfile.write(ab_path, np.concatenate((wet, silence, restored)), RATE, subtype="FLOAT")
        sequence.extend((wet, silence, restored, silence))
        documents[name] = {"forward": list(order), "restore": list(plan(order)), "source": str(item["record"].wet), "metrics": item["metrics"], "files": {"wet": wet_path.name, "restored": restored_path.name, "ab": ab_path.name}}
    show_path = args.output / "show.wav"; soundfile.write(show_path, np.concatenate(sequence[:-1]), RATE, subtype="FLOAT")
    result = {"schema": 1, "group": args.group, "description": "picked Telecaster phrase; show order is Clean, Reverb-Drive Wet, Restored, Drive-Reverb Wet, Restored", "sample_rate": RATE, "subtype": "FLOAT", "crop_seconds": [start / RATE, end / RATE], "automatic_normalization": False, "automatic_limiting": False, "lossy_reencoding": False, "physical_audio_devices_used": False, "files": {"clean": clean_path.name, "show": show_path.name}, "orders": documents}
    for path in args.output.glob("*.wav"):
        result.setdefault("sha256", {})[path.name] = sha256(path)
    (args.output / "demo.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
