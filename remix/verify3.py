#!/usr/bin/env python3
"""Third one-shot independent multi-author NAM A2 seal for router model7.

The seal evaluates public capture models against public DI files entirely in
memory.  It never accesses a physical audio device and retains no rendered
audio.  Its result is capture-model evidence, not raw hardware evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .a2 import A2Runtime, RATE, loader
from .identity import IdentityRuntime
from .packages import digest
from .router import RouterRuntime


NAM = {
    "repository": "https://github.com/sdatkinson/neural-amp-modeler",
    "version": "0.13.0",
    "commit": "f26112906de06ec6b796ad6d1982e29eed83144e",
    "tree": "f8cd0987ec2cff11d4b886b794f6672575dfeda8",
    "license": "MIT",
}
MODELS = (
    ("rat", "jhs10", "jhs10.nam", "https://api.tone3000.com/storage/v1/object/public/models/8a824315602ffa6c_a2.nam", "https://www.tone3000.com/tones/jhs-distortion-rat-1570", "T3K"),
    ("rat", "jhs5", "jhs5.nam", "https://api.tone3000.com/storage/v1/object/public/models/5a030e907195cc4b_a2.nam", "https://www.tone3000.com/tones/jhs-distortion-rat-1570", "T3K"),
    ("rat", "large1", "large1.nam", "https://api.tone3000.com/storage/v1/object/public/models/wcjfmk99z4_a2.nam", "https://www.tone3000.com/tones/proco-rat-large-box-reissue-2000s-46960", "T3K"),
    ("rat", "large2", "large2.nam", "https://api.tone3000.com/storage/v1/object/public/models/1n5hgcz1oxrj_a2.nam", "https://www.tone3000.com/tones/proco-rat-large-box-reissue-2000s-46960", "T3K"),
    ("other", "coron1", "coron1.nam", "https://api.tone3000.com/storage/v1/object/public/models/mmek8a3pj0q_a2.nam", "https://www.tone3000.com/tones/coron-distortion-32679", "T3K"),
    ("other", "coron2", "coron2.nam", "https://api.tone3000.com/storage/v1/object/public/models/s0dmm0o3ybb_a2.nam", "https://www.tone3000.com/tones/coron-distortion-32679", "T3K"),
    ("other", "dod1", "dod1.nam", "https://api.tone3000.com/storage/v1/object/public/models/bf2yu5qfats_a2.nam", "https://www.tone3000.com/tones/dod-fx53-american-metal-distortion-pedal-30779", "T3K"),
    ("other", "dod2", "dod2.nam", "https://api.tone3000.com/storage/v1/object/public/models/hvrfnumcg6_a2.nam", "https://www.tone3000.com/tones/dod-fx53-american-metal-distortion-pedal-30779", "T3K"),
)
CLIPS = (
    "data/corpus/guitarset/audio_mono-pickup_mix/00_Rock3-117-Bb_solo_mix.wav",
    "data/corpus/guitarset/audio_mono-pickup_mix/01_Jazz2-110-Bb_solo_mix.wav",
    "data/corpus/guitarset/audio_mono-pickup_mix/02_Funk1-114-Ab_solo_mix.wav",
    "data/corpus/guitarset/audio_mono-pickup_mix/03_Rock2-85-F_comp_mix.wav",
)
GATES = {
    "rat_recall": 0.75,
    "other_false_route": 0.10,
    "rat_models": 4,
    "other_models": 4,
    "automatic_deliveries": 0,
    "source_mutations": 0,
    "nonfinite_renders": 0,
}


def _array(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def clip(path: Path, seconds: int = 5) -> tuple[np.ndarray, dict]:
    source, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 44_100 or source.shape[1] != 1:
        raise ValueError(f"seal requires mono 44.1 kHz GuitarSet pickup mix: {path}")
    mono = resample_poly(source[:, 0], 160, 147).astype(np.float32)
    frames = seconds * RATE
    starts = list(range(0, len(mono) - frames + 1, RATE))
    if not starts:
        raise ValueError(f"seal DI is too short: {path}")
    start = min(starts, key=lambda item: (-float(np.mean(mono[item:item + frames] ** 2)), item))
    value = mono[start:start + frames].copy()
    return value, {
        "path": path.as_posix(), "file_sha256": digest(path), "source_rate": rate,
        "source_channels": 1, "channel_policy": "mono",
        "analysis_rate": RATE, "analysis_resampler": "scipy.signal.resample_poly/160:147",
        "analysis_start": start, "analysis_frames": frames, "analysis_sha256": _array(value),
        "analysis_peak": float(np.max(np.abs(value))),
        "analysis_rms": float(np.sqrt(np.mean(value * value))),
    }


def gate(rows: list[dict]) -> tuple[bool, list[str], dict]:
    positive = [row for row in rows if row["scope"] == "rat"]
    negative = [row for row in rows if row["scope"] == "other"]
    metrics = {
        "rat_examples": len(positive), "other_examples": len(negative),
        "rat_models": len({row["model"] for row in positive}),
        "other_models": len({row["model"] for row in negative}),
        "rat_recall": sum(row["routed"] for row in positive) / max(len(positive), 1),
        "other_false_route": sum(row["routed"] for row in negative) / max(len(negative), 1),
        "automatic_deliveries": sum(row["automatic_delivery"] for row in rows),
        "source_mutations": sum(not row["inputs_unchanged"] for row in rows),
        "nonfinite_renders": sum(row["nonfinite"] for row in rows),
    }
    failures = []
    for name in ("rat_models", "other_models"):
        if metrics[name] < GATES[name]: failures.append(f"{name}<{GATES[name]}")
    if metrics["rat_recall"] < GATES["rat_recall"]: failures.append(f"rat_recall<{GATES['rat_recall']}")
    if metrics["other_false_route"] > GATES["other_false_route"]: failures.append(f"other_false_route>{GATES['other_false_route']}")
    for name in ("automatic_deliveries", "source_mutations", "nonfinite_renders"):
        if metrics[name] != GATES[name]: failures.append(f"{name}!={GATES[name]}")
    return not failures, failures, metrics


def evaluate(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(f"sealed report already exists: {args.output}")
    root, model_root = args.root.resolve(), args.models.resolve()
    head = subprocess.check_output(["git", "-C", str(args.nam), "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "-C", str(args.nam), "rev-parse", "HEAD:nam"], text=True).strip()
    if head != NAM["commit"] or tree != NAM["tree"]:
        raise ValueError("official NAM source revision differs")
    clips, clip_records = [], []
    for relative in CLIPS:
        value, record = clip(root / relative)
        record["path"] = relative
        clips.append(value)
        clip_records.append(record)
    model_records = []
    for scope, name, filename, url, page, license_id in MODELS:
        path = model_root / filename
        document = json.loads(path.read_text())
        model_records.append({
            "scope": scope, "id": name, "filename": filename, "url": url, "page": page,
            "license": license_id, "sha256": digest(path), "bytes": path.stat().st_size,
            "metadata": document.get("metadata") or {},
        })
    lock = {
        "schema": 1, "kind": "third-independent-multi-author-a2-router-seal",
        "gates": GATES, "router_sha256": digest(args.router),
        "identity_package_sha256": digest(args.identity / "package.json"),
        "seal_sha256": digest(Path(__file__)),
        "runtime_sha256": digest(Path(__file__).with_name("a2.py")), "nam": NAM,
        "environment": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "soundfile", "scipy", "joblib", "scikit-learn")},
        "models": model_records, "clips": clip_records,
        "model_files_redistributed": False, "physical_audio_devices_used": False,
        "analysis_resampling": True,
        "retuning_after_lock": False,
    }
    if args.lock.exists():
        if json.loads(args.lock.read_text()) != lock:
            raise ValueError("existing seal lock differs")
    else:
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        args.lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")

    identity = IdentityRuntime(args.identity)
    router = RouterRuntime(args.router, identity)
    init = loader(args.nam, args.deps)
    rows = []
    for record in model_records:
        runtime = A2Runtime(model_root / record["filename"], init)
        for dry, source in zip(clips, clip_records):
            before = _array(dry)
            wet = runtime.render(dry)
            decision = router.infer_pair(dry, wet, RATE)
            row = {
                "scope": record["scope"], "model": record["id"],
                "clip": Path(source["path"]).stem, "routed": decision["decision"] == "candidate",
                "score": decision["score"], "threshold": decision["threshold"],
                "base_candidate": decision["base"]["candidate"], "base_score": decision["base"]["score"],
                "automatic_delivery": decision["automatic_delivery"],
                "inputs_unchanged": before == _array(dry), "nonfinite": int(not np.isfinite(wet).all()),
                "wet_sha256": _array(wet), "wet_peak": float(np.max(np.abs(wet))),
                "wet_rms": float(np.sqrt(np.mean(wet * wet))), "frames": len(wet), "sample_rate": RATE,
            }
            rows.append(row)
            print(json.dumps({"model": record["id"], "clip": row["clip"], "route": row["routed"], "score": row["score"]}), flush=True)
        runtime.assert_unchanged()
    passed, failures, metrics = gate(rows)
    identity.assert_artifacts_unchanged()
    router.assert_artifacts_unchanged()
    files_unchanged = all(digest(root / item["path"]) == item["file_sha256"] for item in clip_records)
    models_unchanged = all(digest(model_root / item["filename"]) == item["sha256"] for item in model_records)
    if not files_unchanged: failures.append("source-file-mutation")
    if not models_unchanged: failures.append("model-file-mutation")
    passed = passed and files_unchanged and models_unchanged
    report = {
        "schema": 1, "status": "accepted-capture-model-seal" if passed else "rejected", "accepted": passed,
        "scope": "untouched multi-author NAM A2 capture models and four-player GuitarSet pickup audio; not raw physical-device recordings",
        "lock_sha256": digest(args.lock), "metrics": metrics, "failures": failures, "rows": rows,
        "artifacts_unchanged": True, "source_files_unchanged": files_unchanged, "model_files_unchanged": models_unchanged,
        "source_audio_modified": False, "physical_audio_devices_used": False,
        "analysis_resampling": True,
        "automatic_normalization": False, "automatic_limiting": False, "automatic_dither": False,
        "lossy_reencoding": False, "intermediate_audio_retained": False, "retuning_after_lock": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--nam", type=Path, required=True)
    parser.add_argument("--deps", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args)
    print(json.dumps({"accepted": report["accepted"], "metrics": report["metrics"], "failures": report["failures"]}))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
