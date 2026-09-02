#!/usr/bin/env python3
"""One-shot independent multi-author NAM A2 seal for router model3.

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
    ("rat", "rat09", "rat09.nam", "https://api.tone3000.com/storage/v1/object/public/models/624b436a59ebbf3e_a2.nam", "https://www.tone3000.com/tones/proco-rat-vintage-reissue-2298", "T3K"),
    ("rat", "rat11", "rat11.nam", "https://api.tone3000.com/storage/v1/object/public/models/ed667a2b4a59cf65_a2.nam", "https://www.tone3000.com/tones/proco-rat-vintage-reissue-2298", "T3K"),
    ("rat", "rat2d2", "rat2d2.nam", "https://api.tone3000.com/storage/v1/object/public/models/4c7beo6djdx.nam", "https://www.tone3000.com/tones/proco-rat-2-82634", "T3K"),
    ("rat", "rat2d6", "rat2d6.nam", "https://api.tone3000.com/storage/v1/object/public/models/7hngx1rpulo.nam", "https://www.tone3000.com/tones/proco-rat-2-82634", "T3K"),
    ("other", "diy", "diy.nam", "https://api.tone3000.com/storage/v1/object/public/models/52fb8bd98c2b3afa_a2.nam", "https://www.tone3000.com/tones/diy-drive-pedal-bjt-silicon-celestion-eight-15-ir-5700", "CC0-1.0"),
    ("other", "grind1", "grind1.nam", "https://api.tone3000.com/storage/v1/object/public/models/b95c836b3165aa05_a2.nam", "https://www.tone3000.com/tones/dna-klirrton-grindstein-2620", "CC0-1.0"),
    ("other", "grind2", "grind2.nam", "https://api.tone3000.com/storage/v1/object/public/models/e40b39aa1b99bff8_a2.nam", "https://www.tone3000.com/tones/dna-klirrton-grindstein-2620", "CC0-1.0"),
    ("other", "grind3", "grind3.nam", "https://api.tone3000.com/storage/v1/object/public/models/b1dea544c5388d34_a2.nam", "https://www.tone3000.com/tones/dna-klirrton-grindstein-2620", "CC0-1.0"),
)
CLIPS = (
    "data/corpus/guitar-techs/P2_chords/P2_chords/audio/directinput/directinput_Set1_aug.wav",
    "data/corpus/guitar-techs/P2_chords/P2_chords/audio/directinput/directinput_Set2_min.wav",
    "data/corpus/guitar-techs/P2_chords/P2_chords/audio/directinput/directinput_Set3_maj.wav",
    "data/corpus/guitar-techs/P2_chords/P2_chords/audio/directinput/directinput_Set4_dim.wav",
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
    if rate != RATE or source.shape[1] != 2 or not np.array_equal(source[:, 0], source[:, 1]):
        raise ValueError(f"seal requires exact dual-mono 48 kHz DI: {path}")
    mono = source[:, 0]
    frames = seconds * RATE
    starts = list(range(0, len(mono) - frames + 1, RATE))
    if not starts:
        raise ValueError(f"seal DI is too short: {path}")
    start = min(starts, key=lambda item: (-float(np.mean(mono[item:item + frames] ** 2)), item))
    value = mono[start:start + frames].copy()
    return value, {
        "path": path.as_posix(), "file_sha256": digest(path), "source_rate": rate,
        "source_channels": 2, "channel_policy": "exact-dual-mono-left",
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
        "schema": 1, "kind": "independent-multi-author-a2-router-seal",
        "gates": GATES, "router_sha256": digest(args.router),
        "identity_package_sha256": digest(args.identity / "package.json"),
        "seal_sha256": digest(Path(__file__)),
        "runtime_sha256": digest(Path(__file__).with_name("a2.py")), "nam": NAM,
        "environment": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "soundfile", "scipy", "joblib", "scikit-learn")},
        "models": model_records, "clips": clip_records,
        "model_files_redistributed": False, "physical_audio_devices_used": False,
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
        "scope": "untouched multi-author NAM A2 capture models and independent-performer DI; not raw physical-device recordings",
        "lock_sha256": digest(args.lock), "metrics": metrics, "failures": failures, "rows": rows,
        "artifacts_unchanged": True, "source_files_unchanged": files_unchanged, "model_files_unchanged": models_unchanged,
        "source_audio_modified": False, "physical_audio_devices_used": False,
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
