#!/usr/bin/env python3
"""One-shot independent capture-model seal for the frozen RAT route.

This evaluates downloaded Proteus model files by rendering read-only public DI
audio in memory.  It never touches a physical audio device and never exports
audio.  The result is capture-model evidence, not raw physical-device evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .identity import IdentityRuntime
from .packages import digest
from .proteus import ProteusRuntime, RATE
from .route import RouteRuntime


PACK_URL = "https://github.com/GuitarML/ToneLibrary/releases/download/v1.0/Proteus_Tone_Packs.zip"
PACK_SHA256 = "f5737bd873c095a71cc092e40508d780cb9b817b56bf46587e80338ecb877636"
SOURCE = {
    "title": "GuitarML Proteus Tone Packs v1.0",
    "url": PACK_URL,
    "repository": "https://github.com/GuitarML/ToneLibrary",
    "license": "GPL-3.0",
    "role": "evaluation-only capture models; files are not redistributed",
}
MODELS = (
    ("rat", "proco", "Proteus_Tone_Packs/PedalPack1/ProcoRatPedal_HighGain.json"),
    ("rat", "df7", "Proteus_Tone_Packs/PedalPack4/DigitechDF-7_Rat.json"),
    ("rat", "ratsbane", "Proteus_Tone_Packs/PedalPack4/RatsbaneStandardV1.json"),
    ("other", "mt2", "Proteus_Tone_Packs/PedalPack1/BossMT2_PedalHighGain.json"),
    ("other", "muff", "Proteus_Tone_Packs/PedalPack1/LittleBigMuff_HighGainPedal.json"),
    ("other", "ts9", "Proteus_Tone_Packs/PedalPack1/TS9_HighDrive.json"),
    ("other", "wildfire", "Proteus_Tone_Packs/PedalPack2/WildFire_Pedal.json"),
    ("other", "metal", "Proteus_Tone_Packs/PedalPack4/DODAmericanMetal.json"),
    ("other", "ts808", "Proteus_Tone_Packs/PedalPack4/Ibanez808TubeScreamer.json"),
)
CLIPS = (
    "data/corpus/guitar-techs/P1_techniques/P1_techniques/audio/directinput/directinput_Harmonics.wav",
    "data/corpus/guitar-techs/P1_techniques/P1_techniques/audio/directinput/directinput_PalmMute.wav",
    "data/corpus/guitar-techs/P1_techniques/P1_techniques/audio/directinput/directinput_PinchHarmonics.wav",
    "data/corpus/guitar-techs/P1_chords/P1_chords/audio/directinput/directinput_Set1_maj.wav",
)
GATES = {
    "rat_recall": 0.75,
    "other_false_route": 0.10,
    "rat_models": 3,
    "other_models": 6,
    "automatic_deliveries": 0,
    "source_mutations": 0,
    "nonfinite_renders": 0,
}


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _analysis_clip(path: Path, seconds: int = 5) -> tuple[np.ndarray, dict]:
    source, source_rate = sf.read(path, dtype="float32", always_2d=True)
    if source.shape[1] != 1:
        raise ValueError(f"seal source is not mono: {path}")
    mono = source[:, 0]
    if source_rate != RATE:
        common = np.gcd(source_rate, RATE)
        mono = resample_poly(mono, RATE // common, source_rate // common).astype(np.float32)
    frames = seconds * RATE
    if len(mono) < frames:
        mono = np.pad(mono, (0, frames - len(mono)))
        start = 0
    else:
        hop = RATE
        starts = list(range(0, len(mono) - frames + 1, hop))
        final = len(mono) - frames
        if starts[-1] != final:
            starts.append(final)
        start = min(starts, key=lambda value: (-float(np.mean(mono[value : value + frames] ** 2)), value))
        mono = mono[start : start + frames]
    value = np.asarray(mono, dtype=np.float32)
    return value, {
        "path": str(path),
        "file_sha256": digest(path),
        "source_rate": int(source_rate),
        "source_channels": int(source.shape[1]),
        "analysis_rate": RATE,
        "analysis_start": int(start),
        "analysis_frames": len(value),
        "analysis_sha256": _hash_array(value),
        "analysis_peak": float(np.max(np.abs(value))),
        "analysis_rms": float(np.sqrt(np.mean(value * value))),
    }


def gate(rows: list[dict]) -> tuple[bool, list[str], dict]:
    rat = [row for row in rows if row["scope"] == "rat"]
    other = [row for row in rows if row["scope"] == "other"]
    metrics = {
        "rat_examples": len(rat),
        "other_examples": len(other),
        "rat_models": len({row["model"] for row in rat}),
        "other_models": len({row["model"] for row in other}),
        "rat_recall": sum(row["routed"] for row in rat) / max(len(rat), 1),
        "other_false_route": sum(row["routed"] for row in other) / max(len(other), 1),
        "automatic_deliveries": sum(row["automatic_delivery"] for row in rows),
        "source_mutations": sum(not row["inputs_unchanged"] for row in rows),
        "nonfinite_renders": sum(row["nonfinite"] for row in rows),
    }
    failures = []
    for key in ("rat_models", "other_models"):
        if metrics[key] < GATES[key]:
            failures.append(f"{key}<{GATES[key]}")
    if metrics["rat_recall"] < GATES["rat_recall"]:
        failures.append(f"rat_recall<{GATES['rat_recall']}")
    if metrics["other_false_route"] > GATES["other_false_route"]:
        failures.append(f"other_false_route>{GATES['other_false_route']}")
    for key in ("automatic_deliveries", "source_mutations", "nonfinite_renders"):
        if metrics[key] != GATES[key]:
            failures.append(f"{key}!={GATES[key]}")
    return not failures, failures, metrics


def evaluate(
    root: Path,
    archive: Path,
    identity_package: Path,
    route_model: Path,
    lock_path: Path,
    output: Path,
) -> dict:
    if output.exists():
        raise FileExistsError(f"sealed report already exists: {output}")
    root, archive = root.resolve(), archive.resolve()
    if digest(archive) != PACK_SHA256:
        raise ValueError("Proteus pack digest differs")
    clip_values = []
    clip_records = []
    for relative in CLIPS:
        value, record = _analysis_clip(root / relative)
        record["path"] = relative
        clip_values.append(value)
        clip_records.append(record)
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        if any(entry not in names for _, _, entry in MODELS):
            raise ValueError("Proteus pack is missing a sealed model")
        model_bytes = {entry: bundle.read(entry) for _, _, entry in MODELS}
    lock = {
        "schema": 1,
        "kind": "independent-capture-model-route-seal",
        "source": {**SOURCE, "sha256": PACK_SHA256},
        "gates": GATES,
        "route_sha256": digest(route_model),
        "identity_package_sha256": digest(identity_package / "package.json"),
        "models": [
            {"scope": scope, "id": name, "entry": entry, "sha256": _hash_bytes(model_bytes[entry])}
            for scope, name, entry in MODELS
        ],
        "clips": clip_records,
        "physical_audio_devices_used": False,
        "retuning_after_lock": False,
    }
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != lock:
            raise ValueError("existing seal lock differs")
    else:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")

    identity = IdentityRuntime(identity_package)
    route = RouteRuntime(route_model, identity)
    route_before = digest(route_model)
    rows = []
    for scope, name, entry in MODELS:
        renderer = ProteusRuntime.bytes(model_bytes[entry])
        for clip, clip_record in zip(clip_values, clip_records):
            dry_before = _hash_array(clip)
            wet = renderer.render(clip)
            decision = route.infer_pair(clip, wet, RATE)
            row = {
                "scope": scope,
                "model": name,
                "clip": Path(clip_record["path"]).stem,
                "routed": decision["decision"] == "candidate",
                "score": decision["score"],
                "threshold": decision["threshold"],
                "base_candidate": decision["base"]["candidate"],
                "base_score": decision["base"]["score"],
                "automatic_delivery": decision["automatic_delivery"],
                "inputs_unchanged": dry_before == _hash_array(clip),
                "nonfinite": int(not np.isfinite(wet).all()),
                "wet_sha256": _hash_array(wet),
                "wet_peak": float(np.max(np.abs(wet))),
                "wet_rms": float(np.sqrt(np.mean(wet * wet))),
                "frames": len(wet),
                "sample_rate": RATE,
            }
            rows.append(row)
            print(json.dumps({"model": name, "clip": row["clip"], "route": row["routed"], "score": row["score"]}), flush=True)
    passed, failures, metrics = gate(rows)
    identity.assert_artifacts_unchanged()
    artifacts_unchanged = route_before == digest(route_model)
    if not artifacts_unchanged:
        passed = False
        failures.append("route-artifact-mutation")
    source_files_unchanged = all(digest(root / item["path"]) == item["file_sha256"] for item in clip_records)
    if not source_files_unchanged:
        passed = False
        failures.append("source-file-mutation")
    report = {
        "schema": 1,
        "status": "accepted-capture-model-seal" if passed else "rejected",
        "accepted": passed,
        "scope": "independently published capture models; not raw physical-device recordings",
        "lock_sha256": digest(lock_path),
        "metrics": metrics,
        "failures": failures,
        "rows": rows,
        "artifacts_unchanged": artifacts_unchanged,
        "source_files_unchanged": source_files_unchanged,
        "source_audio_modified": False,
        "physical_audio_devices_used": False,
        "automatic_normalization": False,
        "automatic_limiting": False,
        "automatic_dither": False,
        "lossy_reencoding": False,
        "intermediate_audio_retained": False,
        "retuning_after_lock": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.root, args.archive, args.identity, args.route, args.lock, args.output)
    print(json.dumps({"accepted": report["accepted"], "metrics": report["metrics"], "failures": report["failures"]}))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

