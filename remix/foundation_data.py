"""Read-only admission and pairing for the first restoration foundation pilots.

This module deliberately knows about datasets, not model architectures.  It
turns each admitted source into the same Wet/Predecessor pair contract and
refuses split leakage before training can start.  It never opens an audio input
device and never writes derived audio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile
from scipy.signal import resample_poly

from .asrnn_effects import EFFECT_SPECS, audit_effect, effect_files, read_effect_pair


RATE = 48_000
LOSSLESS = {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}
FOUNDATION_SPLITS = ("fit", "calibration", "development", "locked-final")


@dataclass(frozen=True)
class Pair:
    id: str
    mechanism: str
    family: str
    device: str
    split: str
    group: str
    source: str
    wet: str
    origin: str
    gradient_scope: str
    weight_scope: str
    offset_seconds: float = 0.0


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def _geometry(path: Path) -> dict:
    info = soundfile.info(path)
    if info.subtype not in LOSSLESS:
        raise ValueError(f"lossy or unsupported audio subtype: {path}: {info.subtype}")
    if info.frames <= 0 or info.channels <= 0 or info.samplerate <= 0:
        raise ValueError(f"invalid audio geometry: {path}")
    return {
        "frames": info.frames,
        "channels": info.channels,
        "sample_rate": info.samplerate,
        "subtype": info.subtype,
    }


def _assert_group_disjoint(pairs: Iterable[Pair]) -> None:
    owners: dict[str, str] = {}
    for pair in pairs:
        previous = owners.setdefault(pair.group, pair.split)
        if previous != pair.split:
            raise ValueError(
                f"performance group crosses foundation splits: {pair.group}: "
                f"{previous}/{pair.split}"
            )


def asrnn_pairs(root: Path) -> tuple[list[Pair], dict]:
    """Admit ASRNN as fit/development only.

    Its official train/eval programs are disjoint, but it has no independent
    calibration or locked-final session.  Inventing extra splits from control
    settings would leak the repeated Dry performance, so this adapter does not.
    """

    pairs: list[Pair] = []
    reports = {}
    mapping = {
        "rat": ("nonlinear", "distortion"),
        "dfz": ("nonlinear", "fuzz"),
        "cs3": ("dynamics", "compressor"),
    }
    for device, (mechanism, family) in mapping.items():
        report = audit_effect(root, device, scan_audio=True)
        reports[device] = report
        for upstream, split in (("train", "fit"), ("eval", "development")):
            # The same Dry program is repeated across settings inside each
            # official split.  One group per device/program prevents a future
            # trainer from pretending those settings are independent sessions.
            group = f"asrnn:{device}:{upstream}"
            for path in effect_files(root, device, upstream):
                pairs.append(
                    Pair(
                        id=f"asrnn-{device}-{upstream}-{_digest(path.name)}",
                        mechanism=mechanism,
                        family=family,
                        device=EFFECT_SPECS[device].display_name,
                        split=split,
                        group=group,
                        source=str(path.resolve()),
                        wet=str(path.resolve()),
                        origin="asrnn-stereo-dry-wet",
                        gradient_scope="internal-research-only",
                        weight_scope="blocked-from-product-by-cc-by-nc-4.0",
                    )
                )
    _assert_group_disjoint(pairs)
    return pairs, {
        "source": "asrnn-physical-effects",
        "admitted_for_research": True,
        "admitted_for_product_gradients": False,
        "weight_release": "blocked-by-cc-by-nc-4.0",
        "scope": "internal non-commercial physical pilot",
        "split_mapping": {"train": "fit", "eval": "development"},
        "calibration_available": False,
        "locked_final_available": False,
        "devices": reports,
    }


def apple_pairs(root: Path, corpus_root: Path) -> tuple[list[Pair], dict]:
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema") != 1 or payload.get("role") != "real plugin wet captures; split follows the original performance group":
        raise ValueError(f"unexpected Apple AU manifest contract: {manifest_path}")
    renderer = payload.get("renderer", {})
    if renderer.get("format") != "Audio Unit v2 offline render":
        raise ValueError("Apple AU captures were not produced by the admitted offline renderer")

    split_map = {
        "train": "fit",
        "valid": "calibration",
        "calibrate": "development",
        "test": "locked-final",
    }
    mechanism = {"drive": "nonlinear", "delay": "temporal", "reverb": "temporal"}
    pairs: list[Pair] = []
    counts = Counter()
    geometry = Counter()
    padded = Counter()
    maximum_padding_seconds = 0.0
    source_owners: dict[str, str] = {}
    for record in payload.get("captures", []):
        label = record.get("label")
        if label not in mechanism or record.get("split") not in split_map:
            raise ValueError(f"unsupported Apple AU capture record: {record}")
        split = split_map[record["split"]]
        wet = (root / record["path"]).resolve()
        source = Path(record["source_path"]).resolve()
        if not _inside(wet, root) or not _inside(source, corpus_root):
            raise ValueError(f"Apple AU pair escapes admitted corpus roots: {record}")
        if not wet.is_file() or not source.is_file():
            raise FileNotFoundError(f"missing Apple AU pair member: {wet} / {source}")
        wet_info, source_info = _geometry(wet), _geometry(source)
        seconds = wet_info["frames"] / wet_info["sample_rate"]
        available = source_info["frames"] / source_info["sample_rate"] - float(record["source_offset"])
        # The historical offline renderer zero-padded a selected five-second
        # source window when it reached EOF.  Preserve that exact predecessor
        # geometry and make the padded subset visible in the audit.
        padding_seconds = max(0.0, seconds - available)
        if padding_seconds > 0.0:
            padded[(split, label)] += 1
            maximum_padding_seconds = max(maximum_padding_seconds, padding_seconds)
        group = str(record["source_group"])
        previous = source_owners.setdefault(group, split)
        if previous != split:
            raise ValueError(f"Apple AU source group crosses splits: {group}: {previous}/{split}")
        component = str(record["component"]["name"])
        pairs.append(
            Pair(
                id=f"apple-{label}-{_digest(record['path'])}",
                mechanism=mechanism[label],
                family=label,
                device=component,
                split=split,
                group=f"apple:{group}",
                source=str(source),
                wet=str(wet),
                origin="apple-au-external-clean",
                gradient_scope="internal-research-only",
                weight_scope="blocked-until-written-output-and-weight-clearance",
                offset_seconds=float(record["source_offset"]),
            )
        )
        counts[(split, label)] += 1
        geometry[(wet_info["sample_rate"], wet_info["channels"], wet_info["subtype"])] += 1
    _assert_group_disjoint(pairs)
    if not pairs:
        raise ValueError("Apple AU manifest contains no admitted captures")
    return pairs, {
        "source": "apple-au-local",
        "admitted_for_research": True,
        "admitted_for_product_gradients": False,
        "weight_release": "blocked-until-written-output-and-weight-clearance",
        "scope": "internal research; derived-weight release needs review",
        "captures": len(pairs),
        "counts": {f"{split}/{label}": count for (split, label), count in sorted(counts.items())},
        "source_groups": len(source_owners),
        "geometry": {
            f"{rate}Hz/{channels}ch/{subtype}": count
            for (rate, channels, subtype), count in sorted(geometry.items())
        },
        "zero_padded_predecessors": {
            "captures": sum(padded.values()),
            "maximum_seconds": maximum_padding_seconds,
            "counts": {
                f"{split}/{label}": count
                for (split, label), count in sorted(padded.items())
            },
            "side": "right",
        },
        "split_mapping": split_map,
    }


def aachen_audit(root: Path) -> dict:
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and "brir" in "/".join(part.lower() for part in path.parts)
    )
    if len(paths) < 8:
        raise ValueError(f"expected at least eight measured Aachen BRIRs, found {len(paths)}")
    geometries = Counter()
    silent = []
    for path in paths:
        info = _geometry(path)
        geometries[(info["sample_rate"], info["channels"], info["subtype"])] += 1
        with soundfile.SoundFile(path) as stream:
            probe = stream.read(min(info["frames"], info["sample_rate"] * 2), dtype="float32", always_2d=True)
        if not np.isfinite(probe).all() or float(np.max(np.abs(probe))) <= 1.0e-8:
            silent.append(str(path))
    if silent:
        raise ValueError(f"invalid or silent Aachen RIRs: {silent[:3]}")
    boundaries = (int(len(paths) * 0.60), int(len(paths) * 0.76), int(len(paths) * 0.88))
    parts = {
        "fit": paths[: boundaries[0]],
        "calibration": paths[boundaries[0] : boundaries[1]],
        "development": paths[boundaries[1] : boundaries[2]],
        "locked-final": paths[boundaries[2] :],
    }
    digests = {name: {_digest(str(path.resolve())) for path in values} for name, values in parts.items()}
    for left in FOUNDATION_SPLITS:
        for right in FOUNDATION_SPLITS:
            if left < right and digests[left] & digests[right]:
                raise ValueError(f"Aachen RIR split overlap: {left}/{right}")
    return {
        "source": "aachen-chapel-rir",
        "admitted_for_research": True,
        "admitted_for_product_gradients": True,
        "weight_release": "allowed-with-cc-by-4.0-attribution",
        "scope": "CC-BY-4.0 ephemeral reverb pair generation",
        "rirs": len(paths),
        "splits": {name: len(values) for name, values in parts.items()},
        "geometry": {
            f"{rate}Hz/{channels}ch/{subtype}": count
            for (rate, channels, subtype), count in sorted(geometries.items())
        },
        "derived_audio_written": False,
    }


def inventory(workspace: Path) -> tuple[list[Pair], dict]:
    corpus = (workspace / "data/corpus").resolve()
    asrnn, asrnn_report = asrnn_pairs(corpus / "asrnn-physical-effects")
    apple, apple_report = apple_pairs(corpus / "apple-au", corpus)
    aachen_report = aachen_audit(corpus / "aachen-chapel-rir")
    pairs = asrnn + apple
    _assert_group_disjoint(pairs)
    coverage = defaultdict(lambda: Counter())
    for pair in pairs:
        coverage[pair.mechanism][pair.split] += 1
    report = {
        "schema": 1,
        "status": "research-audit-only",
        "product_training_authorized": False,
        "product_training_pair_count": 0,
        "research_pair_count": len(pairs),
        "source_audio_read_only": True,
        "derived_audio_written": False,
        "physical_audio_devices_used": False,
        "pair_count": len(pairs),
        "coverage": {
            mechanism: {split: counts.get(split, 0) for split in FOUNDATION_SPLITS}
            for mechanism, counts in sorted(coverage.items())
        },
        "sources": [asrnn_report, apple_report, aachen_report],
        "limitations": [
            "all materialized aligned pairs in this inventory are isolated from product gradients",
            "ASRNN has no independent calibration or locked-final session",
            "Apple AU is software-domain evidence and weight release needs review",
            "Aachen is product-eligible only when paired ephemerally with separately product-eligible clean programs and attributed",
            "no current physical modulation, spectral, pitch, delay or reverb paired pilot",
        ],
    }
    return pairs, report


def _resample(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == RATE:
        return audio.astype(np.float32, copy=False)
    divisor = int(np.gcd(source_rate, RATE))
    return resample_poly(audio, RATE // divisor, source_rate // divisor).astype(np.float32)


def read_pair(pair: Pair) -> tuple[np.ndarray, np.ndarray]:
    """Return finite mono ``(wet, predecessor)`` at 48 kHz without normalization."""

    if pair.origin == "asrnn-stereo-dry-wet":
        device = {"ProCo RAT": "rat", "Darkglass Duality Fuzz": "dfz", "Boss CS-3": "cs3"}[pair.device]
        clean, wet, _ = read_effect_pair(Path(pair.wet), device)
        return wet, clean
    if pair.origin != "apple-au-external-clean":
        raise ValueError(f"unsupported pair origin: {pair.origin}")
    wet, wet_rate = soundfile.read(pair.wet, dtype="float32", always_2d=True)
    wet = wet.mean(axis=1)
    with soundfile.SoundFile(pair.source) as stream:
        stream.seek(round(pair.offset_seconds * stream.samplerate))
        frames = round(len(wet) * stream.samplerate / wet_rate)
        clean = stream.read(frames, dtype="float32", always_2d=True).mean(axis=1)
        clean_rate = stream.samplerate
        if len(clean) < frames:
            clean = np.pad(clean, (0, frames - len(clean)))
    wet, clean = _resample(wet, wet_rate), _resample(clean, clean_rate)
    length = min(len(wet), len(clean))
    wet, clean = wet[:length], clean[:length]
    if length < RATE or not np.isfinite(wet).all() or not np.isfinite(clean).all():
        raise ValueError(f"invalid paired audio: {pair.id}")
    return wet, clean


def write_inventory(workspace: Path, output: Path) -> dict:
    pairs, report = inventory(workspace)
    output.mkdir(parents=True, exist_ok=True)
    (output / "pairs.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "training_authorized": False,
                "purpose": "isolated research benchmarks only",
                "pairs": [asdict(pair) for pair in pairs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/data"))
    args = parser.parse_args()
    print(json.dumps(write_inventory(args.workspace.resolve(), args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
