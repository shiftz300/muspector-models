#!/usr/bin/env python3
"""Lock and seal the chain on device-disjoint IDMT guitar pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile
import torch

from .chain import ChainRuntime
from .family import FRAMES, RATE


TARGETS = ("clean", "drive", "delay", "reverb")
EFFECTS = {
    "drive": ("Distortion", "Overdrive"),
    "delay": ("FeedbackDelay", "SlapbackDelay"),
    "reverb": ("Reverb",),
}
GATES = {
    "coverage": 0.35,
    "accepted_exact": 0.98,
    "target_accepted": 4,
    "target_exact": 0.95,
    "input_mutations": 0,
    "artifact_mutations": 0,
    "runtime_errors": 0,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def spread(rows: list[dict], count: int) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (row["group"], row["wet"]))
    unique = []
    seen = set()
    for row in ordered:
        if row["group"] not in seen:
            unique.append(row)
            seen.add(row["group"])
    if len(unique) < count:
        raise ValueError(f"not enough device-disjoint groups: {len(unique)} < {count}")
    indices = np.linspace(0, len(unique) - 1, count).round().astype(int)
    return [unique[int(index)] for index in indices]


def select(pairs: list[dict], per_target: int, split: str = "test") -> list[dict]:
    test = [row for row in pairs if row["split"] == split]
    selected = []
    clean_sources = {}
    for row in test:
        if row["target"] in EFFECTS:
            clean_sources.setdefault(row["group"], row)
    for row in spread(list(clean_sources.values()), per_target):
        selected.append({**row, "target": "clean", "effect": "NoFX", "wet": row["dry"]})
    for target, effects in EFFECTS.items():
        base, remainder = divmod(per_target, len(effects))
        for position, effect in enumerate(effects):
            count = base + int(position < remainder)
            selected.extend(spread(
                [row for row in test if row["target"] == target and row["effect"] == effect],
                count,
            ))
    if len(selected) != per_target * len(TARGETS):
        raise ValueError("IDMT seal selection geometry changed")
    return sorted(selected, key=lambda row: (TARGETS.index(row["target"]), row["effect"], row["group"]))


class Dataset(torch.utils.data.Dataset):
    """Balanced aligned IDMT pairs for gate training only."""

    def __init__(self, corpus: Path, pairs: Path, split: str, per_target: int):
        payload = json.loads(pairs.read_text())
        self.corpus = corpus
        self.records = select(payload["pairs"], per_target, split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        target = record["target"]
        topology = [-1, -1, -1]
        if target != "clean":
            topology[0] = ("drive", "delay", "reverb").index(target)
        return {
            "dry": torch.from_numpy(audio(self.corpus / record["dry"])),
            "wet": torch.from_numpy(audio(self.corpus / record["wet"])),
            "topology": torch.tensor(topology, dtype=torch.long),
            "controls": torch.zeros(9, dtype=torch.float32),
            "control_mask": torch.zeros(9, dtype=torch.float32),
            "order": torch.zeros(3, dtype=torch.float32),
            "order_mask": torch.zeros(3, dtype=torch.float32),
        }


def lock(args) -> dict:
    if args.output.exists():
        raise FileExistsError(f"IDMT lock already exists: {args.output}")
    payload = json.loads(args.pairs.read_text())
    if payload.get("schema") != 1 or payload.get("doi") != "10.5281/zenodo.7544032":
        raise ValueError("unsupported IDMT aligned-pair manifest")
    rows = select(payload["pairs"], args.per_target)
    frozen = []
    for row in rows:
        dry, wet = args.corpus / row["dry"], args.corpus / row["wet"]
        frozen.append({
            "target": row["target"],
            "effect": row["effect"],
            "group": row["group"],
            "instrument_setting": row["instrument_setting"],
            "dry": row["dry"],
            "wet": row["wet"],
            "dry_sha256": digest(dry),
            "wet_sha256": digest(wet),
        })
    report = {
        "schema": 1,
        "status": "locked-unopened",
        "source": "IDMT-SMT-Audio-Effects",
        "doi": payload["doi"],
        "license": payload["license"],
        "split": "device-9-test",
        "selection": "effect-stratified, unique note groups, evenly spaced before inference",
        "per_target": args.per_target,
        "targets": list(TARGETS),
        "pairs_manifest_sha256": digest(args.pairs),
        "records": frozen,
        "candidate_exposure": "excluded from active family, gate, order, and knob candidates",
        "physical_audio_devices_used": False,
        "source_audio_modified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def audio(path: Path) -> np.ndarray:
    value, rate = soundfile.read(path, dtype="float32", always_2d=True)
    if rate != RATE:
        raise ValueError(f"IDMT seal requires {RATE} Hz input: {path}")
    mono = value.mean(axis=1, dtype=np.float32)
    if len(mono) >= FRAMES:
        return np.asarray(mono[:FRAMES], dtype=np.float32)
    return np.pad(mono, (0, FRAMES - len(mono))).astype(np.float32, copy=False)


def metrics(rows: list[dict]) -> dict:
    accepted = [row for row in rows if row["accepted"]]
    return {
        "examples": len(rows),
        "accepted": len(accepted),
        "coverage": len(accepted) / max(len(rows), 1),
        "accepted_exact": sum(row["exact"] for row in accepted) / max(len(accepted), 1),
        "accepted_errors": sum(not row["exact"] for row in accepted),
        "runtime_errors": sum(not row["runtime_valid"] for row in rows),
        "input_mutations": sum(not row["inputs_unchanged"] for row in rows),
    }


def seal(args) -> dict:
    if args.output.exists():
        raise FileExistsError(f"IDMT seal already exists: {args.output}")
    locked_bytes = args.lock.read_bytes()
    locked = json.loads(locked_bytes)
    if locked.get("status") != "locked-unopened":
        raise ValueError("IDMT source was not locked before inference")
    runtime = ChainRuntime(
        args.family, args.manifest, args.gate, args.order, args.bundle, args.evidence
    )
    rows = []
    for index, record in enumerate(locked["records"]):
        dry_path, wet_path = args.corpus / record["dry"], args.corpus / record["wet"]
        if digest(dry_path) != record["dry_sha256"] or digest(wet_path) != record["wet_sha256"]:
            raise ValueError("locked IDMT source changed")
        dry, wet = audio(dry_path), audio(wet_path)
        before = hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest()
        result = runtime.infer(dry, wet)
        accepted = result["decision"] != "abstain"
        predicted = result["active"]
        truth = [] if record["target"] == "clean" else [record["target"]]
        knobs = result["knobs"]
        runtime_valid = (
            result["quality"]["analysis_only"]
            and not result["quality"]["physical_audio_devices_used"]
            and (not accepted or not predicted or (knobs is not None and knobs["active"] == predicted))
        )
        rows.append({
            "index": index,
            "target": record["target"],
            "effect": record["effect"],
            "group": record["group"],
            "accepted": accepted,
            "confidence": result["gate"]["confidence"],
            "predicted": predicted,
            "exact": predicted == truth,
            "runtime_valid": bool(runtime_valid),
            "inputs_unchanged": before == hashlib.sha256(dry.tobytes() + wet.tobytes()).hexdigest(),
        })
        if (index + 1) % 8 == 0 or index + 1 == len(locked["records"]):
            print(json.dumps({"done": index + 1, "total": len(locked["records"]), "target": record["target"]}), flush=True)
    runtime.assert_artifacts_unchanged()
    for record in locked["records"]:
        if digest(args.corpus / record["dry"]) != record["dry_sha256"] or digest(args.corpus / record["wet"]) != record["wet_sha256"]:
            raise ValueError("IDMT source changed during seal")
    overall = metrics(rows)
    domains = {target: metrics([row for row in rows if row["target"] == target]) for target in TARGETS}
    failures = []
    if overall["coverage"] < GATES["coverage"]: failures.append("coverage")
    if overall["accepted_exact"] < GATES["accepted_exact"]: failures.append("accepted-exact")
    if overall["runtime_errors"] > GATES["runtime_errors"]: failures.append("runtime-error")
    if overall["input_mutations"] > GATES["input_mutations"]: failures.append("input-mutation")
    for target, values in domains.items():
        if values["accepted"] < GATES["target_accepted"]: failures.append(f"{target}.accepted")
        if values["accepted_exact"] < GATES["target_exact"]: failures.append(f"{target}.exact")
    report = {
        "schema": 1,
        "status": "accepted-sealed-research" if not failures else "rejected",
        "accepted": not failures,
        "source": "IDMT device-9 test",
        "selection_role": "independent for current chain candidates; older archived IDMT experiments are unrelated",
        "license_boundary": "CC-BY-NC-ND research-only; no audio or derived weights redistributed",
        "lock_sha256": hashlib.sha256(locked_bytes).hexdigest(),
        "gates": GATES,
        "metrics": overall,
        "domains": domains,
        "failures": failures,
        "capabilities": {
            "family": "sealed exact-set recognition with abstention",
            "order": "runtime safety only; source contains single effects",
            "knobs": "runtime safety only; source has no exact normalized controls",
        },
        "rows": rows,
        "artifacts_unchanged": True,
        "source_audio_modified": False,
        "physical_audio_devices_used": False,
        "automatic_normalization": False,
        "automatic_limiting": False,
        "lossy_reencoding": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    locking = commands.add_parser("lock")
    locking.add_argument("--pairs", type=Path, required=True)
    locking.add_argument("--corpus", type=Path, required=True)
    locking.add_argument("--per-target", type=int, default=32)
    locking.add_argument("--output", type=Path, required=True)
    sealing = commands.add_parser("seal")
    sealing.add_argument("--lock", type=Path, required=True)
    sealing.add_argument("--corpus", type=Path, required=True)
    sealing.add_argument("--family", type=Path, required=True)
    sealing.add_argument("--manifest", type=Path, required=True)
    sealing.add_argument("--gate", type=Path, required=True)
    sealing.add_argument("--order", type=Path, required=True)
    sealing.add_argument("--bundle", type=Path, required=True)
    sealing.add_argument("--evidence", type=Path, required=True)
    sealing.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = lock(args) if args.mode == "lock" else seal(args)
    print(json.dumps({"status": report["status"], "accepted": report.get("accepted"), "failures": report.get("failures", [])}))
    if args.mode == "seal" and not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
